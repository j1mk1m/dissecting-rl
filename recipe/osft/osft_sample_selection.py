# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""On-policy SFT reward processing: sample selection and advantage computation.

Three training modes are supported, selected via ``reward_baseline``:

  "none"  (default, original OSFT):
    Binary filter per uid — keep only mixed-correctness groups (0 < C < N).
    Optionally keep incorrect rollouts as negatives with weight ``-negative_sample_loss_scale``.

  "mean"  (baseline-subtracted):
    Subtract the group mean reward from each sample's reward.
    Samples with zero advantage (degenerate all-correct / all-incorrect groups) are dropped.
    Weight = reward - group_mean.

  "mean" + reward_normalize_std=True  (GRPO):
    Same as "mean" but also divide by the group standard deviation.
    Weight = (reward - group_mean) / (group_std + std_eps).
"""

from __future__ import annotations

import torch

from verl import DataProto

# Per-sequence weight w: loss uses (-log_prob * w) aggregated over tokens.
# w=+1  → standard CE on the completion (encourage).
# w<0   → minimise w*(-log_prob) (discourage, scaled by |w|).
# w=0   → sample contributes nothing to the loss (filtered out effectively).
SAMPLE_WEIGHT_KEY = "sample_weight"

# Backward-compatibility alias used in older checkpoints / configs.
OSFT_LOSS_SIGN_KEY = SAMPLE_WEIGHT_KEY


def apply_reward_processing(
    batch: DataProto,
    *,
    reward_baseline: str = "none",
    reward_normalize_std: bool = False,
    reward_std_eps: float = 1e-8,
    enable_negative_sample_training: bool = False,
    negative_sample_loss_scale: float = 1.0,
    dp_world_size: int = 1,
    rollout_n: int | None = None,
) -> DataProto:
    """Filter and weight rollouts for on-policy training.

    Args:
        batch: DataProto with ``token_level_scores``, ``response_mask``, and
            ``uid`` in ``non_tensor_batch``.
        reward_baseline: How to compute the per-sample weight from the raw reward.
            ``"none"``  — binary OSFT filter (original behaviour).
            ``"mean"``  — subtract group mean reward.
        reward_normalize_std: When True (and ``reward_baseline="mean"``), also
            divide by the group standard deviation → GRPO.
        reward_std_eps: Small value added to the group std for numerical stability.
        enable_negative_sample_training: (Only for ``reward_baseline="none"``)
            If True, also keep incorrect rollouts for groups that have kept
            positive samples; weights are ``-negative_sample_loss_scale``.
        negative_sample_loss_scale: Scale of the negative weight (``reward_baseline="none"`` only).
        dp_world_size: Number of data-parallel ranks; batch is trimmed to a
            multiple of this value.
        rollout_n: Expected number of rollouts per prompt. Used by the "none"
            baseline to detect all-correct groups even when the batch was already
            trimmed.  Pass ``None`` to infer from the data.

    Returns:
        A (possibly smaller) ``DataProto`` with ``SAMPLE_WEIGHT_KEY`` set in
        ``batch`` when any sample has a non-unit weight.
    """
    if reward_baseline == "none":
        return _apply_osft_filter(
            batch,
            enable_negative_sample_training=enable_negative_sample_training,
            negative_sample_loss_scale=negative_sample_loss_scale,
            dp_world_size=dp_world_size,
            rollout_n=rollout_n,
        )
    elif reward_baseline == "mean":
        return _apply_advantage_weighting(
            batch,
            normalize_std=reward_normalize_std,
            std_eps=reward_std_eps,
            dp_world_size=dp_world_size,
        )
    else:
        raise ValueError(
            f"Unknown reward_baseline={reward_baseline!r}. Choose 'none' or 'mean'."
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _apply_osft_filter(
    batch: DataProto,
    *,
    enable_negative_sample_training: bool,
    negative_sample_loss_scale: float,
    dp_world_size: int,
    rollout_n: int | None,
) -> DataProto:
    """Original OSFT binary-filtering logic (unchanged behaviour).

    Positive rule per uid:
      C = count with sequence score > 0;  N = rollout count
      C == 0 → drop entire uid
      C == N → skip entire uid (all-correct groups are not trained)
      0 < C < N → keep all correct samples

    When ``enable_negative_sample_training`` is True, also keep incorrect
    rollouts for uids that have at least one kept positive.  Each row gets
    ``SAMPLE_WEIGHT_KEY``: ``+1.0`` for positives, ``-negative_sample_loss_scale``
    for negatives.
    """
    uids = [str(u) for u in batch.non_tensor_batch["uid"]]
    scores = batch.batch["token_level_scores"].sum(-1)

    per_uid: dict[str, list[tuple[int, float]]] = {}
    for i, uid in enumerate(uids):
        per_uid.setdefault(uid, []).append((i, float(scores[i].item())))

    positive_indices: list[int] = []
    for uid, lst in per_uid.items():
        correct = [idx for (idx, s) in lst if s > 0]
        c = len(correct)
        if c == 0:
            continue
        n = rollout_n if rollout_n is not None else len(lst)
        if c == n:
            continue
        positive_indices.extend(correct)

    if not positive_indices:
        return batch.select_idxs([])

    positive_set = set(positive_indices)

    if enable_negative_sample_training:
        negative_indices: list[int] = [
            idx
            for uid, lst in per_uid.items()
            if any(idx in positive_set for (idx, _) in lst)
            for (idx, s) in lst
            if s <= 0
        ]
        merged_indices = sorted(positive_set | set(negative_indices))
    else:
        merged_indices = sorted(positive_set)

    if dp_world_size > 1:
        trim_len = (len(merged_indices) // dp_world_size) * dp_world_size
        if trim_len == 0:
            return batch.select_idxs([])
        merged_indices = merged_indices[:trim_len]

    out = batch.select_idxs(merged_indices)

    if enable_negative_sample_training:
        device = out.batch["token_level_scores"].device
        signs = [
            1.0 if float(scores[idx].item()) > 0 else -float(negative_sample_loss_scale)
            for idx in merged_indices
        ]
        if any(s < 0 for s in signs):
            out.batch[SAMPLE_WEIGHT_KEY] = torch.tensor(
                signs, device=device, dtype=torch.float32
            )

    return out


def _apply_advantage_weighting(
    batch: DataProto,
    *,
    normalize_std: bool,
    std_eps: float,
    dp_world_size: int,
) -> DataProto:
    """Advantage-weighted sample selection for baseline-subtracted / GRPO training.

    For each uid group:
      advantage_i = reward_i - mean(rewards)                  [baseline="mean"]
      advantage_i /= (std(rewards) + std_eps)                 [+ normalize_std=True → GRPO]

    Samples with zero advantage (degenerate all-correct or all-incorrect groups)
    are dropped naturally.
    """
    uids = [str(u) for u in batch.non_tensor_batch["uid"]]
    scores = batch.batch["token_level_scores"].sum(-1)

    per_uid: dict[str, list[tuple[int, float]]] = {}
    for i, uid in enumerate(uids):
        per_uid.setdefault(uid, []).append((i, float(scores[i].item())))

    advantages: list[float] = [0.0] * len(uids)

    for uid, lst in per_uid.items():
        indices = [idx for idx, _ in lst]
        values = [s for _, s in lst]

        mean = sum(values) / len(values)
        advs = [v - mean for v in values]

        if normalize_std:
            # Population variance (mean of squared deviations; mean(advs)=0)
            variance = sum(a * a for a in advs) / len(advs)
            std = variance ** 0.5
            advs = [a / (std + std_eps) for a in advs]

        for idx, adv in zip(indices, advs):
            advantages[idx] = adv

    kept_indices = [i for i, adv in enumerate(advantages) if adv != 0.0]

    if not kept_indices:
        return batch.select_idxs([])

    if dp_world_size > 1:
        trim_len = (len(kept_indices) // dp_world_size) * dp_world_size
        if trim_len == 0:
            return batch.select_idxs([])
        kept_indices = kept_indices[:trim_len]

    out = batch.select_idxs(kept_indices)
    device = out.batch["token_level_scores"].device
    out.batch[SAMPLE_WEIGHT_KEY] = torch.tensor(
        [advantages[idx] for idx in kept_indices], device=device, dtype=torch.float32
    )
    return out


# ---------------------------------------------------------------------------
# Legacy public alias (kept for backward compatibility)
# ---------------------------------------------------------------------------

def apply_osft_sample_selection(
    batch: DataProto,
    *,
    enable_negative_sample_training: bool,
    negative_sample_loss_scale: float,
    dp_world_size: int,
    rollout_n: int | None,
) -> DataProto:
    """Deprecated: use apply_reward_processing with reward_baseline='none' instead."""
    return apply_reward_processing(
        batch,
        reward_baseline="none",
        enable_negative_sample_training=enable_negative_sample_training,
        negative_sample_loss_scale=negative_sample_loss_scale,
        dp_world_size=dp_world_size,
        rollout_n=rollout_n,
    )
