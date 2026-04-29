from __future__ import annotations

import copy
import glob
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from verl import DataProto


def _clone_dataproto_to_cpu(data: DataProto) -> DataProto:
    """Deep-copy a DataProto snapshot to CPU for replay usage."""
    tensors = {}
    if data.batch is not None:
        for key, tensor in data.batch.items():
            tensors[key] = tensor.detach().clone().cpu()

    non_tensors = {}
    for key, val in data.non_tensor_batch.items():
        non_tensors[key] = np.array(val, copy=True)

    return DataProto.from_dict(
        tensors=tensors,
        non_tensors=non_tensors,
        meta_info=copy.deepcopy(data.meta_info),
    )


@dataclass
class DataSourceConfig:
    mode: str = "on_policy"  # one of: on_policy | bootstrap | iterative | teacher
    iterative_k: int = 1
    seed: int = 1
    teacher_dataproto_globs: tuple[str, ...] = ()


class TrajectoryDataSourceController:
    """Selects which trajectory batch should drive actor updates."""

    def __init__(self, cfg: DataSourceConfig, total_training_steps: int):
        self.cfg = cfg
        self.total_training_steps = max(int(total_training_steps), 1)

        self._bootstrap_batches: deque[DataProto] = deque()
        self._iterative_batches: deque[DataProto] = deque()
        self._teacher_batches: list[DataProto] = []
        self._teacher_idx = 0

        if self.cfg.mode == "teacher":
            if self.cfg.teacher_dataproto_globs:
                self._teacher_batches = self._load_teacher_batches(self.cfg.teacher_dataproto_globs)

    @classmethod
    def from_omegaconf(cls, cfg, total_training_steps: int) -> "TrajectoryDataSourceController":
        mode = str(cfg.get("mode", "on_policy")).strip().lower()
        ds_cfg = DataSourceConfig(
            mode=mode,
            iterative_k=int(cfg.get("iterative_k", 1)),
            seed=int(cfg.get("seed", 1)),
            teacher_dataproto_globs=tuple(cfg.get("teacher_dataproto_globs", [])),
        )
        supported = {"on_policy", "bootstrap", "iterative", "teacher"}
        if ds_cfg.mode not in supported:
            raise ValueError(f"Unknown trainer.data_source.mode={ds_cfg.mode!r}. Supported: {sorted(supported)}")
        if ds_cfg.iterative_k < 1:
            raise ValueError("trainer.data_source.iterative_k must be >= 1.")
        return cls(ds_cfg, total_training_steps=total_training_steps)

    def needs_rollout_generation(self) -> bool:
        if self.cfg.mode == "teacher":
            return len(self._teacher_batches) == 0
        if self.cfg.mode == "bootstrap":
            return len(self._bootstrap_batches) == 0
        if self.cfg.mode == "iterative":
            return len(self._iterative_batches) == 0
        return True

    @property
    def mode(self) -> str:
        return self.cfg.mode

    def set_bootstrap_batches(self, batches: list[DataProto]) -> None:
        self._bootstrap_batches = deque(_clone_dataproto_to_cpu(batch) for batch in batches)

    def iterative_chunk_size(self) -> int:
        return int(self.cfg.iterative_k)

    def set_iterative_batches(self, batches: list[DataProto]) -> None:
        self._iterative_batches = deque(_clone_dataproto_to_cpu(batch) for batch in batches)

    def set_teacher_batches(self, batches: list[DataProto]) -> None:
        self._teacher_batches = [_clone_dataproto_to_cpu(batch) for batch in batches]
        self._teacher_idx = 0

    def teacher_pool_size(self) -> int:
        return len(self._teacher_batches)

    def select_training_batch(self, current_on_policy_batch: DataProto | None) -> tuple[DataProto, dict]:
        mode = self.cfg.mode
        mode_id = {"on_policy": 0, "bootstrap": 1, "iterative": 2, "teacher": 3}[mode]
        metrics = {"training/data_source_mode_id": mode_id}

        if mode == "teacher":
            if not self._teacher_batches:
                raise ValueError(
                    "No teacher trajectories available. Provide DataProto files via "
                    "trainer.data_source.teacher_dataproto_globs or teacher parquet "
                    "train_files with a 'responses' column."
                )
            teacher_batch = self._teacher_batches[self._teacher_idx]
            self._teacher_idx = (self._teacher_idx + 1) % len(self._teacher_batches)
            out = _clone_dataproto_to_cpu(teacher_batch)
            metrics["training/data_source_is_off_policy"] = 1
            metrics["training/data_source_teacher_pool_size"] = len(self._teacher_batches)
            return out, metrics

        if mode == "bootstrap":
            if not self._bootstrap_batches:
                raise ValueError(
                    "No bootstrap trajectories available. Precompute with the base model "
                    "and call set_bootstrap_batches(...) before training steps."
                )
            metrics["training/data_source_is_off_policy"] = 1
            metrics["training/data_source_cached_batches"] = len(self._bootstrap_batches)
            out = self._bootstrap_batches.popleft()
            return out, metrics

        if mode == "on_policy":
            if current_on_policy_batch is None:
                raise ValueError("current_on_policy_batch must be provided for mode='on_policy'.")
            metrics["training/data_source_is_off_policy"] = 0
            return current_on_policy_batch, metrics

        # mode == "iterative":
        if not self._iterative_batches:
            raise ValueError(
                "No iterative trajectories available. Generate the next k batches from "
                "the current checkpoint and call set_iterative_batches(...)."
            )
        metrics["training/data_source_is_off_policy"] = 1
        metrics["training/data_source_selected_kind_id"] = 2
        metrics["training/data_source_cached_batches"] = len(self._iterative_batches)
        out = self._iterative_batches.popleft()
        return out, metrics

    def _load_teacher_batches(self, globs: tuple[str, ...]) -> list[DataProto]:
        loaded: list[DataProto] = []
        seen: set[Path] = set()
        for pattern in globs:
            for matched in glob.glob(pattern):
                path = Path(matched).expanduser().resolve()
                if path in seen:
                    continue
                seen.add(path)
                loaded.append(DataProto.load_from_disk(str(path)))
        return loaded
