# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import uuid
from contextlib import nullcontext
from pprint import pprint
from typing import Optional, Type

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm

from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayWorkerGroup
from verl.trainer.ppo.metric_utils import (
    compute_throughout_metrics,
    compute_timing_metrics,
)
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager, Role, compute_response_mask
from verl.utils.debug.performance import _timer
from verl.utils.metric import (
    reduce_metrics,
)
from verl.utils.tracking import ValidationGenerationsLogger

from recipe.osft.data_source_controller import TrajectoryDataSourceController
from recipe.osft.osft_sample_selection import SAMPLE_WEIGHT_KEY, apply_reward_processing

WorkerType = Type[Worker]


class RayOSFTTrainer(RayPPOTrainer):
    """
    RayOSFTTrainer is a trainer for online supervised fine-tuning (OSFT) using Ray.
    """

    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name="cuda",
    ):
        """Initialize distributed PPO trainer with Ray backend."""
        
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn
        self.use_critic = False

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = False
        self.use_rm = False
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name
        self.validation_generations_logger = ValidationGenerationsLogger()

        self._validate_config()
        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _build_on_policy_training_batch(self, batch_dict, metrics: dict, timing_raw: dict) -> DataProto | None:
        """Create an on-policy training batch from prompt-only dataloader input."""
        batch: DataProto = DataProto.from_single_dict(batch_dict)

        # pop those keys for generation
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
        if "multi_modal_data" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("multi_modal_data")
        if "raw_prompt" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("raw_prompt")
        if "tools_kwargs" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("tools_kwargs")
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
        )

        gen_timer = _timer("gen", timing_raw) if timing_raw is not None else nullcontext()
        with gen_timer:
            if not self.async_rollout_mode:
                gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
            else:
                self.async_rollout_manager.wake_up()
                gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                self.async_rollout_manager.sleep()
            if timing_raw is not None:
                timing_raw.update(gen_batch_output.meta_info["timing"])
            gen_batch_output.meta_info.pop("timing", None)

        batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)

        # repeat to align with repeated responses in rollout
        batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
        batch = batch.union(gen_batch_output)
        batch.batch["response_mask"] = compute_response_mask(batch)
        return self._apply_reward_processing_and_metrics(batch, metrics)

    def _apply_reward_processing_and_metrics(self, batch: DataProto, metrics: dict) -> DataProto | None:
        """Apply reward scoring/filtering and write shared training metrics."""
        if self.config.trainer.enable_train_reward and self.reward_fn is not None:
            reward_tensor = self.reward_fn(batch)
            batch.batch["token_level_scores"] = reward_tensor
            # we do not have adv, therefore we use scores as rewards
            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

            sequence_score = batch.batch["token_level_scores"].sum(-1)
            metrics.update(
                {
                    "reward/score/mean": torch.mean(sequence_score).item(),
                    "reward/score/max": torch.max(sequence_score).item(),
                    "reward/score/min": torch.min(sequence_score).item(),
                }
            )

            batch, group_metrics = apply_reward_processing(
                batch,
                reward_baseline=self.config.trainer.get("reward_baseline", "none"),
                reward_normalize_std=bool(self.config.trainer.get("reward_normalize_std", False)),
                reward_std_eps=float(self.config.trainer.get("reward_std_eps", 1e-8)),
                enable_negative_sample_training=self.config.trainer.get("enable_negative_sample_training", False),
                negative_sample_loss_scale=float(self.config.trainer.get("negative_sample_loss_scale", 1.0)),
                dp_world_size=getattr(self.actor_rollout_wg, "world_size", 1),
                rollout_n=getattr(self.config.actor_rollout_ref.rollout, "n", None),
            )
            metrics.update(group_metrics)
            if len(batch) == 0:
                return None
            if SAMPLE_WEIGHT_KEY in batch.batch.keys():
                weights = batch.batch[SAMPLE_WEIGHT_KEY]
                metrics["training/n_positive_seq"] = int((weights > 0).sum().item())
                metrics["training/n_negative_seq"] = int((weights < 0).sum().item())
                metrics["training/weight_mean"] = float(weights.mean().item())
                metrics["training/weight_std"] = float(weights.std().item()) if weights.numel() > 1 else 0.0
        return batch

    def _build_teacher_training_batch(self, batch_dict, metrics: dict) -> DataProto | None:
        """Build a trainable batch from teacher parquet rows with `responses` lists."""
        if "responses" not in batch_dict:
            raise ValueError("Teacher parquet mode expects a 'responses' column.")
        if "raw_prompt_ids" not in batch_dict:
            raise ValueError("Teacher parquet mode expects RL dataset field 'raw_prompt_ids'.")

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        if pad_id is None:
            pad_id = 0

        flat_prompt_ids = []
        flat_response_ids = []
        uids = []
        non_tensor_flat = {
            "data_source": [],
            "ability": [],
            "reward_model": [],
            "extra_info": [],
            "uid": [],
        }

        raw_prompt_ids_arr = batch_dict["raw_prompt_ids"]
        responses_arr = batch_dict["responses"]
        batch_size = len(responses_arr)

        for i in range(batch_size):
            prompt_ids = [int(x) for x in list(raw_prompt_ids_arr[i])]
            if len(prompt_ids) == 0:
                continue
            responses = list(responses_arr[i]) if responses_arr[i] is not None else []
            if len(responses) == 0:
                continue

            uid = str(uuid.uuid4())
            for resp in responses:
                response_text = str(resp)
                response_ids = self.tokenizer.encode(response_text, add_special_tokens=False)
                flat_prompt_ids.append(prompt_ids)
                flat_response_ids.append(response_ids)
                uids.append(uid)
                non_tensor_flat["uid"].append(uid)
                non_tensor_flat["data_source"].append(
                    batch_dict["data_source"][i] if "data_source" in batch_dict else "unknown"
                )
                non_tensor_flat["ability"].append(
                    batch_dict["ability"][i] if "ability" in batch_dict else "unknown"
                )
                non_tensor_flat["reward_model"].append(
                    batch_dict["reward_model"][i] if "reward_model" in batch_dict else {}
                )
                non_tensor_flat["extra_info"].append(
                    batch_dict["extra_info"][i] if "extra_info" in batch_dict else {}
                )

        if len(flat_prompt_ids) == 0:
            return None

        max_prompt_len = max(len(p) for p in flat_prompt_ids)
        max_resp_len = max(len(r) for r in flat_response_ids) if flat_response_ids else 1
        max_total_len = max(len(p) + len(r) for p, r in zip(flat_prompt_ids, flat_response_ids))

        prompts_tensor = []
        responses_tensor = []
        input_ids_tensor = []
        attention_mask_tensor = []
        position_ids_tensor = []

        for prompt_ids, response_ids in zip(flat_prompt_ids, flat_response_ids):
            p_pad = max_prompt_len - len(prompt_ids)
            r_pad = max_resp_len - len(response_ids)
            full_ids = prompt_ids + response_ids
            full_pad = max_total_len - len(full_ids)

            prompts_tensor.append(prompt_ids + [pad_id] * p_pad)
            responses_tensor.append(response_ids + [pad_id] * r_pad)
            input_ids_tensor.append(full_ids + [pad_id] * full_pad)
            attention_mask_tensor.append([1] * len(full_ids) + [0] * full_pad)
            position_ids_tensor.append(list(range(len(full_ids))) + [0] * full_pad)

        batch = DataProto.from_dict(
            tensors={
                "prompts": torch.tensor(prompts_tensor, dtype=torch.long),
                "responses": torch.tensor(responses_tensor, dtype=torch.long),
                "input_ids": torch.tensor(input_ids_tensor, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask_tensor, dtype=torch.long),
                "position_ids": torch.tensor(position_ids_tensor, dtype=torch.long),
            },
            non_tensors=non_tensor_flat,
        )
        batch.batch["response_mask"] = compute_response_mask(batch)
        return self._apply_reward_processing_and_metrics(batch, metrics)

    def fit(self):
        """
        The training loop of SFT.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the SFT dataflow.
        """

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()
        
        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return
   
        data_source_controller = TrajectoryDataSourceController.from_omegaconf(
            self.config.trainer.get("data_source", {}),
            total_training_steps=self.total_training_steps,
        )
        precomputed_steps = None
        bootstrap_mode = data_source_controller.mode == "bootstrap"
        teacher_mode = data_source_controller.mode == "teacher"
        iterative_mode = data_source_controller.mode == "iterative"
        if bootstrap_mode:
            bootstrap_batches = []
            for precompute_batch_dict in self.train_dataloader:
                precompute_metrics = {}
                precompute_timing = {}
                precomputed_batch = self._build_on_policy_training_batch(
                    precompute_batch_dict,
                    metrics=precompute_metrics,
                    timing_raw=precompute_timing,
                )
                if precomputed_batch is not None and len(precomputed_batch) > 0:
                    bootstrap_batches.append(precomputed_batch)
            if not bootstrap_batches:
                raise ValueError("Bootstrap mode produced zero trainable batches from the training dataset.")
            data_source_controller.set_bootstrap_batches(bootstrap_batches)
            if self.total_training_steps != len(bootstrap_batches):
                raise ValueError(
                    "Bootstrap mode requires one-pass training over pre-generated data. "
                    f"Set total_training_steps to exactly {len(bootstrap_batches)} "
                    f"(current: {self.total_training_steps}) to avoid replay or partial usage."
                )
            precomputed_steps = len(bootstrap_batches)
            print(f"[bootstrap] Prepared {len(bootstrap_batches)} off-policy batches from base-model trajectories.")
        if teacher_mode and data_source_controller.needs_rollout_generation():
            teacher_batches = []
            for teacher_batch_dict in self.train_dataloader:
                teacher_metrics = {}
                teacher_batch = self._build_teacher_training_batch(
                    teacher_batch_dict,
                    metrics=teacher_metrics,
                )
                if teacher_batch is not None and len(teacher_batch) > 0:
                    teacher_batches.append(teacher_batch)
            if not teacher_batches:
                raise ValueError(
                    "Teacher mode could not build any trainable batches from train_files. "
                    "Ensure the parquet contains non-empty 'responses' lists."
                )
            data_source_controller.set_teacher_batches(teacher_batches)
            if self.total_training_steps != len(teacher_batches):
                raise ValueError(
                    "Teacher mode expects one-pass training over teacher trajectories. "
                    f"Set total_training_steps to exactly {len(teacher_batches)} "
                    f"(current: {self.total_training_steps})."
                )
            precomputed_steps = len(teacher_batches)
            print(f"[teacher] Prepared {len(teacher_batches)} teacher off-policy batches.")
        elif teacher_mode:
            precomputed_steps = data_source_controller.teacher_pool_size()
            if precomputed_steps <= 0:
                raise ValueError("Teacher mode has no available batches.")
            if self.total_training_steps != precomputed_steps:
                raise ValueError(
                    "Teacher mode expects one-pass training over teacher trajectories. "
                    f"Set total_training_steps to exactly {precomputed_steps} "
                    f"(current: {self.total_training_steps})."
                )

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        epoch_iter = range(1) if (bootstrap_mode or teacher_mode) else range(self.config.trainer.total_epochs)
        for epoch in epoch_iter:
            if bootstrap_mode or teacher_mode:
                data_iter = iter(range(precomputed_steps))
            else:
                data_iter = iter(self.train_dataloader)

            while True:
                metrics = {}
                timing_raw = {}

                is_last_step = self.global_steps >= self.total_training_steps

                with _timer("step", timing_raw):
                    if bootstrap_mode or teacher_mode:
                        try:
                            next(data_iter)
                        except StopIteration:
                            break
                        batch, data_source_metrics = data_source_controller.select_training_batch(None)
                    elif iterative_mode:
                        if data_source_controller.needs_rollout_generation():
                            chunk_batch_dicts = []
                            chunk_size = data_source_controller.iterative_chunk_size()
                            for _ in range(chunk_size):
                                try:
                                    chunk_batch_dicts.append(next(data_iter))
                                except StopIteration:
                                    break
                            if not chunk_batch_dicts:
                                break
                            iterative_batches = []
                            for chunk_batch_dict in chunk_batch_dicts:
                                precompute_metrics = {}
                                precomputed_batch = self._build_on_policy_training_batch(
                                    chunk_batch_dict,
                                    metrics=precompute_metrics,
                                    timing_raw=None,
                                )
                                if precomputed_batch is not None and len(precomputed_batch) > 0:
                                    iterative_batches.append(precomputed_batch)
                            if not iterative_batches:
                                continue
                            data_source_controller.set_iterative_batches(iterative_batches)
                        batch, data_source_metrics = data_source_controller.select_training_batch(None)
                    else:
                        try:
                            batch_dict = next(data_iter)
                        except StopIteration:
                            break
                        current_on_policy_batch = self._build_on_policy_training_batch(
                            batch_dict,
                            metrics=metrics,
                            timing_raw=timing_raw,
                        )
                        if current_on_policy_batch is None:
                            continue
                        batch, data_source_metrics = data_source_controller.select_training_batch(current_on_policy_batch)
                    metrics.update(data_source_metrics)
                    if len(batch) == 0:
                        continue

                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # make tau_t = tau_s
                    if self.config.trainer.enable_train_temperature:
                        batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
                    else:
                        batch.meta_info["temperature"] = 1.0 # tau_t = 1

                    # update actor (core part)
                    with _timer("update_actor", timing_raw):
                        batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                        actor_output = self.actor_rollout_wg.update_actor(batch)
                    actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                    metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with _timer("dump_rollout_generations", timing_raw):
                            # print(batch.batch.keys())
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            # scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()

                            # --- Scores and Rewards (from reward_fn) ---
                            if "token_level_scores" in batch.batch and batch.batch["token_level_scores"] is not None:
                                sequence_score = batch.batch["token_level_scores"].sum(-1)
                                scores = sequence_score.cpu().tolist()
                                #metrics.update(
                                #    {
                                #        "reward/score/mean": torch.mean(sequence_score).item(),
                                #        "reward/score/max": torch.max(sequence_score).item(),
                                #        "reward/score/min": torch.min(sequence_score).item(),
                                #    }
                                #)
                            else:
                                print("DEBUG dump_rollout_generations: 'token_level_scores' not found.")
                                scores = [0 for _ in range(len(inputs))]  # placeholder, since we don't have scores in OSFT

                            response_lengths = batch.batch["response_mask"].sum(dim=-1).cpu().tolist()

                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                scores=scores,
                                reward_extra_infos_dict={"response_lengths": response_lengths},
                                dump_path=rollout_data_dir,
                            )

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                        with _timer("testing", timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics

                # no reward_fn, so no reward metrics from compute_data_metrics
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1
                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

    def _validate_base(self):
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None:
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)

        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            import json
            json.dump(val_metrics, open(f"{val_data_dir}/the_sub_metric.json", "w"), indent=4)

        if "swanlab" in logger.logger:
            logger.logger["swanlab"].finish()
        return val_metrics
