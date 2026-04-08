# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
make a general fairseq task for MM pretraining.
"""

import atexit
import os
import random

from fairseq.tasks import LegacyFairseqTask, register_task

from .task import Task
from .retritask import RetriTask
from ..datasets import FairseqMMDataset
from .. import utils

try:
    import mlflow as _mlflow
    _MLFLOW = True
except ImportError:
    _MLFLOW = False


@register_task("mmtask")
class FairseqMMTask(LegacyFairseqTask):
    @staticmethod
    def add_args(parser):
        # Add some command-line arguments for specifying where the data is
        # located and the maximum supported input length.
        parser.add_argument(
            "taskconfig",
            metavar="FILE",
            help=("taskconfig to load all configurations" "outside fairseq parser."),
        )

    @classmethod
    def setup_task(cls, args, **kwargs):
        return FairseqMMTask(args)

    def __init__(self, args):
        super().__init__(args)
        config = utils.load_config(args)
        self.mmtask = Task.config_task(config)
        self.mmtask.build_dataset()
        self.mmtask.build_model()
        self.mmtask.build_loss()

        if getattr(config.model, 'freeze_bert', False):
            for name, param in self.mmtask.model.named_parameters():
                if 'videomlp' not in name:
                    param.requires_grad_(False)
            trainable = [n for n, p in self.mmtask.model.named_parameters() if p.requires_grad]
            print(f"[freeze_bert] Trainable parameters ({len(trainable)}): {trainable}")

        local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0)))
        if _MLFLOW and local_rank == 0:
            try:
                _mlflow.set_tracking_uri("https://mlflow.ai.mytkhgroup.com/")
                _mlflow.set_experiment("signclip-cnn")
                run_name = os.path.splitext(os.path.basename(args.taskconfig))[0]
                try:
                    _mlflow.enable_system_metrics_logging()
                except AttributeError:
                    pass  # requires mlflow>=2.8 and pynvml
                _mlflow.start_run(run_name=run_name)
                _mlflow.log_params({
                    "vfeat_dim": int(config.model.vfeat_dim),
                    "max_video_len": int(config.dataset.max_video_len),
                    "batch_size": int(config.fairseq.dataset.batch_size),
                    "lr": float(config.fairseq.optimization.lr[0]),
                    "max_epoch": int(config.fairseq.optimization.max_epoch),
                    "num_hidden_video_layers": int(config.model.num_hidden_video_layers),
                    "meta_processor": str(config.dataset.meta_processor),
                    "restore_file": str(getattr(config.fairseq.checkpoint, "restore_file", "none")),
                })
                _mlflow.set_tag("slurm_job_id", os.environ.get("SLURM_JOB_ID", "unknown"))
                _mlflow.set_tag("slurm_node", os.environ.get("SLURMD_NODENAME", "unknown"))
                print("[MLflow] Run started:", _mlflow.active_run().info.run_id)

                job_id = os.environ.get("SLURM_JOB_ID", "")
                job_name = os.environ.get("SLURM_JOB_NAME", "")
                self._mlflow_log_file = (
                    os.path.join(os.getcwd(), "jobs", "output",
                                 f"slurm_{job_name}_{job_id}.txt")
                    if job_id and job_name else None
                )

                def _mlflow_cleanup():
                    try:
                        if _mlflow.active_run() is None:
                            return
                        if self._mlflow_log_file and os.path.exists(self._mlflow_log_file):
                            _mlflow.log_artifact(self._mlflow_log_file)
                        _mlflow.end_run()
                    except Exception as exc:
                        print(f"[MLflow] Warning: cleanup failed: {exc}")

                atexit.register(_mlflow_cleanup)
            except Exception as exc:
                print(f"[MLflow] Warning: could not initialize run: {exc}")
                self._mlflow_log_file = None

    def begin_epoch(self, epoch, model):
        super().begin_epoch(epoch, model)
        from ..losses.fairseqmmloss import MMCriterion
        local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0)))
        if _MLFLOW and local_rank == 0 and MMCriterion._valid_loss_count > 0:
            if _mlflow.active_run() is not None:
                avg = MMCriterion._valid_loss_sum / MMCriterion._valid_loss_count
                _mlflow.log_metric("valid_loss", round(avg, 5),
                                   step=MMCriterion._mlflow_step)
        MMCriterion._valid_loss_sum = 0.0
        MMCriterion._valid_loss_count = 0
        MMCriterion._phase = "train"

    def begin_valid_epoch(self, epoch, model):
        from ..losses.fairseqmmloss import MMCriterion
        MMCriterion._phase = "valid"
        local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0)))
        if _MLFLOW and local_rank == 0:
            try:
                log_file = getattr(self, "_mlflow_log_file", None)
                if log_file and os.path.exists(log_file) and _mlflow.active_run() is not None:
                    _mlflow.log_artifact(log_file)
            except Exception as exc:
                print(f"[MLflow] Warning: could not upload log artifact: {exc}")

    def load_dataset(self, split, **kwargs):
        split_map = {
            "train": self.mmtask.train_data,
            "valid": self.mmtask.val_data,
            "test": self.mmtask.test_data,
        }
        if split not in split_map:
            raise ValueError("unknown split type.")
        if split_map[split] is not None:
            self.datasets[split] = FairseqMMDataset(split_map[split])

    def get_batch_iterator(
        self,
        dataset,
        max_tokens=None,
        max_sentences=None,
        max_positions=None,
        ignore_invalid_inputs=False,
        required_batch_size_multiple=1,
        seed=1,
        num_shards=1,
        shard_id=0,
        num_workers=0,
        epoch=1,
        data_buffer_size=0,
        disable_iterator_cache=False,
        skip_remainder_batch=False,
        grouped_shuffling=False,
        update_epoch_batch_itr=False,
    ):
        random.seed(epoch)
        if dataset.mmdataset.split == "train" and isinstance(self.mmtask, RetriTask):
            if epoch >= self.mmtask.config.retri_epoch:
                if not hasattr(self.mmtask, "retri_dataloader"):
                    self.mmtask.build_dataloader()
                self.mmtask.retrive_candidates(epoch)

        return super().get_batch_iterator(
            dataset,
            max_tokens,
            max_sentences,
            max_positions,
            ignore_invalid_inputs,
            required_batch_size_multiple,
            seed,
            num_shards,
            shard_id,
            num_workers,
            epoch,
            data_buffer_size,
            disable_iterator_cache,
            grouped_shuffling,
            update_epoch_batch_itr,
        )

    @property
    def source_dictionary(self):
        return None

    @property
    def target_dictionary(self):
        return None
