# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import glob
import argparse
import pprint
import omegaconf

from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from mmpt.utils import load_config, set_seed
from mmpt.evaluators import Evaluator
from mmpt.evaluators import predictor as predictor_path
from mmpt.tasks import Task
from mmpt import processors
from mmpt.datasets import MMDataset


def get_dataloader(config, verbose=True):
    meta_processor_cls = getattr(processors, config.dataset.meta_processor)
    video_processor_cls = getattr(processors, config.dataset.video_processor)
    text_processor_cls = getattr(processors, config.dataset.text_processor)
    aligner_cls = getattr(processors, config.dataset.aligner)

    meta_processor = meta_processor_cls(config.dataset)
    video_processor = video_processor_cls(config.dataset)
    text_processor = text_processor_cls(config.dataset)
    aligner = aligner_cls(config.dataset)

    test_data = MMDataset(
        meta_processor,
        video_processor,
        text_processor,
        aligner,
    )
    print("test_len", len(test_data))
    if verbose:
        output = test_data[0]
        test_data.print_example(output)

    test_dataloader = DataLoader(
        test_data,
        batch_size=config.fairseq.dataset.batch_size,
        shuffle=False,
        num_workers=config.fairseq.dataset.num_workers,
        collate_fn=test_data.collater,
    )
    return test_dataloader


def _log_test_metrics_to_mlflow(config, taskconfig_path, metrics):
    """Log test metrics into the corresponding MLflow training run.

    Derives the training run name by stripping the leading "test_" from the
    test config filename, then searches the same experiment for that run and
    appends test_t2v_* / test_v2t_* metrics.
    """
    try:
        import mlflow as _mlflow
        experiment_name = getattr(config, "mlflow_experiment", None)
        if not experiment_name:
            print("[MLflow] Warning: 'mlflow_experiment' not set in config; skipping test metric logging.")
            return
        test_config_name = os.path.splitext(os.path.basename(taskconfig_path))[0]
        train_run_name = (
            test_config_name[len("test_"):]
            if test_config_name.startswith("test_")
            else test_config_name
        )
        _mlflow.set_tracking_uri("https://mlflow.ai.mytkhgroup.com/")
        _mlflow.set_experiment(experiment_name)
        client = _mlflow.tracking.MlflowClient()
        exp = client.get_experiment_by_name(experiment_name)
        if exp is None:
            print(f"[MLflow] Experiment '{experiment_name}' not found; skipping test metric logging.")
            return
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string=f"attributes.run_name = '{train_run_name}'",
            order_by=["start_time DESC"],
            max_results=1,
        )
        if not runs:
            print(f"[MLflow] No run named '{train_run_name}' found; skipping test metric logging.")
            return
        run_id = runs[0].info.run_id
        flat = {}
        for direction, direction_metrics in metrics.items():
            for name, value in direction_metrics.items():
                if name != "error":
                    flat[f"test_{direction}_{name}"] = float(value)
        with _mlflow.start_run(run_id=run_id):
            _mlflow.log_metrics(flat)
        print(f"[MLflow] Logged test metrics to run {run_id} ({train_run_name}): "
              + ", ".join(f"{k}={v:.4f}" for k, v in flat.items()))
    except Exception as exc:
        print(f"[MLflow] Warning: could not log test metrics: {exc}")


def main(args):
    config = load_config(args)
    verbose = not args.quiet

    if verbose:
        if isinstance(config, omegaconf.dictconfig.DictConfig):
            print(OmegaConf.to_yaml(config))
        else:
            pp = pprint.PrettyPrinter(indent=4)
            pp.print(config)

    mmtask = Task.config_task(config)
    mmtask.build_model()

    configs = [config]

    if config['dataset']['test_separately']:
        configs = [OmegaConf.create(dict(config, dataset=dict(config['dataset'], test_datasets=[test_dataset]))) for test_dataset in config['dataset']['test_datasets']]

    for config in configs:
        test_dataloader = get_dataloader(config, verbose=verbose)
        checkpoint_search_path = os.path.dirname(config.eval.save_path)
        results = []

        prefix = os.path.basename(args.taskconfig)
        if prefix.startswith("test"):
            # loop all checkpoint for datasets without validation set.
            if "best" not in config.fairseq.common_eval.path:
                print("eval each epoch.")
                for checkpoint in glob.glob(checkpoint_search_path + "/checkpoint*"):
                    model = mmtask.load_checkpoint(checkpoint)
                    ckpt = os.path.basename(checkpoint)
                    evaluator = Evaluator(config)
                    output = evaluator.evaluate(
                        model, test_dataloader, ckpt + "_merged")
                    results.append((checkpoint, output))
            # use the one specified by the config lastly.
            model = mmtask.load_checkpoint(config.fairseq.common_eval.path)
            evaluator = Evaluator(config)
            output = evaluator.evaluate(model, test_dataloader)
            results.append((config.fairseq.common_eval.path, output))

            best_result = None
            best_metric = 0.
            for checkpoint, result in results:
                # print(checkpoint)
                # evaluator.metric.print_computed_metrics(result)
                best_score = evaluator.metric.best_metric(result)
                if best_score > best_metric:
                    best_result = (checkpoint, result)
                    best_metric = best_score
            print("best results:")
            print(best_result[0])
            evaluator.metric.print_computed_metrics(best_result[1])
            _log_test_metrics_to_mlflow(config, args.taskconfig, best_result[1])

        elif prefix.startswith("vis"):
            model = mmtask.load_checkpoint(config.fairseq.common_eval.path)
            predictor_cls = getattr(predictor_path, config.predictor)
            predictor = predictor_cls(config)
            predictor.predict_loop(model, test_dataloader, mmtask, None)
        else:
            raise ValueError("unknown prefix of the config file", args.taskconfig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("taskconfig", type=str)
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Suppress config dump and per-example debug output; print only final metrics.")
    args = parser.parse_args()
    main(args)
