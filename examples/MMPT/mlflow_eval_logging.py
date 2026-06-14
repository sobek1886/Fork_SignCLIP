"""Shared MLflow logging for the ASL-Citizen feature evals (Eval A / Eval B).

Every eval invocation creates a FRESH MLflow run (never reuses/overwrites a previous
one), named after the feature set (e.g. "baseline", "run2", "signclip_aug_ft") and
tagged with eval_type=A|B so re-runs accumulate as separate, timestamped runs.
Metric names are prefixed ("evalA_" / "evalB_").

Server: https://mlflow.ai.mytkhgroup.com/  (auth via ~/.mlflow/credentials).
Failures are swallowed with a warning so an eval never dies on a logging hiccup.

MLflow metric names disallow '@', so "Rec@1" is logged as "Rec1".
"""

import os

TRACKING_URI = "https://mlflow.ai.mytkhgroup.com/"


def _sanitize(key: str) -> str:
    return key.replace("@", "")


def log_eval_to_mlflow(experiment, run_name, prefix, metrics,
                       params=None, tags=None, tracking_uri=TRACKING_URI):
    """Log one eval's metrics into a per-feature-set MLflow run.

    Args:
        experiment:  MLflow experiment name (created if missing).
        run_name:    run name = the feature set (e.g. "run2").
        prefix:      metric prefix, e.g. "evalA_" or "evalB_" (also sets eval_type tag).
        metrics:     dict of metric_name -> value (non-numeric values are skipped).
        params:      optional dict of params to log (str-cast).
        tags:        optional dict of tags to set.
    Always starts a FRESH run (no reuse/overwrite).
    Returns True on success, False if logging was skipped/failed.
    """
    try:
        import mlflow
    except ImportError:
        print("[MLflow] mlflow not installed; skipping logging.")
        return False

    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment)

        numeric = {}
        for k, v in metrics.items():
            try:
                numeric[_sanitize(prefix + k)] = float(v)
            except (TypeError, ValueError):
                continue

        eval_type = prefix.replace("eval", "").rstrip("_") or prefix  # "evalA_" -> "A"
        with mlflow.start_run(run_name=run_name):   # fresh run every call
            mlflow.log_metrics(numeric)
            if params:
                mlflow.log_params({k: str(v) for k, v in params.items()})
            base_tags = {"eval_type": eval_type,
                         "node": os.environ.get("SLURMD_NODENAME", "unknown"),
                         "job_id": os.environ.get("SLURM_JOB_ID", "unknown")}
            if tags:
                base_tags.update({k: str(v) for k, v in tags.items()})
            mlflow.set_tags(base_tags)

        print(f"[MLflow] logged run '{run_name}' (eval {eval_type}) in '{experiment}': "
              + ", ".join(f"{k}={v:.2f}" for k, v in numeric.items()))
        return True
    except Exception as exc:
        print(f"[MLflow] WARNING: could not log to MLflow: {exc}")
        return False


def log_evals_to_mlflow(experiment, run_name, groups,
                        params=None, tags=None, tracking_uri=TRACKING_URI):
    """Log several eval metric groups into ONE fresh MLflow run (A + B together).

    Args:
        groups: dict of prefix -> metrics dict, e.g.
                {"evalA_": {"DCG": ..., "Rec@1": ...}, "evalB_": {...}}.
    Always starts a FRESH run (no reuse/overwrite). Returns True on success.
    """
    try:
        import mlflow
    except ImportError:
        print("[MLflow] mlflow not installed; skipping logging.")
        return False

    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment)

        numeric = {}
        eval_types = []
        for prefix, metrics in groups.items():
            for k, v in metrics.items():
                try:
                    numeric[_sanitize(prefix + k)] = float(v)
                except (TypeError, ValueError):
                    continue
            et = prefix.replace("eval", "").rstrip("_")
            if et:
                eval_types.append(et)

        with mlflow.start_run(run_name=run_name):
            mlflow.log_metrics(numeric)
            if params:
                mlflow.log_params({k: str(v) for k, v in params.items()})
            base_tags = {"eval_type": "+".join(eval_types) or "?",
                         "node": os.environ.get("SLURMD_NODENAME", "unknown"),
                         "job_id": os.environ.get("SLURM_JOB_ID", "unknown")}
            if tags:
                base_tags.update({k: str(v) for k, v in tags.items()})
            mlflow.set_tags(base_tags)

        print(f"[MLflow] logged run '{run_name}' (evals {'+'.join(eval_types)}) "
              f"in '{experiment}': " + ", ".join(f"{k}={v:.2f}" for k, v in numeric.items()))
        return True
    except Exception as exc:
        print(f"[MLflow] WARNING: could not log to MLflow: {exc}")
        return False
