from dataclasses import dataclass
from typing import Dict, Any
import mlflow

@dataclass
class MLflowRunContext:
    run_id: str

    def end(self):
        mlflow.end_run()

def start_mlflow_run(experiment_name: str, run_name: str = "run") -> MLflowRunContext:
    mlflow.set_experiment(experiment_name)
    run = mlflow.start_run(run_name=run_name)
    return MLflowRunContext(run_id=run.info.run_id)

def log_dict_metrics_params(ctx: MLflowRunContext, params: Dict[str, Any], metrics: Dict[str, float]):
    for k, v in params.items():
        try:
            mlflow.log_param(k, v)
        except Exception:
            mlflow.log_param(k, str(v))
    for k, v in metrics.items():
        try:
            mlflow.log_metric(k, float(v))
        except Exception:
            pass
