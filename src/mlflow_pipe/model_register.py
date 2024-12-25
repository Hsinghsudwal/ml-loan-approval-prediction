import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from src.constants import *



mlflow.set_tracking_uri("http://127.0.0.1:8080")
mlflow.set_experiment(EXPERIMENT_NAME)

def register_latest_model():
    try:

        client = MlflowClient()

        # get best run
        experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
        best_run = client.search_runs(
            experiment_ids=experiment.experiment_id,
            run_view_type=ViewType.ACTIVE_ONLY,
            max_results=1,
            order_by=["metrics.accuracy asc"],#desc acs
        )[0]

        # register the best model
        run_id = best_run.info.run_id
        model_uri = f"runs:/{run_id}/model"
        model_score = best_run.data.metrics["accuracy"]

        model_details = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)
        client.update_registered_model(
            name=model_details.name, 
            description=f"Model's best accuracy: {model_score}",
        )

        print('Model Register Completed')
    except Exception as e:
        raise Exception(e)
