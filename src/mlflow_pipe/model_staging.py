from mlflow.tracking import MlflowClient
from datetime import datetime
import mlflow
import time
from src.constants import *


mlflow.set_tracking_uri("http://127.0.0.1:8080")
mlflow.set_experiment(EXPERIMENT_NAME)

def transition_model_stage():
    try:
        date = datetime.today().date()

        client = MlflowClient()
        latest_versions = client.get_latest_versions(MODEL_NAME, stages=["None"])

        for version in latest_versions:
            model_version = version.version
            # model_run_id = version.run_id

        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=model_version,
            stage=STAGE,
            archive_existing_versions=True,
        )

        client.update_model_version(
            name=MODEL_NAME,
            version=model_version,
            description=f"This model version {model_version} was transition to {STAGE} on {date}.",
        )

        print('Stage Staging Complete')
    except Exception as e:
        raise Exception (e)
    
