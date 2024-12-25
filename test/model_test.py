import os
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
from src.constants import *
from sklearn.metrics import accuracy_score
from test.test_utils import preprocess_pipe

import unittest

class TestModel(unittest.TestCase):
    # testing stage and model performance

    def load_stage(self):
        try:
            client = MlflowClient()

            # Get the latest model stage-staging
            versions = client.get_latest_versions(MODEL_NAME, stages=[STAGE])

            # latest_staging_version = versions[0]
            
            self.assertGreater(len(versions), 0, "Model not found in the {STAGE} stage.")
        
        except Exception as e:
            raise Exception(e)
        

    def load_model(self):

        try:
            client = MlflowClient()

            # Get the latest model stage-staging
            versions = client.get_latest_versions(MODEL_NAME, stages=[STAGE])

            latest_staging_version = versions[0]

            # Get the version details
            
            run_id = versions[0].run_id  # get run ID
            # print(run_id)

            # load model using its run ID
            logged = f"runs:/{run_id}/{MODEL_NAME}"

            loaded_model = mlflow.pyfunc.load_model(logged)

            if not loaded_model:
                self.fail("Failed to load the model:")
            
            print("Model Loaded")
            self.assertIsNotNone(loaded_model, "No Loaded Model.")
        
        except Exception as e:
            raise Exception(e)

    def model_test(self,path):
                # Model Drift (Retraining the Model)

                testdata=pd.read_csv(path, index_col=False)

                x_train, x_test, y_train, y_test= preprocess_pipe(testdata)

                client = MlflowClient()
                versions = client.get_latest_versions(MODEL_NAME, stages=[STAGE])

                # latest_version = versions[0].version
                run_id = versions[0].run_id

                logged = f"runs:/{run_id}/{MODEL_NAME}"

                loaded_model = mlflow.pyfunc.load_model(logged)
                
                ypred=loaded_model.predict(x_test)

                accuracy=round(accuracy_score(y_test, ypred),2)
                print('accuracy_score: ',accuracy)

                train_accuracy=0.99
                result= train_accuracy - 0.1
                if accuracy < result: 
                    # or p_value < 0.05
                    print("Model drifted...retraining required.")
                    

                    self.assertGreaterEqual(accuracy, result, "Model perform below threshold, retraining needed.")

                # Model train with new independent and dependant variables.
                # model.fit(X_new, y_new)
                # store/ save/ logged model 
                # update mlflow model
                # deployed

                
                

