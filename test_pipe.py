import sys
from src.mlflow_pipe.model_register import register_latest_model
from src.mlflow_pipe.model_staging import transition_model_stage
from src.mlflow_pipe.model_production import model_stage_production
# from test.test_production import ModelDrift
from test.model_test import TestModel

    
def pipe():
    print('Registering Model')
    register_latest_model()

    print('Staging Model')  
    transition_model_stage()

    print("Testing stage and load model ")
    testmodel=TestModel()
    testmodel.load_stage()
    testmodel.load_model()

    print("Test model performance")
    path=r'output/raw/test_data.csv'
    # print(path)
    testmodel.model_test(path)

    print("Model to production")
    model_stage_production()

if __name__=="__main__":
    pipe()
    

    