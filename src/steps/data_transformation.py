import pandas as pd
import numpy as np
import json
import joblib
from src.utility import *
from src.constants import *
import os

class DataTransformation:
    def __init__(self):
        pass
    
    def dataValidate():
        try:
            with open("output/validate/report.json") as f_in:
                result=json.load(f_in)
            
            dict_list_value = list(result.values())
            # print(dict_list_value[-1])
            return dict_list_value[-1]
        except Exception as e:
            raise Exception(e)
        
    def dataTransform(self, isfound,train_path,test_path):
        try:
            # data Transform
            status = False
            value=DataTransformation.dataValidate()
            
            if isfound != status and isfound !=value:
                print('validate: ', value!=status)
                return 'Error-Data Drift Detected'
            
            elif isfound == status:
                print('No Drift Detected')
                
                train_path[' loan_status']=train_path[' loan_status'].map({' Approved':1, ' Rejected': 0})
                test_path[' loan_status']=test_path[' loan_status'].map({' Approved':1, ' Rejected': 0})

                xtrain_data= train_path.drop(columns=['loan_id',' loan_status'], axis=1)
                xtest_data= test_path.drop(columns=['loan_id',' loan_status'], axis=1)

                train_y=train_path[' loan_status']
                test_y=test_path[' loan_status']

                # pipeline
                x, y = Utility.pipeline_process(xtrain_data,xtest_data)

                train_x, test_x = Utility.label_encoder(x, y)

                # create folder to store processor
                transformer_path = os.path.join(OUTPUT,TRANSFORMATION_FOLDER)
                os.makedirs(transformer_path,exist_ok=True)

                train_x.to_csv(os.path.join(transformer_path,str(X_TRAIN_PROCESS_DATA)))
                train_y.to_csv(os.path.join(transformer_path,str(Y_TRAIN_PROCESS_DATA)))
                test_x.to_csv(os.path.join(transformer_path,str(X_TEST_PROCESS_DATA)))
                test_y.to_csv(os.path.join(transformer_path,str(Y_TEST_PROCESS_DATA)))

            print('Data Transformation Completed')
            return train_x, train_y, test_x, test_y

        except Exception as e:
            raise Exception(e)
