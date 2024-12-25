import pandas as pd
import numpy as np
import joblib
from src.utility import *
from src.constants import *
import os
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import mlflow

class ModelTrainer:
    
    def __init__(self):
        pass
    

    def bestModel(self, xtrain,ytrain,xtest,ytest):
        try:

            # print(xtrain.values)
            x_train = xtrain.values
            y_train = ytrain.values
            x_test = xtest.values
            y_test = ytest.values

            models, params=Utility.models_params(self)

            results=[]

            for name, model in models.items():
               model.fit(x_train, y_train)
               y_pred = model.predict(x_test)
               acc_test = accuracy_score(y_test, y_pred)
            #    print(acc_test)
               param_name = params[name]
            #    print(name, model)

               results.append({'Model_name': name, 'Model':model,'accuracy':acc_test,'Params':param_name})

            model_results=pd.DataFrame(results)

            model_sorting=model_results.sort_values(by='accuracy', ascending=False)

            get_first_row=model_sorting.iloc[0].to_list()

            model_name=get_first_row[1]
            model_param=get_first_row[3]
            # print(model_name, model_param)

            gs = GridSearchCV(model_name, model_param, cv=3, scoring='accuracy',n_jobs=-1)
            # gs=model_name
            gs.fit(x_train, y_train)

            best_estimator=gs.best_estimator_
            best_params=gs.best_params_
            
            # create folder to store model
            model_path = os.path.join(OUTPUT,MODEL_FOLDER)
            os.makedirs(model_path,exist_ok=True)

            model_sorting.to_csv(os.path.join(model_path,str(MODEL_LIST)))
            model_file_path=os.path.join(model_path,BEST_MODEL_NAME)

            model_file=joblib.dump(best_estimator, model_file_path)

            print('Model Trainer Completed')
            return best_estimator, best_params, x_train, y_train, x_test, y_test

        except Exception as e:
            raise Exception(e)