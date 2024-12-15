import os
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from scipy.stats import uniform, randint
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class Utility:
    def __init__(self) -> None:
        pass

    def read_data(data):
        try:
            # read dataframe
            df = pd.read_csv(data)
            return df

        except Exception as e:
            raise e

    def pipeline_process(self, x,y):
        try:
            numerical_features = x.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_features = x.select_dtypes(include=['object']).columns.tolist()

            numerical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')), 
                ('scaler', StandardScaler())  
                ])
            categorical_transformer=Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('ohe', OneHotEncoder(handle_unknown='ignore')),
                ])
            preprocessor = ColumnTransformer(
                transformers=[('num', numerical_transformer, numerical_features),
                              ('cat', categorical_transformer,categorical_features)]
                )
            
            x_preprocessed = preprocessor.fit_transform(x)
            y_preprocessed = preprocessor.transform(y)

            features_columns=preprocessor.get_feature_names_out()

            df_preprocess_x = pd.DataFrame(x_preprocessed, columns=features_columns)
            df_preprocess_y = pd.DataFrame(y_preprocessed, columns=features_columns)

            return df_preprocess_x, df_preprocess_y,
        
        except Exception as e:
            raise Exception(e)
        
    def models_params(self):

        try:
           
           models={
            #   'tree':DecisionTreeClassifier(),
              'gbc':GradientBoostingClassifier(),
            #   'xgboost': XGBClassifier(),
            #   'random_forest':RandomForestClassifier(),
            #   'svc': SVC(),
            #   'naive_bayes':GaussianNB(),
            #   'k_neighbor':KNeighborsClassifier(),
            #   'BaggingClassifier': BaggingClassifier(),
            #   'catboost':CatBoostClassifier()
    }
           params={
                'tree': {
                    'max_depth': [10, 20, 30, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],

            },
                'gbc':{
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
            },
                'random_forest': {
                    'bootstrap': [True, False],
                    'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                    'max_features': ['auto', 'sqrt'],
                    'min_samples_leaf': [1, 2, 4],
                    'min_samples_split': [2, 5, 10],
                    'n_estimators': [100, 200, 400, 600, 800, 1000, 1500, 2000],        
            },
                'naive_bayes':{

            },
                'k_neighbor':{
                    'n_neighbors':[1,2,3,4,5,6,8,10],
                    'weights':['uniform', 'distance'],
                    'leaf_size':[5,15,30,45,60,75,90],
                    'n_jobs':[-1],
            },
                'xgboost':{
                    'n_estimators':[100, 200, 300, 400, 500],
                    'learning_rate': [0.01, 0.1,0.3],
                    'max_depth': [2,4,6,8,10],
                    'min_child_weight': [1,3,5,7,10],
                    'gamma': [0, 5],
            },

                'BaggingClassifier':{
                    'n_estimators': [10,30,50,70,100],
                    'max_sample':[0.01,1],

            },    
                'svc':{
                    'kernel':['linear','poly','rbf','sigmoid'],
                    'C': 1.0, 
                    'gamma':['scale','auto']
            },
                
                'catboost':{
                    'learning_rate':[0.01, 0.1],
                    'depth':[1,6],
                    'loss_function':['MultiClass']
            },
            }
           
           return models, params
           
        except Exception as e:
            raise Exception(e)
        
    def metrics_score(y_test, y_pred):
        try:

            accuracy = round(accuracy_score(y_test, y_pred),2)
            precision = round(precision_score(y_test, y_pred),2)
            recall = round(recall_score(y_test, y_pred),2)
            f1 = round(f1_score(y_test, y_pred),2)
        
            return accuracy,precision,recall,f1
        
        except Exception as e:
            raise e