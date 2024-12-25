import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

def preprocess_pipe(data):

            try:
    
                data= data.drop(columns=['loan_id'], axis=1)
                data[' loan_status']=data[' loan_status'].map({' Approved':1, ' Rejected': 0})

                num_features = data.select_dtypes(include=['int64', 'float64']).columns
                cat_features = data.select_dtypes(include=['object']).columns
                num_col = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='mean')), 
                    ('scaler', StandardScaler())  
                    ])
                cat_col=Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),

                    ])
                preprocessor = ColumnTransformer(
                        transformers=[('num', num_col, num_features),
                                  ('cat', cat_col,cat_features),
                                ],
                                  remainder='passthrough'
                    )
                preprocessed = preprocessor.fit_transform(data)
                features=preprocessor.get_feature_names_out()
                df_preprocess = pd.DataFrame(preprocessed, columns=features)
                cat = df_preprocess.select_dtypes(include='object')

                for i in cat:
                    df_preprocess[i]=LabelEncoder().fit_transform(df_preprocess[i])

                x= df_preprocess.drop(columns=['num__ loan_status'], axis=1)

                y=df_preprocess['num__ loan_status']

                xtrain, xtest, ytrain, ytest = train_test_split(x,y, test_size= 0.2, random_state=42)

                x_train=xtrain.values
                y_train=ytrain.values
                x_test=xtest.values
                y_test=ytest.values

                return x_train, x_test, y_train, y_test
              
            except Exception as e:
                raise Exception(e)