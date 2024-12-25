import pandas as pd
import numpy as np
import joblib
import json
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from flask import Flask, request, jsonify

model = joblib.load(open('best_model.joblib','rb'))

app = Flask('Loan Approval')

def processing(x):

    numerical_features = x.select_dtypes(include=['int64', 'float64']).columns#.tolist()
    categorical_features = x.select_dtypes(include=['object']).columns#.tolist()

    numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())  
                ])
    categorical_transformer=Pipeline(steps=[
    ('ohe', OneHotEncoder()),
                ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer,categorical_features),
            ])
            
    preprocessed = preprocessor.fit_transform(x)
    features=preprocessor.get_feature_names_out()
    
    df_preprocess = pd.DataFrame(preprocessed, columns=features)
    cat = df_preprocess.select_dtypes(include='object')

    for i in cat:
        df_preprocess[i]=LabelEncoder().fit_transform(df_preprocess[i])

    return df_preprocess

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
   
    df=pd.DataFrame([data])

    processed=processing(df)

    datapass = processed.values.reshape(1,-1)

    prediction = model.predict(datapass)
    
    status=""
    if prediction == 1:
        status = "Approved"

    else:
        status = "Rejected"
    
    result={
        int(prediction):(status)
    }
    
    return jsonify([result])

if __name__ == "__main__":
    app.run(debug = True, host = '0.0.0.0', port = 9696)
