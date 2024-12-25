# ml-loan-approval-prediction

## Table of Content
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Setup](#setup)
- [Development](#development)
- [Orchestration](#orchestration)
- [Model Experiments](#model-experiments)
- [Retraining](#retraining)
- [Deployment](#deployment)
- [Next Step](#next-step)



## Problem Statement:

**Content**
A loan application is used by borrowers to apply for a loan. Through the loan application, borrowers reveal key details about their finances to the lender. The loan application is crucial to determining whether the lender will grant the request for funds.

**Problem Statement**
When it comes to banks, they have go through the loan applications to filter the people who can be granted with loan or not. In which they have to look after various factors such as cibil score, income, employment status, loan term, loan amount, assets value, and education which is time-consuming process. We wants to automate it and increase his bank’s efficiency. 

**Objective**
The idea is to build an Machine Learning model to predict whether the candidate’s profile is relevant or not using key features. Then create Flask application that the bank can use to classify if a user can be granted a loan or not. Deploy on docker container.


## Setup
**Installation: Clone the repository** `git clone https://github.com/Hsinghsudwal/ml-loan-approval-prediction.git`

1. Set up Environment for managing libraries and running python scripts.
   ```bash
   pip install pipenv
   ```
2. **Activate environment**
   ```bash
   pipenv shell 
   ```
   This will create pipfile and pipfilelock within the environment.

3. **Initialize a New Pipenv Environment and Install Dependencies**:
   ```bash
   pipenv install 
   ```
   `pipenv install -r requirements.txt`
   

## Development
**Run Jupyter Notebook**:
From within Pipenv virtual environment, now can run the Notebook. On the terminal, from your main project directory to
```bash
   cd notebook
   jupyter lab
   ```
### Data Collection:
Gather historical loan data, which should include both approved and rejected loans.
### Preprocessing:
Clean and preprocess the data by handling missing values, encoding categorical variables, and scaling/normalizing numerical features.
### Exploratory Data Analysis: EDA
Check the number of rows and columns, data types, and the presence of missing values.
Examine the distribution of the target variable, which is typically whether a loan was approved or rejected. Creating a bar plot to visualize the distribution etc.
Explore the distribution of individual features, both numerical and categorical. Use histograms, box plots, or bar charts to visualize the data. Identify outliers using box plots.
### Data Splitting:
Split your data into training and testing sets to evaluate your model's performance.
### Model Selection:
Choose an appropriate machine learning algorithm. I have used common algorithms for this binary classification problem like Logistic Regression, Random Forests, Support Vector Machines and XGBoost.
### Model Training:
Train your chosen model on the training data.
### Model Evaluation :
To evaluate the model's performance I have used appropriate metrics, such as accuracy, precision, recall, F1-score.
### Hyper-Tuning:
Optimize the model's hyperparameters to improve its performance. This can be done through techniques like grid search Grid Search cv.


## Orchestration
**Run Python Scripts**:
1. Save the working jupyter notebook to `src` directory once completed.
2. Convert jupyter notebook to script by

   `jupyter nbconvert --to script notebook.ipynb`

3. Edit script into modular code. To performe pipeline functions: which are located `src/steps- data_ingestion, data_validation, data_transformation,model_trainer, and model evaluation`. To run scripts from main project directory:  
   ```bash
   python main.py
   ```
This script will output verious files and model to save into `output` and use in deployment

## Model Experiments
**Mlflow:** using tracking runs and its storage allow different parameters to register best model and put that model to production.

**Steps:**
1. mlflow tracking server:
`mlflow server --backend-store-uri sqlite:///mlruns.db --default-artifact-root artifacts -p 8080`

2. Set our tracking server uri for logging
`mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")`

3. Create a new MLflow Experiment
`mlflow.set_experiment("experiment_name")`

4. Start an MLflow run
```
with mlflow.start_run():
    # Log the hyperparameters
    mlflow.log_params(params)

    # Log the loss metric
    mlflow.log_metric("accuracy", accuracy)

    # Set a tag that we can use to remind ourselves what this run was for
    mlflow.set_tag("Training Info", "Basic model")

    # Infer the model signature
    signature = infer_signature(X_train, lr.predict(X_train))

    # Log the model
    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model_path",
        signature=signature,
        input_example=X_train,
    )
```

5. Load the model back for predictions as a generic Python Function model.

    `loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)`

    `predictions = loaded_model.predict(X_test)`


## Retraining
**In order to perform retraining:**

Required:

    1. model needs to register with model mlflow registery.
    2. model needs to in `Staging` for testing.
    3. Script `test_pipe.py` will check if model is register and exists.
    4. Load the model and perform basic score and compare to threshold if model needs retraining.
    5. Model stage to `Production`

To see if retraining is required run scripts from main project directory:  
   ```bash
   python test_pipe.py
   ```


## Deployment
**Creating Docker image and Flask application**:

**Steps:** From the main project directory create `directory deployment`. Then `cd deployment`.
1. Create script to run flaskscript within the Pipenv virtual environment. When completed run via
   ```bash
   python predict.py
   ```

2. **Test APIs**: Create `test.py` to test the model contains a `input` data. In new terminal, pass this data.
   ```bash
   python deploment/test.py
   ```
3. **DOCKER:** From the main project directory `cd deployment`.
* Create docker file for container!
 ```
    FROM python 
    RUN
    WORKDIR
    COPY
    RUN
    COPY
    EXPOSE
    ENTRYPOINT
```

4. Docker -check image to see if docker is install and/or working on your machine
   ```bash
   docker images
   ```
4. Build Docker
   ```bash
   docker build -t loan_approval .
   ```
5. Running Docker 
   ```bash
   docker run -it --rm -p 9696:9696 loan_approval
   ```
   or detach running flag -d

   `docker run -it -d --rm -p 9696:9696 loan_approval`

6. **Test your API**:
    After running the container, on the main project directory contains `api.py` file. Which contain sample. you can change the parameters to see the different results. By edit `api.py` or passing 

   ```bash
   python api.py
   ```


## Next Step
    - Best-practices
    - Tests
    - Try to deploy on cloud
    - Monitor the model performance
    - Ci/Cd pipline
