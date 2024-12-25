import sys
from src.steps.data_ingestion import DataIngestion
from src.steps.data_validation import DataValidation
from src.steps.data_transformation import DataTransformation
from src.steps.model_trainer import ModelTrainer
from src.steps.model_evaluation import ModelEvaluation


def pipeline():
    try:
        path=r"data/loan_approval_dataset.csv"
        # file=r"output/model/model_gbc.joblib"
        data_ingest=DataIngestion()
        train_data,test_data=data_ingest.dataIngestion(path)

        data_validate=DataValidation()
        isfound,train_data,test_data=data_validate.dataValidation(train_data, test_data)

        preprocess=DataTransformation()
        xtrain,ytrain,xtest,ytest = preprocess.dataTransform(isfound, train_data, test_data)
        
        model_trainer=ModelTrainer()
        estimator, param, x_train, y_train, x_test, y_test = model_trainer.bestModel(xtrain,ytrain,xtest,ytest)

        model_evaluate=ModelEvaluation()
        model_evaluate.modelEvaluation(estimator,param,x_train,y_train,x_test,y_test)

        
    except Exception as e:
        raise Exception(e)
    

if __name__=="__main__":
    pipeline()
