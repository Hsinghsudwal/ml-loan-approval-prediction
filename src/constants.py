# BASE:
PROJECT_NAME = "Loan Approval"
AUTHOR = "Harinder Singh Sudwal"

OUTPUT='output'

# Data Ingestion
TEST_SIZE = 0.2
RAW_FOLDER='raw'
TRAIN_DATA_FILE='train_data.csv'
TEST_DATA_FILE='test_data.csv'

# Data Validation
THRESHOLD=0.05
VALIDATE_FOLDER='validate'
REPORT_VALIDATE_JSON='report.json'

# Data Transformer
TRANSFORMATION_FOLDER='transformation'
X_TRAIN_PROCESS_DATA='x_train_process.csv'
Y_TRAIN_PROCESS_DATA='y_train_process.csv'
X_TEST_PROCESS_DATA='x_test_process.csv'
Y_TEST_PROCESS_DATA='y_test_process.csv'

# Model Trainer
MODEL_LIST='model_list.csv'
MODEL_FOLDER='model'
BEST_MODEL_NAME='best_model.joblib'
# MODELS_FILE='model.csv'

# Model Evaluation
EVALUATION_FOLDER='evaluate'
EVAL_JSON_NAME='model_eval_metrics.json'
CLASSI_REPORT= 'classification_report.csv'

# Mlflow
EXPERIMENT_NAME = 'predict_loan'
MODEL_NAME = "best_model"
ARCHIVED="Archived"
STAGE = "Staging"
PROD = "Production"
