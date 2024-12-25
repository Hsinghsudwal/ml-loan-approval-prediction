import pandas as pd
import numpy as np
import json
import joblib
from src.utility import *
from src.constants import *
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix,roc_curve, auc
from sklearn.model_selection import KFold, cross_val_score
# from sklearn.metrics import roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
import mlflow
from mlflow.models import infer_signature


class ModelEvaluation:
    def __init__(self):
        pass

    mlflow.set_tracking_uri("http://127.0.0.1:8080")
    mlflow.set_experiment(EXPERIMENT_NAME)

    def modelEvaluation(self, best_estimator,best_param,x_train,y_train,x_test,y_test):
        try:
           #    modelevaluation
           model_evaluation_path = os.path.join(OUTPUT,EVALUATION_FOLDER)
           os.makedirs(model_evaluation_path,exist_ok=True)
           
           with mlflow.start_run(nested=True):
                # mlflow.autolog()
                mlflow.set_tag("Project", PROJECT_NAME)
                mlflow.set_tag("Dev", AUTHOR)

                # model_path=os.path.join(OUTPUT,MODEL_FOLDER,str(BEST_MODEL_NAME))
        #         print(dir)
                # model=joblib.load(model_path,"r")
                # ypred=model.predict(x_test)
                model=best_estimator
                ypred=model.predict(x_test)

                mlflow.log_params(best_param)

                print('train score: ' ,model.score(x_train, y_train))
                # print('Test accuracy: ', accuracy)

                accuracy,precision,recall,f1=Utility.metrics_score(y_test,ypred)

                metric_dic={'accuracy':accuracy,
                            'precision':precision,
                            'recall':recall,
                            'f1':f1
                            }
                
                print(metric_dic)

                metric_file=os.path.join(model_evaluation_path,str(EVAL_JSON_NAME))

                mlflow.log_metric('accuracy', accuracy )
                mlflow.log_metric('precision', precision )
                mlflow.log_metric('recall', recall )
                mlflow.log_metric('f1', f1 )

                classified_report=classification_report(y_test, ypred, output_dict=True)

                df=pd.DataFrame(classified_report).transpose()
                classification = df.to_csv(os.path.join(model_evaluation_path,str(CLASSI_REPORT)))

                cm=confusion_matrix(y_test,ypred)
                sns.heatmap(cm, annot=True, fmt='g')
                plt.xlabel('Predicted Labels')
                plt.ylabel('True Labels')
                
                # Log confusion matrix artifact
                cm_path = f'confusion_matrix_{model}.png'
                cmpath=os.path.join(model_evaluation_path,str(cm_path))
                plt.savefig(cmpath)
                mlflow.log_artifact(cmpath)

                # Calculate the ROC curve
                fpr, tpr, thresholds = roc_curve(y_test, ypred)

                # Calculate the AUC score
                auc_score = auc(fpr, tpr)
                x1=np.linspace(0,1,100)
                # Plot the ROC curve
                plt.plot(fpr, tpr, label="ROC curve (AUC = %0.2f)" % auc_score)
                plt.plot(x1,x1,label='baseline')
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title("ROC curve classification")
                plt.legend()

                # Log AUC Score artifact
                auc_path = f'auc_score_{model}.png'
                aucpath=os.path.join(model_evaluation_path,str(auc_path))
                plt.savefig(aucpath)
                # mlflow.log_artifact(aucpath)

                mlflow.log_artifact(__file__)

                signature = infer_signature(x_test,model.predict(x_test))

                mlflow.sklearn.log_model(model, artifact_path=MODEL_NAME,signature=signature)
                

                with open(metric_file,'w') as f_in:
                 json.dump(metric_dic,f_in,indent=4)

                print('Model Evaluation Completed')
                return metric_dic

        except Exception as e:
            raise Exception(e)
        
