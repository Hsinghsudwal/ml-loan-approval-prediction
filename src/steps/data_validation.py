import pandas as pd
import numpy as np
import json
from scipy.stats import ks_2samp
from src.utility import *
from src.constants import *

import os

class DataValidation:
    def __init__(self):
        pass

    def dataValidation(self,train_data,test_data):
        try:
            # validate data
            drift_results={}
            for feature in train_data.columns:
                
                ks_stat, p_value = ks_2samp(train_data[feature], test_data[feature])
                drift_results[feature] = p_value
                
                if p_value < THRESHOLD:
                    # print(f'feature {feature}: (pvalue:{p_value})')
                    is_found =  True
                else:
                    
                    is_found = False
           
            drift_results.update({
            "drift_status":is_found})

            # create folder to store report
            data_validate_path = os.path.join(OUTPUT,VALIDATE_FOLDER)
            os.makedirs(data_validate_path,exist_ok=True)
            # save report to output
            report_file=os.path.join(data_validate_path,REPORT_VALIDATE_JSON)
              
            with open(report_file,'w') as f_in:
                json.dump(drift_results,f_in,indent=4)
            
            print('Data Validation Completed')
            return is_found,train_data,test_data

        except Exception as e:
            raise Exception(e)
