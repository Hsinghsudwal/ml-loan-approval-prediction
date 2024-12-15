import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.utility import *
from src.constants import *

import os

class DataIngestion:
    def __init__(self):
        pass

    def dataIngestion(self, path):
        try:
            
            data=Utility.read_data(path)
            # print(data.head())
            
            # split into train test data
            train_data, test_data = train_test_split(data, test_size= TEST_SIZE, random_state=42)
            # create folder to store data
            raw_data_path = os.path.join(OUTPUT,RAW_FOLDER)
            os.makedirs(raw_data_path,exist_ok=True)
            # save train test to csv
            train_data.to_csv(os.path.join(raw_data_path,str(TRAIN_DATA_FILE)))
            test_data.to_csv(os.path.join(raw_data_path,str(TEST_DATA_FILE)))
            print('Data Ingestion Completed')
            return train_data, test_data

        except Exception as e:
            raise Exception(e)