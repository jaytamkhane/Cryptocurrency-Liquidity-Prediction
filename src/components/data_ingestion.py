import pandas as pd
import os
import sys
from src.exception import CustomException
from src.logger import logging

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entering the data ingestion method or component")
        try:
            df_16 = pd.read_csv('notebook/data/coin_gecko_2022-03-16.csv')
            df_17 = pd.read_csv('notebook/data/coin_gecko_2022-03-17.csv')
            logging.info("Dataset read as pandas DataFrame")

            # Combine the two DataFrames
            df = pd.concat([df_16, df_17], ignore_index=True)
            logging.info("DataFrames concatenated")

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Train Test Split initiated")

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            logging.info("Train and test sets created")

            # Perform feature engineering before saving
            data_transformation = DataTransformation()
            train_set = data_transformation.feature_engineering(train_set)
            test_set = data_transformation.feature_engineering(test_set)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")
            logging.info("Train and test data saved successfully")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
                self.ingestion_config.raw_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data, raw_data = obj.initiate_data_ingestion()
    
    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(train_data, test_data)