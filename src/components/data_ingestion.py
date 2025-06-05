import pandas as pd
import os
import sys
from src.exception import CustomException
from src.logger import logging

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

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

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Train and test data saved to: %s and %s",
                         self.ingestion_config.train_data_path, 
                         self.ingestion_config.test_data_path)

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
    print(f"Train Data Path: {train_data}")
    print(f"Test Data Path: {test_data}")
    print(f"Raw Data Path: {raw_data}")