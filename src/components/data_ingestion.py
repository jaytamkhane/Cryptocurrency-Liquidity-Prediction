import pandas as pd
import numpy as np
import os
import sys
from src.exception import CustomException
from src.logger import logging

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer


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
    # Step 1: Ingest data
    obj = DataIngestion()
    train_data_path, test_data_path, raw_data_path = obj.initiate_data_ingestion()

    # Step 2: Transform data
    data_transformation = DataTransformation()
    X_train, y_train, X_test, y_test, preprocessor_path = data_transformation.initiate_data_transformation(
        train_data_path, test_data_path
    )

    # ✅ Step 2.5: Remove rows with NaN in y_train or y_test
    import numpy as np
    if np.isnan(y_train).any() or np.isnan(y_test).any():

        train_valid_idx = ~np.isnan(y_train)
        test_valid_idx = ~np.isnan(y_test)

        X_train = X_train[train_valid_idx]
        y_train = y_train[train_valid_idx]

        X_test = X_test[test_valid_idx]
        y_test = y_test[test_valid_idx]

    # Step 3: Combine X and y into arrays
    train_array = np.c_[X_train, y_train]
    test_array = np.c_[X_test, y_test]

    # Step 4: Model training
    model_trainer = ModelTrainer()
    best_model_name, model_report, mse, r2 = model_trainer.initiate_model_trainer(
        train_array, test_array, preprocessor_path
    )

print(f"Best model: {best_model_name}")
print(f"Mean Squared Error: {mse}")
print(f"R2 Score: {r2}")
