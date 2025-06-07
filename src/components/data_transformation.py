import numpy as np
import pandas as pd
import sys, os

from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    """
    This class is responsible for transforming the data.
    It includes feature engineering, scaling, and saving the preprocessor object.
    """
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            df = df.copy()

            # Step 1: Convert date to datetime if available
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')

            # Step 2: Create new features
            if set(['24h_volume', 'mkt_cap']).issubset(df.columns):
                df['liquidity_score'] = df['24h_volume'] / df['mkt_cap']

            if set(['24h', 'price']).issubset(df.columns):
                df['price_change_ratio'] = df['24h'] / df['price']

            if set(['24h_volume', 'price']).issubset(df.columns):
                df['volume_to_price'] = df['24h_volume'] / df['price']

            if 'symbol' in df.columns:
                stable_coins = ['USDT', 'USDC', 'BUSD']
                df['is_stable_coin'] = df['symbol'].apply(lambda x: 1 if x in stable_coins else 0)

            return df
        
        except Exception as e:
            raise CustomException(e, sys)

    def get_data_transformer_object(self):

        """
        This function creates a preprocessor object that will be used to transform the data.
        It includes numerical feature scaling and imputation.
        """
        try:
            numerical_features = ['price', '1h', '24h', '7d', '24h_volume', 'mkt_cap','price_change_ratio', 
                                  'volume_to_price', 'is_stable_coin']
            logging.info("Creating numerical feature list for preprocessing")

            # Numerical pipeline
            numerical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            logging.info(f"Numerical columns: {numerical_features}")

            # Combined preprocessor
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numerical_pipeline, numerical_features)
                ]
            )

            return preprocessor
            
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Data Transformation initiated")

            # Feature engineering
            train_df = self.feature_engineering(train_df)
            test_df = self.feature_engineering(test_df)

            # Get preprocessor object
            preprocessor_obj = self.get_data_transformer_object()

            # Fit and transform the training data
            X_train = preprocessor_obj.fit_transform(train_df[['price', '1h', '24h', '7d', '24h_volume', 'mkt_cap',
                                                            'price_change_ratio', 'volume_to_price', 'is_stable_coin'
                                                            ]])
            y_train = train_df['liquidity_score'].values

            # Transform the test data
            X_test = preprocessor_obj.transform(test_df[['price', '1h', '24h', '7d', '24h_volume', 'mkt_cap',
                                                        'price_change_ratio', 'volume_to_price', 'is_stable_coin'
                                                        ]])
            y_test = test_df['liquidity_score'].values

            if os.path.exists(self.data_transformation_config.preprocessor_obj_file_path):
                os.remove(self.data_transformation_config.preprocessor_obj_file_path)


            # Save the preprocessor object
            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )

            return X_train, y_train, X_test, y_test, self.data_transformation_config.preprocessor_obj_file_path
        
        except Exception as e:
            raise CustomException(e, sys)