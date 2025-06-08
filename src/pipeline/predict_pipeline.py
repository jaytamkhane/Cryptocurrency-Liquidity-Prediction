import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        model_path = 'artifacts/model.pkl'
        preprocessor_path = 'artifacts/preprocessor.pkl'
        model= load_object(file_path=model_path)
        preprocessor = load_object(file_path=preprocessor_path)
        
        try:
            data_scaled = preprocessor.transform(features)
            prediction = model.predict(data_scaled)
            
            return prediction
        
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self, 
                 price: float,
                 h1: float,        # 1h
                 h24: float,       # 24h
                 d7: float,        # 7d
                 volume_24h: float,
                 mkt_cap: float,
                 price_change_ratio: float,
                 volume_to_price: float,
                 is_stable_coin: int):

        self.price = price
        self.h1 = h1
        self.h24 = h24
        self.d7 = d7
        self.volume_24h = volume_24h
        self.mkt_cap = mkt_cap
        self.price_change_ratio = price_change_ratio
        self.volume_to_price = volume_to_price
        self.is_stable_coin = is_stable_coin

    def get_data_as_dataframe(self):
        try:
            data = {
                'price': [self.price],
                '1h': [self.h1],
                '24h': [self.h24],
                '7d': [self.d7],
                '24h_volume': [self.volume_24h],
                'mkt_cap': [self.mkt_cap],
                'price_change_ratio': [self.price_change_ratio],
                'volume_to_price': [self.volume_to_price],
                'is_stable_coin': [self.is_stable_coin]
            }

            return pd.DataFrame(data)
        
        except Exception as e:
            raise CustomException(e, sys)