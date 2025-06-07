import os
import sys
import dill

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging

def save_object(file_path: str, obj: object):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
        logging.info(f"Object saved at {file_path}")
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X, y, X_test, y_test, models, params):
    try:
        report = {}
        
        for i in range(len(list(models))):
            model_name = list(models.keys())[i]
            para = params[list(models.keys())[i]]
            
            gs = GridSearchCV(estimator=models[model_name], param_grid=para, cv=3,
                n_jobs=-1, verbose=1)
            gs.fit(X, y)

            model = gs.best_estimator_
            model.fit(X, y)
            logging.info(f"Training {model_name} with parameters: {gs.best_params_}")
            
            
            # Predict on test data
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            report[model_name] = {
                "MSE": mse,
                "R2 Score": r2
            }

        return report
    
    except Exception as e:
        raise CustomException(e, sys)