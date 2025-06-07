import os, sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

from sklearn.metrics import mean_squared_error, r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array, preprocessed_path):
        try:
            logging.info("Splitting training and testing data")
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            models = {
                "Linear Regression": LinearRegression(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "KNeighbors": KNeighborsRegressor(),
                "SVR": SVR(),
                "XGBoost": XGBRegressor(),
                "CatBoost": CatBoostRegressor(verbose=0)
            }

            model_report:dict = evaluate_models(X=X_train, y=y_train, X_test=X_test, y_test=y_test, models=models)
            logging.info("Starting model training")

            best_model_score = max(model_report.values(), key=lambda x: x['R2 Score'])
            best_model_name = [name for name, metrics in model_report.items() if metrics == best_model_score][0]

            best_model = models[best_model_name]
            logging.info(f"Best model found: {best_model_name} with R2 Score: {best_model_score['R2 Score']}")

            if best_model_score['R2 Score'] < 0.6:
                raise CustomException("No suitable model found with R2 Score above 0.6", sys)
            logging.info("Best model found and validated")

            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path, 
                obj=best_model)
            logging.info(f"Model saved at {self.model_trainer_config.trained_model_file_path}")

            predicted = best_model.predict(X_test)

            mse = mean_squared_error(y_test, predicted)
            r2 = r2_score(y_test, predicted)
            logging.info(f"Model evaluation completed with MSE: {mse}, R2 Score: {r2}")

            return best_model_name, model_report, mse, r2

            
        except Exception as e:
            raise CustomException(e, sys)
