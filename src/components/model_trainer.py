import os
import sys

from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import (

    AdaBoostClassifier,
    GradientBoostingRegressor,
    RandomForestRegressor,
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfiguration:
    train_model_file_path = os.path.join("artifact","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfiguration()

    def initiate_model_trainer(self,train_array,test_array):
        try:

            logging.info("------model tranier started---------------")
            X_train,y_train,X_test,y_test = (

                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models={

                "Random forest": RandomForestRegressor(),
                "Linear Regression": LinearRegression(),
                "Gradient Boost": GradientBoostingRegressor(),
                "K-Nearest classifier": KNeighborsRegressor(),
                "xgboos" : XGBRegressor(),
                "catboost":CatBoostRegressor(verbose=False),
            }

            params={

                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    #'splitter': ['best', 'random'],
                    #'max_features': ['sqrt', 'log2', None]
                },
                'Random foreest': {
                    #cititerion: ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                    #'max_features': ['sqrt', 'log2', None]
                },
                'Gradient Boost': {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                    #'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    #'max_features': ['sqrt', 'log2', None],

                },
                'Linear Regression': {},
                'K-Nearest classifier': {
                    'n_neighbors': [5, 7, 9, 11],
                    #'weights': ['uniform', 'distance'],
                    #'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
                },
                'xgboos': {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                'catboost': {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                    'depth': [6, 8, 10]
                }

            }   


            model_report:dict=evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,params=params)

            # To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            # To get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
        
            logging.info(f"Best model found on both training and testing")

            save_object(
                file_path=self.model_trainer_config.train_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test,predicted)
            return r2_square

        except Exception as e:
            raise CustomException(e,sys)
