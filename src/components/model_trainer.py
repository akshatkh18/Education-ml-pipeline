import os
import sys 
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import(
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost  import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts", "model.pkl")
    catboost_log_dir = os.path.join("artifacts", "catboost_logs")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
        os.makedirs(self.model_trainer_config.catboost_log_dir, exist_ok=True)


    def initiate_model_trainer(self, train_arr,test_arr):
            try:
                logging.info("Splitting training and test input data")
                X_train, y_train, X_test, y_test=(
                    train_arr[:,:-1],
                    train_arr[:, -1],
                    test_arr[:,:-1],
                    test_arr[:, -1]
                )
                
                models={
                    "Linear Regression": LinearRegression(),
                    "Lasso": Lasso(),
                    "Ridge": Ridge(),
                    "K-Neighbors Regressor": KNeighborsRegressor(),
                    "XGBRegressor": XGBRegressor(),
                    "Random Forest Regressor": RandomForestRegressor(),
                    "CatBoosting Regressor": CatBoostRegressor(verbose=False, train_dir=self.model_trainer_config.catboost_log_dir),
                    "AdaBoost Regressor": AdaBoostRegressor(),
                    "Decision Tree":DecisionTreeRegressor(),
                    "Gradient Boosting":GradientBoostingRegressor()
                }

                params={
                    "Decision Tree":{
                        'criterion':['squared_error', 'absolute_error', 'friedman_mse', 'poisson'],
                        'splitter':['best', 'random'],
                        'max_features':['sqrt', 'log2'],
                    },
                    "Random Forest Regressor":{
                        'criterion':['squared_error', 'absolute_error', 'friedman_mse', 'poisson'],
                        'n_estimators':[8, 16, 32, 64, 128, 256],
                        'max_features':['sqrt', 'log2'],

                    },

                    "Gradient Boosting":{
                        "n_estimators":[8, 16, 32, 64, 128, 256],
                        "learning_rate":[0.1,0.01,0.001,0.0001],
                        'subsample':[0.6,0.7,0.75,0.8,0.8,0.85,0.9],
                        'loss':['squared_error', 'absolute_error', 'huber', 'quantile'],
                        'max_features':['sqrt', 'log2'],
                        'criterion':['squared_error', 'friedman_mse'],
                    },

                    "Linear Regression":{},
                    "Lasso": {},
                    "Ridge": {},

                    "K-Neighbors Regressor":{
                        "n_neighbors":[5,7,9,11],
                        "weights":['uniform','distance'],
                        'algorithm':['ball_tree', 'kd_tree','brute']
                    },

                    "XGBRegressor":{
                        "learning_rate":[0.1,0.01,0.001,0.0001],
                        "n_estimators":[8, 16, 32, 64, 128, 256]
                    },
                    "CatBoosting Regressor":{
                        'depth':[6,8,10],
                        "learning_rate":[0.1,0.01,0.001,0.0001],
                        'iterations':[30,50,100],
                    },

                    "AdaBoost Regressor":{
                        'learning_rate':[0.1,0.01,0.001,0.0001],
                        'loss':['linear','square','exponential'],
                        "n_estimators":[8, 16, 32, 64, 128, 256]
                    }
                }
                
                model_report,best_models=evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models, param=params)

                best_model_name = max(model_report, key=model_report.get)
                best_model = best_models[best_model_name]
                best_model_score = model_report[best_model_name]

                if best_model_score<0.6:
                    raise CustomException("No Best Model Found",sys)
                
                logging.info(f"Best model found on both training and testing dataset")

                save_object(
                    file_path=self.model_trainer_config.trained_model_file_path,
                    obj=best_model
                )

                predicted=best_model.predict(X_test)
                r2_square=r2_score(y_test,predicted)
                return r2_square             

            except Exception as e:
                raise CustomException(e,sys)
