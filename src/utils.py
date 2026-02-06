import os
import sys 
import dill

import pandas as pd 
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

from src.exception import CustomException


def save_object(file_path, obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
    
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_model(X_train, y_train,X_test,y_test,models,param):
    try:
        report={}
        best_models={}
        for name, model in models.items():            
            para = param.get(name, {})

            if para:
                gs = GridSearchCV(model,para,cv=3,scoring='r2',n_jobs=-1)              
                gs.fit(X_train, y_train)
                best_model=gs.best_estimator_                

            else:
                model.fit(X_train, y_train)
                best_model = model

            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)           

            report[name] = test_model_score
            best_models[name] = best_model

        return report,best_models

    except Exception as e:
        raise CustomException(e, sys) 