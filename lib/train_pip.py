from lib.data_clean import DataClean
from lib.classifier_trainer import ClassifierJob

from sklearn.ensemble import GradientBoostingClassifier
from joblib import dump
import datetime
import pandas as pd
import numpy as np

class custom_train_pip():
    """Train model and drop it with unique key"""
    
    def __init__(self,df_init):
        self.df = df_init
        self.df_clean = None
        self.model_train = None
        self.unique_key = str(datetime.datetime.now()).replace(" ","_").replace(":","_").replace(".","_")
        
    def clean_job(self):
        try:
            cleandata = DataClean(self.df)
            self.df_clean = cleandata.clean_job() ### SET cleandata.clean_job(self.unique_key) if databricks needed
            return self.df_clean
        except Exception as e:
            raise Exception(f"clean failed : {e}")
    
    def retrain_job(self, model_init, test_size=0.2, seed=69, learning_curve_mod=False, normalize=False, 
                    list_col_name_drop=None):
        try:
            cl = ClassifierJob(self.df_clean,model_init)
            if list_col_name_drop != None:
                x_full, y_full = cl.split_features_target(list_col_name_drop)
            else:
                x_full, y_full = cl.split_features_target()
            self.model_train = cl.fit_and_eval(test_size, seed,
                                            learning_curve_mod=learning_curve_mod, normalize=normalize)
            return self.model_train
        except Exception as e:
            raise Exception(f"Train failed : {e}")
    
    def drop_model(self):
        if (self.model_train != None):
            try:
                
                path = str(f'../Model/clf_model_{self.unique_key}.joblib')
                dump(self.model_train, path)
                return str(f"Model succesfully Create at {self.unique_key}")
            except Exception as e:
                raise Exception(f"Create model failed : {e}")
        else:
            raise Exception(f"Create model failed : self.model_train == None")