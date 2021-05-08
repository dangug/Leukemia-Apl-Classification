import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

from lib.plot.explainers import shap_explainer, shap_explainer_cat_pipeline, eli5_explainer
from lib import data_explore

class ConsumeModelClf():
    """Use Model Created on this project to make prediction """
    
    def __init__(self, df_inputs_clean, model_fited, validation=False, true_y=None):
        """params:
        df_inputs_clean : A pandas dataframe with no target and index columns
        prod_model : Model or sklearn pipline fited
        validation : false by default, usable to validation a trained model or pipline. type bool
        true_y : array of validation y data  type np.array 
         """
        self.df = df_inputs_clean 
        self.model = model_fited
        self.validation = validation
        self.Id_list = None
        self.true_y = true_y
        self.Prediction_result = None
        self.Prediction_proba = None

    def get_Id_list(self):
        """Get List of Patients Numbers (ID)"""
        try:
            return self.Id_list
        except Exception as e:
            raise Exception(f"get ID failed : {e}")
            
    def predict_job(self):
        """load model as describe in init and return numpy list of prediction and confidence"""  
        try:
            self.Prediction_result = self.model.predict(self.df)
            self.Prediction_proba = self.model.predict_proba(self.df)
            return self.Prediction_result, self.Prediction_proba
        except Exception as e:
                raise Exception(f"Prediction failed : {e}")
        
    def show_result(self):
        """Return list Id, Prediction, confidance as pandas DataFrame"""
        try:
            final_df = pd.DataFrame()
            self.Id_list = self.df.index
            final_df["Id"] = self.Id_list
            final_df["Predictions"] = self.Prediction_result

            count = 0
            for index, pred_val in zip(final_df.index, final_df["Predictions"]):
                if pred_val==0:
                    final_df.at[index,"Trust%"] = round(self.Prediction_proba[count][0]*100,2)
                else:
                    final_df.at[index,"Trust%"] = round(self.Prediction_proba[count][1]*100,2)
                count +=1

            if self.validation:
                final_df["True_value"] = self.true_y
                ac = metrics.accuracy_score(self.true_y, self.Prediction_result)
                print("Score Validation :",round(ac,2))
            return final_df
        except Exception as e:
            raise Exception(f"return result failed : {e}")

    def explain(self, shap_mod=True, list_cat_features=None,list_num_features=None,pipeline_is_normalized=False):
        """params:
        shap_mod= True if model is not pipeline, other params must be set by default
        if shap_mod = False :
            list_cat_features=None, cat list paste during fit
            list_num_features=None, num list paste during fit
            pipeline_is_normalized=False  sk learn pipeline was normalized ? 
        return: figure explainer of prediction.    
        """
        if shap_mod :
            fig = shap_explainer(self.df,self.model,show_dependencies=False)
        else:
            fig = shap_explainer_cat_pipeline(self.df, list_cat_features, list_num_features, self.model, pipeline_is_normalized=pipeline_is_normalized)
        return fig 
