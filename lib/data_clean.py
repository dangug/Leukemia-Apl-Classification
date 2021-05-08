import pandas as pd
import numpy as np
from joblib import dump, load

class DataClean():
    """Class contains clean data job for our case """
    
    def __init__(self, dataframe, unique_key=None):
        self.init_df = dataframe
        self.unique_key = unique_key
        
    def get_Id(self):
        """return id, can be used for concatenate with prediction at the end """
        id_list = self.init_df["Patient_numbers"]
        return(id_list)
    
    def clean_job(self):
        """clean data job, return clean dataframe """
        df = self.init_df
        df.columns = df.columns.str.replace(" ", "_").str.replace("/","_")
        df = df.rename(columns={"Témoin_Cas_(0_1)": "target"})
        df = df.drop(columns=["Patient_numbers"])
        cols = df.columns[df.dtypes.eq(object)]
        df = df.replace("<","").replace(">","").replace("/","").replace(" ","_").replace("=","")
        df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
        
        ## creating Databrick joblib to set all the fillNa data on an object, usable for model in pordution
        result = df.append(pd.Series(name='indexNan')) ## create a last row with non values
        df_clean = result.fillna(result.median()) ## Apply fill na
        databrick = df_clean.tail(1) ## get all the Na filled value in last row
        databrick = databrick.to_dict('records') 
        databrick = databrick[0] ## databrick now is a dict 
        if self.unique_key != None:
            path = str(f'../Model/databrick_model_{self.unique_key}.joblib')
            dump(databrick, path)
        df_clean = df_clean[:-1] ## remove last row created before
        return df_clean

    def prod_clean_job(self, databrick):
        df = self.init_df
        df.columns = df.columns.str.replace(" ", "_").str.replace("/","_")
        df = df.rename(columns={"Témoin_Cas_(0_1)": "target"})
        df = df.drop(columns=["Patient_numbers"])
        cols = df.columns[df.dtypes.eq(object)]
        df = df.replace("<","").replace(">","").replace("/","").replace(" ","_").replace("=","")
        df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
        df_clean = df.fillna(value=databrick)
        return df_clean
