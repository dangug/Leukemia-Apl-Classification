from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.pipeline import make_pipeline, Pipeline

import pandas as pd
import numpy as np


class Pca_vectors():

    def __init__(self, x_full):
        self.X_full = x_full
        
        self.pca = None
        self.scaler = None
        
    ##### Accessor #####
    @property
    def get_pca_obj(self):
        return self.pca

    
    @property
    def get_scaler_obj(self):
        return self.scaler
    
   
    def reduce_dimension_train(self, number_of_dim=2, random_state=69):
        pca = PCA(n_components=number_of_dim, random_state=random_state)
        reduced_features = pca.fit_transform(self.X_full)

        ## Normalize matrice
        scaler = PowerTransformer()
        reduced_features = scaler.fit_transform(reduced_features)
        
        self.scaler = scaler
        self.pca = pca
        
        return reduced_features
    
    def reduce_dimension_prod(self, X_input):
        pca = self.pca
        scaler = self.scaler
        X_input = pca.transform(X_input)
        X_input = scaler.transform(X_input)
        return X_input

    def generate_pipeline(self,model_unserpervised):

        preprocess = make_pipeline(PCA(n_components=2, random_state=69), 
                                   PowerTransformer())
        pipeline_build_not_fit = make_pipeline(preprocess,model_unserpervised)
        return  pipeline_build_not_fit