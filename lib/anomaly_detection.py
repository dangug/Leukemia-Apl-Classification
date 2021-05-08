import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import plotly.express as px

from lib.unsupervised.dimension import Pca_vectors

class Anomaly_nature():
    
    def __init__(self, model, X_full, y_full, 
                 left_axes_limit, right_axes_limit,
                 reduce_dim=False):
        """
        params :
            model = Anomalies models IsolationForest, OneClassSVM
            X_full = dataframe or np.array(2d)
            y_full = target data as vector or dataframe
            left_axes_limit = left_axes value of fig
            right_axes_limit = right_axes value of fig
            reduce_dim : if true apply pca 2d and standardscaler on x_full
        """ 
        self.model = model
        self.X_full = X_full
        self.y_full = y_full
        self.reduce_dim = reduce_dim
        self.left_axes_limit = left_axes_limit
        self.right_axes_limit = right_axes_limit

        
    def build_anomalies_model(self, X_input=None, in_color='#FF0017', out_color='#BEBEBE'):
        """
        params :
            X_input = for consume model, pass input data as x_vec (for app usage)
            in_color, out_color = colors for scatters points

            return matplolib figure and self.model
        """ 

        if self.reduce_dim:
            pca_job = Pca_vectors(self.X_full)
            reduced_features = pca_job.reduce_dimension_train() 
        else:
            reduced_features = self.X_full


        self.model.fit(reduced_features)
        y_pred = self.model.predict(reduced_features)

        xx, yy = np.meshgrid(np.linspace(self.left_axes_limit, self.right_axes_limit, 100),
                             np.linspace(self.left_axes_limit, self.right_axes_limit, 100))

        Z = self.model.predict(np.c_[xx.ravel(), yy.ravel()]) ## Love
        Z = Z.reshape(xx.shape)

        fig = plt.figure()
        plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')
        colors = np.array([in_color, out_color])
        plt.scatter(reduced_features[:, 0], reduced_features[:, 1], s=3, color=colors[(y_pred + 1) // 2])

        model_name = type(self.model).__name__
        if model_name =='OneClassSVM':
            kernel_name = self.model.kernel
            plt.title(f"{model_name} : {kernel_name}")
        else:
            plt.title(f"{model_name}")

        if X_input is not None:
            if self.reduce_dim:
                X_input = pca_job.reduce_dimension_prod(X_input)
            plt.scatter(X_input[:, 0], X_input[:, 1])
            anomalie_prediction = self.model.predict(X_input)
            return self.model, fig, anomalie_prediction
        return self.model, fig
    
    def build_3dim(self, X_input=None, X_info=None):
        """
        params :
            X_input = for consume model, pass input data as x_vec (for app usage)
            X_info = ad information to data as Id or categories 
            return plotly 3d figure
        """
        
        pca_job = Pca_vectors(self.X_full)
        reduced_features = pca_job.reduce_dimension_train(number_of_dim=3) 
        
        df = pd.DataFrame(data=reduced_features, columns=["Axe1", "Axe2", "Axe3"])

        if X_info is not None:
            for c in X_info.columns:
                df[f'{c}'] = X_info[f'{c}']
        
        df['target'] = self.y_full
        
        if X_input is not None:
            X_input = pca_job.reduce_dimension_prod(X_input)
            df_inp = pd.DataFrame(data=X_input, columns=["Axe1", "Axe2", "Axe3"])
            df_inp['target'] = 2
            prod_df = pd.concat([df, df_inp], ignore_index=True)
            fig = px.scatter_3d(prod_df, x='Axe1', y='Axe2', z='Axe3', hover_data= df.columns,
                  color='target', symbol='target')

        else:
            fig = px.scatter_3d(df, x='Axe1', y='Axe2', z='Axe3', hover_data= df.columns,
                  color='target', symbol='target')


        return fig

    def build_pipeline(self,model_unserpervised):
        pipeline = Pca_vectors(self.X_full) ## x_full will note be used at this step
        pipeline = pipeline.generate_pipeline(self.model)
        pipeline.fit(self.X_full)
        return pipeline
