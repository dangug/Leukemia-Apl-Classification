# Import class packages
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder, PowerTransformer

from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.impute import SimpleImputer

from sklearn.multiclass import OneVsRestClassifier
import random

# Import data vis packages
import matplotlib.pyplot as plt
import seaborn as sns

# interpretation Modèle
from lib.plot.explainers import shap_explainer, shap_explainer_cat_pipeline, eli5_explainer
from lib.plot import roc
from lib.plot import learning_curves 
from lib import data_explore
from lib.models import calibration
from lib.models import threshold


def MainAiJob(model, df_clean, target_name='target', catagorical_features=False,
                test_size=0.2, random_state=69, learning_curve_mod=True, normalize=False, list_col_name_drop=None, show_explainers=True,
                calibrate_model=False):
    """  For classification work with pandas DataFrame
    call jobs in 1 line of code, example by default : 

    model_fited = MainAiJob(model, df_clean, target_name='target', catagorical_features=False,
                test_size=0.2, random_state=69, learning_curve_mod=True, normalize=False, list_col_name_drop=None, show_explainers=True)
    return model_fited, core(class object)
    """
    if not catagorical_features:
        core = ClassifierJob(df_clean, model, target_name)
        core.split_features_target(list_col_name_drop=list_col_name_drop)
        model_fited = core.fit_and_eval(test_size, random_state, 
                                        learning_curve_mod=learning_curve_mod, normalize=normalize)
        if(show_explainers):
            try:
                core.model_explainer()
            except Exception as e:
                print(e)
        if(calibrate_model):
            model_fited = core.Calibrate_fited_model()
        return model_fited, core
    else:
        core = ClassifierJob_CatNum(df_clean, model, target_name)
        core.split_features_target(list_col_name_drop=list_col_name_drop)
        model_fited = core.fit_and_eval(test_size=test_size, random_state=random_state, 
                        learning_curve_mod=learning_curve_mod, normalize=normalize)
        if(show_explainers):
            try:
                core.model_explainer(model_fited, pipeline_is_normalized=True)
            except Exception as e:
                print(e)
        if(calibrate_model):
            model_fited = core.Calibrate_fited_model()
        return model_fited, core

class ClassifierJob():
    """  For classification binaries case, and only numerical dataset work with pandas DataFrame"""
    def __init__(self,df_clean, model,target_name='target',silence_mod=False):
        """  
        params :
        df_clean = full dataframe ready for train
        model = model observed
        target_name ='target' by default type str
        """
        self.df = df_clean
        self.target_name = target_name
        self.model = model
        self.silence_mod = silence_mod
        self.x_full = None
        self.y_full = None
        self.pipeline_build_not_fit = None
        self.model_fited = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test =None

    ##### Accessor #####
    @property
    def get_x_full(self):
        return self.x_full
    
    @property
    def get_y_full(self):
        return self.y_full
    
    @property
    def get_X_train(self):
        return self.X_train

    @property
    def get_y_train(self):
        return self.y_train
    
    @property
    def get_X_test(self):
        return self.X_test
    
    @property
    def get_y_test(self):
        return self.y_test
    
    @property 
    def get_pipeline_build_not_fit(self):
        return self.pipeline_build_not_fit

    def split_features_target(self, list_col_name_drop=None):
        """
        params :
        list_col_name_drop = None by default, if drop_features pass a list of columns to drop
                             ex : ["col1","col2"]
        return dataframes splited by features/target """
        try:
            self.y_full = self.df[self.target_name] # target
            self.x_full = self.df.drop(columns=[self.target_name]) # features
            if list_col_name_drop != None:
                self.x_full = self.x_full.drop(columns=list_col_name_drop)
        except Exception as e:
            raise Exception(f"Split features/target failed : {e}")        
        return self.x_full, self.y_full

    ##### Fit and Eval #####
    def train_task(self):
        ## Train task
        clf = self.pipeline_build_not_fit.fit(self.X_train, self.y_train)
        sc = clf.score(self.X_train, self.y_train)
        print("Score train :",round(sc,2))
        print("-----------")
        return clf
    
    def test_task(self,clf):
        ## Test task
        # print(self.y_test)
        res = clf.predict(self.X_test)
        ac = metrics.accuracy_score(self.y_test, res)
        # test_data = clf.predict_proba(self.X_test)
        # df = pd.DataFrame(test_data)
        # key = random.randint(1,10000)
        # df["prediction"] = res
        # df["true_value"] = self.y_test
        # df.to_csv(f"res{key}.csv")
        print("Score test :",round(ac,2))
        print("-----------")
        return res
    
    def auc_job(self,test_size):
        try : 
            splits = int(1/test_size)
            result, aucs = roc.roc_curve_cv(self.pipeline_build_not_fit, self.x_full, self.y_full, n_splits=splits)
            print(f"AUC: {round(np.mean(aucs),2)} (std:{round(np.std(aucs),4)}), (splits = {splits})")
            roc.plot_roc_curve_cv(result)
            plt.show()
        except Exception as e:
            print("-----------")
            print("No prob methods for this model : ",e)
            print("-----------")

    def cm_job(self, res):
        cm = confusion_matrix(self.y_test, res)
        df_cm = pd.DataFrame(cm)
        ax = plt.axes()
        sns.heatmap(df_cm, annot=True, fmt="d", linewidths=.5, ax=ax) 
        ax.set_title("Confusion matrix")
        plt.show()

    def clf_report(self, res):
        print("--------")
        print("Classification reporting")
        print(metrics.classification_report(self.y_test, res))

    def learning_report(self):
            try:
                learning_curves.plot_learning_curves(self.pipeline_build_not_fit, self.X_train, self.y_train, 
                                                     n_repeats=1, train_sizes=np.linspace(0.1,1,10), cv=5, 
                                                     scoring='f1', title= 'Learning curve')
                plt.show()
            except Exception as e:
                print("-----------")
                print("No learning curve for this model : ",e)
                print("-----------")

    def _builder_fit_and_eval(self,test_size,learning_curve_mod):
        """real fit and eval job, should not be called outside """
        ## Train task
        clf = self.train_task()
        
        ## Test task
        res = self.test_task(clf)

        if self.silence_mod:
            return clf

        ## Evaluate AUC only for some models
        self.auc_job(test_size)

        ## Confusion matrix
        self.cm_job(res)

        ## Classification report
        self.clf_report(res)

        ## Learning rate
        if learning_curve_mod:
            self.learning_report()
        return clf

    def fit_and_eval(self,test_size=0.2, random_state=69, learning_curve_mod=False, normalize=False):
        """ 
        Split the data on train test, Fit the model, test it, evaluate it and return it

        params :
        test_size  = the size of split, 0.2 by default type float
        random_state = 69 by default, make your process reproducible type int.
        learning_curve = False by default, long and expensive operation or huge dataset type bool.
        normalize = False by default, cover only standardscaler for all data type bool.

        return : model fited
        """

        seed = random_state
        np.random.seed(seed)

        ##Split data in train/test using stratify = take randomly the same sample number from each class.
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.x_full, self.y_full, test_size=test_size, 
                                                            random_state=seed, stratify=self.y_full)

        ######## normalize job #######
        if normalize:
            # preprocess = make_pipeline(StandardScaler(), SimpleImputer(strategy="median"))
            preprocess = make_pipeline(StandardScaler())
            self.pipeline_build_not_fit = make_pipeline(preprocess,self.model) ## var pipeline = standardscaler + model
        else:
            # preprocess = make_pipeline(SimpleImputer(strategy="median"))
            # self.pipeline_build_not_fit = make_pipeline(preprocess,self.model)
            self.pipeline_build_not_fit = self.model ## var pipeline = just our model


        self.model_fited = self._builder_fit_and_eval(test_size,learning_curve_mod)
        return self.model_fited  

    ##### explainer #####
    def model_explainer(self):
        """
        return : Figure of model explained
        """
        fig = shap_explainer(self.x_full,self.model_fited)
        return fig

    def Calibrate_fited_model(self,n_bins=5):
        """
        return : model calibrated and plot before and after
        """
        calibrated = calibration.calibrate(self.pipeline_build_not_fit, self.X_train, self.y_train)

        calibration.plot_reliability_diagram(self.model_fited, self.X_test, self.y_test, title='Uncalibrated',n_bins=n_bins)
        plt.show()
        calibration.plot_reliability_diagram(calibrated, self.X_test, self.y_test, title='Calibrated',n_bins=n_bins)
        plt.show()

        y_proba = self.model_fited.predict_proba(self.X_test)[:, 1]
        score_calibre = calibration.calibration_score(self.y_test, y_proba)
        print("Brier score  = ", score_calibre)

        return calibrated

    def optimize_model(self,model=None,TP=0, FP=200, TN=0, FN=100, minimize=False):
        """
        return : print of best treshold and the best cost
        """
        if model == None:
            model= self.model
        cost, best_threshold, best_cost = threshold.optimize_thresholdcv(model, self.X_test, self.y_test,
                                                                 TP=TP, FP=FP, TN=TN, FN=FN, n_repeats=5,minimize=minimize)
        threshold.plot_cost_curve(cost, best_threshold, best_cost)
        print(f"Best threshold: {best_threshold}\nBest cost: {best_cost}")

class ClassifierJob_CatNum(ClassifierJob):
    """  For classification binaries case, and categorical + numerical dataset work with pandas DataFrame"""

    def __init__(self,df_clean, model, target_name='target'):
        """  
        params :
        df_clean = full dataframe ready for train
        model = model observed
        target_name ='target' by default type str
        """
        super().__init__(df_clean=df_clean, model=model, target_name=target_name)

    def get_columns_list(self):
        list_cat_features = []
        dataviz = data_explore.DataExplore(self.df,self.target_name)
        list_cat_features = dataviz._infer_cat_columns()
        if self.target_name in list_cat_features:
            list_cat_features.remove(self.target_name)
        list_num_features = []
        list_num_features = dataviz._infer_num_columns()
        if self.target_name in list_num_features:
            list_num_features.remove(self.target_name)
        return list_cat_features, list_num_features     
    
    def fit_and_eval(self,test_size=0.2, random_state=69,
                 learning_curve_mod=False, normalize=False):
        """ 
        Split the data on train test, Fit the model, test it, evaluate it and return it
        params :
        test_size  = the size of split, 0.2 by default type float
        random_state = 69 by default, make your process reproducible type int.
        learning_curve = False by default, long and expensive operation or huge dataset type bool.
        normalize = False by default, cover only standardscaler for all data type bool.
        
        return : model fited
        """
        seed = random_state
        np.random.seed(seed)

        list_cat_features, list_num_features = self.get_columns_list()
        ##Split data in train/test using stratify = take randomly the same sample number from each class.
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.x_full, self.y_full, test_size=test_size, 
                                                            random_state=seed, stratify=self.y_full)

        ######## Preprocessing job #######
        if normalize:
            numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
            categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

            preprocessor = ColumnTransformer(transformers=[
                    ('num', numeric_transformer, list_num_features),
                    ('cat', categorical_transformer, list_cat_features)])

            self.pipeline_build_not_fit = Pipeline(steps=[('preprocessor', preprocessor),('model',  self.model)])
        else:
            categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
            preprocessor = ColumnTransformer(transformers=[('cat', categorical_transformer, list_cat_features)])
            self.pipeline_build_not_fit = Pipeline(steps=[('preprocessor', preprocessor),('model',  self.model)])

        self.model_fited = self._builder_fit_and_eval(test_size,learning_curve_mod)
        return self.model_fited

    def model_explainer(self,pipeline,pipeline_is_normalized=False):
        list_cat_features, list_num_features = self.get_columns_list()
        fig = shap_explainer_cat_pipeline(self.x_full, list_cat_features, list_num_features, pipeline,
                                          pipeline_is_normalized=pipeline_is_normalized)
        return fig


