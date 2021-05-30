import shap
import eli5

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def shap_explainer(x_full,model_fited,show_dependencies=False):
    """ Return interpretation of model
    not realy love normalized data and will not support pipelines
    !! don't use it if your train was normalized=True !!

    params:
    x_full = dataframe used for create model !! 
    model_fited = model trained
    show_dependencies = show figure of features dependencies
    return figure of shap interpret
    """
    explainer = shap.TreeExplainer(model_fited)
    shap_values = explainer.shap_values(x_full)
    fig = shap.summary_plot(shap_values, x_full, title="Features Decisions Importances")
    if show_dependencies:
        try:
            shap.dependence_plot("rank(0)", shap_values[0], x_full, title="Features dependencies")
        except:
            shap.dependence_plot("rank(0)", shap_values, x_full, title="Features dependencies")
    return fig


def shap_explainer_cat_pipeline(data, list_cat_features, list_num_features, pipeline, pipeline_is_normalized=False):
    """params:
    data = dataframe as input ( sample of dataset or single input as validation prediction or in production)
    list_cat_features = array of strings contains header of categorial features of df type []
    list_num_features = array of strings contains header of numerical features of df, set only if use normalized type []
    pipeline = SkLearn pipeline returned by fit_and_eval func
    pipeline_is_normalized = False, by default set to true is fit_and_eval pipeline was normalize, type bool
    return figure of shap interpret"""

    if pipeline_is_normalized :
        feature_name = list(pipeline.named_steps['preprocessor'].transformers_[1][1].named_steps['onehot'].get_feature_names(list_cat_features))
    else :
        feature_name = list(pipeline.named_steps['preprocessor'].transformers_[0][1].named_steps['onehot'].get_feature_names(list_cat_features))
                    
    features_list = list(list_num_features)
    features_list.extend(feature_name)
    
    if len(data) >= 500:
        data = data.sample(100)
    input_data = pipeline[0].transform(data)
    inc_data = pd.DataFrame(input_data, columns=features_list)
    fig = shap_explainer(inc_data,pipeline[1])
    return fig


def eli5_explainer(list_cat_features,list_num_features,pipeline,pipeline_is_normalized=False):
    """ Return interpretation of model
    params:
    list_cat_features = array of strings contains header of categorial features of df type []
    list_num_features = array of strings contains header of numerical features of df, set only if use normalized type []
    pipeline = SkLearn pipeline returned by fit_and_eval func
    pipeline_is_normalized = False, by default set to true is fit_and_eval pipeline was normalize, type bool
    return figure of elie5 interpret
    """
    ## Model Explanation
    try:
        if pipeline_is_normalized:
            feature_name = list(pipeline.named_steps['preprocessor'].transformers_[1][1].named_steps['onehot'].get_feature_names(list_cat_features))
            features_list = list(list_num_features)
            features_list.extend(feature_name) 
            df_interpret = eli5.formatters.as_dataframe.explain_weights_df(pipeline.named_steps['model'], 
                    top=50, feature_names=features_list, feature_filter=lambda x: x != '<BIAS>')
            fig = plt.figure(figsize=(12,8))
            sns.barplot(x="weight", y="feature", data=df_interpret,palette="Blues_d")
            plt.title("Features Importance")
            return fig
        else:
            feature_name = list(pipeline.named_steps['preprocessor'].transformers_[0][1].named_steps['onehot'].get_feature_names(list_cat_features))
            df_interpret = eli5.formatters.as_dataframe.explain_weights_df(pipeline.named_steps['model'], 
                    top=50, feature_names=feature_name, feature_filter=lambda x: x != '<BIAS>')
            fig = plt.figure(figsize=(12,8))
            sns.barplot(x="weight", y="feature", data=df_interpret,palette="Blues_d")
            plt.title("Features Importance")
            return fig
    except Exception as e:
        print("-----------")
        print("No Features Importances for this model : ",e)
        print("-----------")

