import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

## GLOBALS ##
numeric_types = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

## Tools ##
def hist_plot(df, col, **kwargs):
    """Plot the histogram on numerical column col
    """
    return sns.histplot(x=col, data=df,  **kwargs)

def count_plot(df, col, **kwargs):
    """Plot the distibution of each unique value on categorical column 
    """
    # plt.figure(figsize=(15,5))
    return sns.countplot(x=col, data=df, **kwargs)

## Main ##
class DataExplore():
    """Tools for easly explore a pandas dataframe"""

    def __init__(self, df, target='target'):
        self.data = df
        self.target = target
        self.cat_columns = self._infer_cat_columns()
        self.num_columns = self._infer_num_columns()
        self.dum_columns = None
        self.data_dummies = None

    def _infer_cat_columns(self):
        """Infer the list of categorical columns from the data types of each column
        Return: A list of column names
        """
        cols_cat = list(self.data.select_dtypes(exclude=numeric_types).columns)
        # # One hot columns
        # for col in [col for col in self.data.columns if col not in cols_cat]:
        #     if set(self.data.unique()).issubset({0, 1, np.nan}):
        #         cols_cat += [col]
        return cols_cat

    def _infer_num_columns(self):
        """Infer the list of numerical columns from the data types of each column
        Return: A list of column names
        """
        cols_num = list(self.data.select_dtypes(numeric_types).columns)
        return cols_num
    
    def _infer_dummies_columns(self):
        """Infer the list of dummies columns from the data types of each column
        Return: A list of column names
        """
        self.data_dummies = pd.get_dummies(self.data[self.cat_columns], drop_first=True)
        self.dum_columns = self.data_dummies.columns
        return self.data_dummies, self.dum_columns

    def count_na(self):
        sum_null = self.data.isnull().sum()
        total = self.data.isnull().count()
        percent_nullvalues = 100* sum_null / total 
        df_null = pd.DataFrame()
        df_null['Total'] = total
        df_null['Null_Count'] = sum_null
        df_null['Percent'] = round(percent_nullvalues,2)
        df_null = df_null.sort_values(by='Null_Count',ascending = False)
        df_null = df_null[df_null.Null_Count > 0]
        return(df_null)

    def plot_na(self):
        df_null = self.data.isnull().sum(axis=1)
        df_null = df_null[df_null!=0]
        return df_null.hist()

    def plot_dist(self, figsize=(15,5)):
        n_plots = len(self.data.columns)
        grid_n_cols = 4
        grid_n_rows = n_plots//grid_n_cols +1

        fig, axes = plt.subplots(nrows=grid_n_rows, ncols=grid_n_cols,
                        figsize=figsize, constrained_layout=True)

        for i, col in enumerate(self.data.columns):
            ax_row = i//grid_n_cols
            ax_col = i % grid_n_cols

            if col in self.cat_columns:
                count_plot(self.data, col, ax=axes[ax_row][ax_col])
            else:
                hist_plot(self.data, col, ax=axes[ax_row][ax_col])
        return fig

    def pair_plot(self ,cols=None):
        #compare num columns
        
        if self.target in self.cat_columns:
            if cols is None:
                cols = self.num_columns
            sns.pairplot(self.data, vars=cols, hue=self.target)
        else:
            if cols is None:
                cols = self.num_columns+[self.target]
            sns.pairplot(self.data, vars=cols)

    def distribution_binaries(self, df=None,subject=None):
        """ show distribution details & vis for subject and return both df if needed for other operation """

        if df is None:
            df = self.data
        if subject is None:
            subject = self.target

        plt.figure(figsize=(15,5))
        sns.countplot(x=subject, data=df)
        plt.show()

        ## je split et isole le df en 2, et les sotcks dans des variables positif/negatif
        panel_positif = df[df[subject].isin([1])]
        panel_negatif = df[df[subject].isin([0])]

        count_pos = panel_positif[subject].count()
        count_neg = panel_negatif[subject].count()
        count_tot = df.shape[0]

        pc_pos = count_pos*100/count_tot
        pc_neg = count_neg*100/count_tot

        print("Nombres de sujet positifs : ", round(count_pos,2), f" = {round(pc_pos,2)} %")
        print("Nombres de sujet negatifs : ", round(count_neg,2), f" = {round(pc_neg,2)} %")
        
        return panel_positif, panel_negatif

    def distribution_Multi(self, df=None,subject=None):
        """Pass your DataFrame, the key of subject (target by default), 
        return : list of panel of each classes """

        if df is None:
            df = self.data
        else:
            df = df
            df[self.target] = self.data[self.target]
        if subject is None:
            subject = self.target

        plt.figure(figsize=(15,5))
        sns.countplot(x=subject, data=df)
        plt.show()

        list_num_subject = df[self.target].unique()
        list_num_subject = np.sort(list_num_subject, axis=None)

        panel_list = []
        count_tot = df.shape[0]
        for sub in list_num_subject:
            panel_list.append(df[df[self.target].isin([sub])]) 
            this_panel = panel_list[sub]
            count_class = this_panel[self.target].count()
            pc_class = count_class*100/count_tot
            print(f"Nombres de sujet classe {sub} : ", round(count_class,2), f" = {round(pc_class,2)} %")
        return panel_list

    def all_num_features_distribution(self, show_outliers=True, swarmplot_mod=False):
        """show_outliers = True by default = show outliers, set to false to plot with outlier removed
        swarmplot_mod : Set to true add swarmPlot on plot, not recommanded for more than 1000 rows;
        return : show distribution details with plot """
        df = self.data 
        df[self.target] = self.data[self.target]
        list_num_cols_name = self.num_columns
        subject = self.target

        plt.rcParams.update({'figure.max_open_warning': 0}) ## Care only for jupyter ... 
        
        for col in list_num_cols_name:
            if col != subject:
                plt.figure(figsize=(15,5))
                sns.boxplot(x=subject, y=col, data=df, showfliers=show_outliers) ## showfliers=false -> remove outliers  from plot 
                if swarmplot_mod:
                    sns.swarmplot(x=subject, y=col, data=df, color=".25")
                ## plt.legend(['0 : Negatif','1 : Positif']) ## problem for multi classes
                plt.title(f"Distribution by target for {col} ")
                plt.show()
    
    def all_dum_features_distribution(self, swarmplot_mod=False):
        """swarmplot_mod : Set to true add swarmPlot on plot, not recommanded for more than 1000 rows;
        return : show distribution details with plot """
        df = self.data_dummies
        df[self.target] = self.data[self.target]
        list_dum_cols_name = self.dum_columns
        subject = self.target

        plt.rcParams.update({'figure.max_open_warning': 0}) ## Care only for jupyter ... 
        
        for col in list_dum_cols_name:
            if col != subject:
                plt.figure(figsize=(15,5))
                sns.countplot(x=col, hue=subject, data=df)
                if swarmplot_mod:
                    sns.swarmplot(x=subject, y=col, data=df, color=".25")
                ## plt.legend(['0 : Negatif','1 : Positif']) ## problem for multi classes
                plt.title(f"Distribution by target for {col} ")
                plt.show()

    def target_encoder(self):
        """  Transform your target categorial columns on number
        strore result df['target'] = y_full
        """
        lbl_enc = LabelEncoder()
        y_full = self.data[self.target]
        y_full = lbl_enc.fit_transform(y_full)
        self.data[self.target] = y_full
        return y_full


