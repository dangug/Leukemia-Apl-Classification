import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def hist_plot(df, col, **kwargs):
    """Plot the histogram on numerical column col
    """
    return sns.histplot(x=col, data=df,  **kwargs)

def count_plot(df, col, **kwargs):
    """Plot the distibution of each unique value on categorical column c
    """
    # plt.figure(figsize=(15,5))
    return sns.countplot(x=col, data=df, **kwargs)

    # list_num_subject = df[col].unique()
    # list_num_subject = np.sort(list_num_subject, axis=None)

    # panel_list = []
    # count_tot = df.shape[0]
    # for sub in list_num_subject:
    #     panel_list.append(df[df[col].isin([sub])]) 
    #     this_panel = panel_list[sub]
    #     count_class = this_panel[col].count()
    #     pc_class = count_class*100/count_tot
    #     print(f"Nombres de sujet classe {sub} : ", round(count_class,2), f" = {round(pc_class,2)} %")

    # return panel_list


numeric_types = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

class SimpleExplorer():

    def __init__(self, df, target):
        self.data = df
        self.target = target
        self.cat_columns = self._infer_cat_columns()
        self.num_columns = self._infer_num_columns()
        

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

    #TODO add function to change num column to catagorical

    def report(self):
        """Full data report
        """
        pass

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


    #TODO seaborn pair plot on numerical

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




