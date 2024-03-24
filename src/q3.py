'''
    Problem Set 7: Question 3
    Joseph Farrell
    DS 5110 Intro to Data Management
    10/29/2023

    This file contains the solution to question 3
'''
import numpy as np
import matplotlib.pyplot as plt  
import pandas as pd  

from q1 import df
from sklearn.datasets import load_diabetes


import numpy as np
import matplotlib.pyplot as plt  
import pandas as pd
import seaborn as sns

############################################################################################
############################# explore_feature_addition_order ###############################
############################################################################################
def explore_feature_addition_order():
    ''' 
        Function: explore_feature_addition_order
        Parameters: None
        Returns: None

        This function will generate a heat map of the feature correlation matrix to help explain
        the order the features were added using SequentialFeatureSelection
    '''
    # load data
    diabetes = load_diabetes()
    X, y = diabetes.data, diabetes.target
    feature_names = np.array(diabetes.feature_names)

    # Convert arrays to DataFrames
    df1 = pd.DataFrame(X, columns = feature_names)
    df2 = pd.DataFrame(y, columns=['target'])

    # Concatenate DataFrames along columns (axis=1)
    df = pd.concat([df1, df2], axis=1)

    corr_matrix = df.corr()
    plt.figure(figsize = (8,7))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Heatmap of Correlation Matrix', weight = 'bold', fontsize = 18)
    plt.xlabel('Features & Target', weight = 'bold')
    plt.ylabel('Features & Target', weight = 'bold')
    plt.tight_layout()

    # save fig
    save_path = ('figs/feature_correlation_heatmap.png')
    plt.savefig(save_path, dpi = 300, bbox_inches = 'tight')
    plt.close()

############################################################################################
###################### invoke explore_feature_addition_order ###############################
############################################################################################
explore_feature_addition_order()