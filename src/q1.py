'''
    Problem Set 7: Question 1
    Joseph Farrell
    DS 5110 Intro to Data Management
    10/29/2023

    This file contains the solution to question 1
'''
import numpy as np
import matplotlib.pyplot as plt  
import pandas as pd  
from sklearn.datasets import load_diabetes

############################################################################################
#################################### rank_uni_r_squared ####################################
############################################################################################
def rank_uni_r_squared() -> None:
    ''' 
        Function: rank_uni_r_squared
        Parameters: None
        Returns: None

        This function will read in the diabetes data. Compute the correlation between each of the 
        features and the target and produce barplot displaying the results.
    '''
    # read in diabetes data and extract features and target
    diabetes = load_diabetes()
    X, y = diabetes.data, diabetes.target

    # compute correlation coefficient between each feature and the target
    feature_target_r_2 = np.square(np.corrcoef(X, y, rowvar = False)[-1, :-1])

    # extract feature names
    feature_names = np.array(diabetes.feature_names)

    # sort values
    sorted_indices = np.argsort(feature_target_r_2)[::-1]
    feature_target_r_2 = feature_target_r_2[sorted_indices]
    feature_names = feature_names[sorted_indices]

    df = pd.DataFrame({'features': feature_names, 'r_squared': feature_target_r_2})

    # generate plot
    plt.bar(df.features, df.r_squared, color = 'firebrick')
    plt.title('Univarite Coefficient of Determination: \n Target ~ Individual Features', 
            weight = 'bold', 
            style = 'italic', 
            fontsize = 14)
    plt.ylabel('Coefficient of Determination $R^2$', weight = 'bold')
    plt.xlabel('Features', weight = 'bold')

    # save figure
    save_path = ('figs/univariate_r_squared.png')
    plt.savefig(save_path, dpi = 300, bbox_inches = 'tight')
 
    plt.close();
 
    return df

############################################################################################
############################ Invoke rank_uni_r_squared ####################################
############################################################################################

df = rank_uni_r_squared()

