'''
    Problem Set 7: Question 2
    Joseph Farrell
    DS 5110 Intro to Data Management
    10/29/2023

    This file contains the solution to question 2
'''
import numpy as np
import matplotlib.pyplot as plt  
import pandas as pd  

from q1 import df
from sklearn.datasets import load_diabetes
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression

import numpy as np
import matplotlib.pyplot as plt  
import pandas as pd
import seaborn as sns

from q1 import df
from sklearn.datasets import load_diabetes
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression

############################################################################################
#################################### plot_sequential_additions #############################
############################################################################################
def plot_sequential_additions():
    ''' 
        Function: plot_sequential_additions
        Parameters: None
        Returns: None

        This function will sklearn SequentialFeatureSelector and LinearRegression to execute MLRs using
        forward feature selection. It loop over SequentialFeatureSelector increasing n (number of features) from
        1 to 9. It will keep track of each feature that is added to model as the number of features increases. 
    '''
    # load the data
    diabetes = load_diabetes()
    X, y = diabetes.data, diabetes.target
    feature_names = np.array(diabetes.feature_names)

    # instantiate model
    model = LinearRegression()
    
    # instantiate a feature added list to keep track of the order
    feature_additions = []

    # loop over the number of features included, 1 to 9
    for i in range(1, len(feature_names)):

        # execute the model and extract the features used
        sfs_forward = SequentialFeatureSelector(model, n_features_to_select = i).fit(X, y)
        features = feature_names[sfs_forward.get_support()]

        # update feature additions list
        for i in features:
            if i not in feature_additions:
                feature_additions.append(i)

    # update the order of the data for plotting
    feature_additions = np.array(feature_additions)
    df_2 = df[df['features'].isin(feature_additions)]
    df_2.set_index('features', inplace = True)
    df_2 = df_2.reindex(feature_additions)

    # generate plot
    fig, ax = plt.subplots(1, 2, figsize=(8, 5), sharey= True)
    fig.suptitle('Univariate Coefficient of Determiation \n versus \n Order of Inclusion in Sequential Feature Selection', 
                weight = 'bold', 
                fontsize = 16)
    fig.supxlabel('Features', 
                weight = 'bold')

    ax[0].bar(df_2.index, df_2['r_squared'])
    ax[0].set_title('Order of Inclusion: SFS', style = 'italic')
    ax[0].set_ylabel('Coefficient of Determination $R^2$', weight='bold')

    ax[1].bar(df['features'], df['r_squared'], color = 'firebrick')
    ax[1].set_title('Univariate $R^2$: Descending Order', style = 'italic')
    ax[1].set_ylabel(None)
    plt.tight_layout()

    # save fig
    save_path = ('figs/sequential_additions.png')
    plt.savefig(save_path, dpi = 300, bbox_inches = 'tight')
    plt.close()

############################################################################################
############################# plot_sequential_additions ####################################
############################################################################################
plot_sequential_additions()





