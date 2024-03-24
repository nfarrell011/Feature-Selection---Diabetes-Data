'''
    Problem Set 7: Question 5
    Joseph Farrell
    DS 5110 Intro to Data Management
    10/29/2023

    This file contains the solution to question 5
'''
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model

############################################################################################
##################################### execute_lasso ########################################
############################################################################################
def execute_lasso():
    '''  
        Function: execute_lasso
        Parameters: None
        Returns: None

        This function will execute a linear regression using a lasso penalty term and
        plot path of feature inclusion
    '''
    diabetes = datasets.load_diabetes()
    X, y = datasets.load_diabetes(return_X_y=True)
    feature_names = np.array(diabetes.feature_names)

    print("Computing regularization path using the LARS ...")
    _, _, coefs = linear_model.lars_path(X, y, method = "lasso", verbose = True)

    xx = np.sum(np.abs(coefs.T), axis=1)
    xx /= xx[-1]
    plt.figure(figsize = (8,5))
    plt.plot(xx, coefs.T, label = 'Components')
    ymin, ymax = plt.ylim()
    plt.vlines(xx, ymin, ymax, linestyle = "dashed")
    plt.xlabel("|coef| / max|coef|")
    plt.ylabel("Coefficients")
    plt.title("LASSO Path")
    plt.legend(labels = feature_names, ncols = 2, title = 'Features', shadow = True)
    plt.axis("tight")
    
    # save fig
    save_path = ('figs/lasso_results.png')
    plt.savefig(save_path, dpi = 300, bbox_inches = 'tight')
    plt.close()

############################################################################################
################################# invoke execute_lasso #####################################
############################################################################################
execute_lasso()