'''
    Problem Set 7: Question 4
    Joseph Farrell
    DS 5110 Intro to Data Management
    10/29/2023

    This file contains the solution to question 4
'''
import numpy as np
from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt


from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

############################################################################################
################################### execute_pcr ############################################
############################################################################################
def execute_pcr() -> None:
    ''' 
        Function: execute_pcr
        Parameters: None
        Returns: None

        This function will execute a PCR and plot the results with respect to the 
        number of componets used in the model.
    '''
    # import data
    diabetes = load_diabetes()
    X, y = diabetes.data, diabetes.target

    # instantaite pipeline
    pcr = make_pipeline(PCA(), LinearRegression())

    # set the range for hyperparameter, alpha
    n_components = np.arange(1, X.shape[1])

    # set the number of folds
    n_folds = 5

    # instanitiate cross validation
    clf = GridSearchCV(pcr, {'pca__n_components': n_components}, 
                    cv = n_folds,
                    refit = True, 
                    return_train_score = True)
    # fit model
    clf.fit(X, y)

    # extract results
    test_scores = clf.cv_results_["mean_test_score"]
    test_scores_std = clf.cv_results_["std_test_score"]
    train_scores = clf.cv_results_["mean_train_score"]
    train_scores_std = clf.cv_results_["std_train_score"]

    # compute standard errors
    test_std_error = test_scores_std / np.sqrt(n_folds)
    train_std_error = train_scores_std / np.sqrt(n_folds)

    # generate figure
    plt.figure().set_size_inches(8, 6)

    # scores
    plt.plot(n_components, test_scores, label = 'Test Scores')
    plt.plot(n_components, train_scores, 'darkorange', label = 'Train Scores')

    # standard error
    plt.fill_between(n_components, test_scores + test_std_error, test_scores - test_std_error, 
                    alpha = 0.2)
    plt.fill_between(n_components, train_scores + train_std_error, train_scores - train_std_error, 
                    alpha = 0.2, color = 'darkorange')

    # max score
    plt.axhline(np.max(test_scores), linestyle = "--", color= 'red', label = 'Max Score')

    # labels
    plt.title(f'MLR Training and Test Scores: \n With Respect to Number of Principal Components', 
            weight = 'bold', 
            style = 'italic', 
            fontsize = 14)
    plt.ylabel(r'CV Score $\pm SE = \frac{\sigma}{\sqrt{n}}$', fontsize = 14, weight = 'bold')
    plt.xlabel('Alpha ~ Number of Principal Components', weight = 'bold')
    plt.annotate(f'Max Score = {np.max(test_scores):.3f}', 
                xy = (1.1, .47), 
                xytext=(1.1, .47), 
                weight = 'bold', 
                color = 'red')
    plt.xlim([n_components[0], n_components[-1]])
    plt.legend(loc = 4, frameon = True)
    plt.grid();

    # save fig
    save_path = ('figs/pcr_results.png')
    plt.savefig(save_path, dpi = 300, bbox_inches = 'tight')
    plt.close()

############################################################################################
################################# invoke execute_pcr########################################
############################################################################################
execute_pcr()