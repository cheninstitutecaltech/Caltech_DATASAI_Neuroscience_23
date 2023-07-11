import numpy as np
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score

from .utils import standardize_x_cols

def find_best_alphas(X,Y, alpha_test_range, cv=None, scoring='r2'):
    """Use SKLearn's RidgeCV to return the best alpha values from a specified alpha_test_range

    Parameters
    ----------
    X : ndarray
        input array of predictor variables
    Y : ndarray
        array of variables to be predicted with ridge regression
    alpha_test_range : ndarray
        an array of alpha values to be tested with RidgeCV
    cv : _type_, optional
        _description_, by default None
    scoring : str, optional
        _description_, by default 'r2'

    Returns
    -------
    ndarray
        an array of alpha values.
    """
    ridge_grid_search = RidgeCV(alphas=alpha_test_range,
                                fit_intercept=False,
                                alpha_per_target=True,
                                cv=cv,
                                scoring=scoring,
                                store_cv_values=True)
    ridge_grid_search.fit(X,Y)
    best_alphas = ridge_grid_search.alpha_
    return best_alphas

    
def cross_val_ridge(X, Y, alphas, n_folds):
    """Compute the cross-validated r_squared for a set of alpha values

    Parameters
    ----------
    X : ndarray
        input array of predictor variables
    Y : ndarray
        array of variables to be predicted with ridge regression
    alphas : ndarray
        list of alphas for each target
    n_folds : int
        number of folds for cross-validation

    Returns
    -------
    [ndarray, ndarray]
        a list containing the r_squared values at the first index and the beta values at the second
    """
    kf = KFold(n_splits=n_folds, shuffle=True) 
    custom_scaler = standardize_x_cols(column_idx = np.arange(832, 1258)) 
    ridge_model = Ridge(alpha=alphas, fit_intercept=False)
    pipe = Pipeline(steps=[('scaler', custom_scaler),
                           ('ridge', ridge_model)])

    betas = []
    r_squared = []
    for train_index, test_index in kf.split(X, Y): #iterate over cv folds
        fits = pipe.fit(X[train_index,:], Y[train_index,:])
        predictions = pipe.predict(X[test_index,:])

        r_squared.append(r2_score(Y[test_index,:], predictions, multioutput='raw_values')) #score test performance
        betas.append(pipe[-1].coef_)

    betas = np.squeeze(betas)
    r_squared = np.squeeze(r_squared)

    return r_squared, betas