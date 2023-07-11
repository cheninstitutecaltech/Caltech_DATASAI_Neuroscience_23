
from sklearn.base import BaseEstimator, TransformerMixin


class standardize_x_cols(BaseEstimator, TransformerMixin):
    '''Class to standardize specified columns in the input matrix X. This class
    is intended to work in an analog fashion to sklearn's StandardScaler. Inherit
    methods from BaseEstimator and TransformerMixin to be able to use objects of
    this class in sklearn pipelines as a transformer. To make this happen the class
    requires a fit and a transform method.

    Parameters
    ---------
    column_idx: array, vector of column indices of the design matrix to be standardized
    ------------------------------------------------------------------------------------
    '''

    def __init__(self, column_idx = None):
        import numpy as np
        self.scale_ = None
        self.mean_ = None
        self.var_ = None

        if column_idx is None:
            column_idx = np.arange(834, 1258) #These are the dlc, video and video motion energy regressors for the miniscope data
        self.column_idx = column_idx

    def fit(self, X, y = None):
        '''Find mean, variance and std. '''
        import numpy as np
        x_array = np.array(X) #To deal with pandas data frames
        self.scale_ = np.ones(x_array.shape[1])  #Create vector of ones with length of feature number. One because this preseves the scale of the unscaled columns
        self.scale_[self.column_idx] = np.std(x_array[:, self.column_idx],axis=0)

        self.mean_ = np.zeros(x_array.shape[1])  #Create vector of zeros with length feature number. Mean zero will ensure that there is no recentering for unscaled variables
        self.mean_[self.column_idx] = np.mean(x_array[:, self.column_idx],axis=0)

        self.var_ = np.ones(x_array.shape[1])  #Create vector with length of feature number
        self.var_[self.column_idx] = np.var(x_array[:, self.column_idx],axis=0)
        return self

    def transform(self, X, y = None):
        '''Apply the found mean and std'''
        import numpy as np
        x_array = np.array(X) #To deal with pandas data frames
        x_array = (x_array - self.mean_) / self.scale_

        return x_array
