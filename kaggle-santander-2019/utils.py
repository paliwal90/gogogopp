import numpy as np 
import pandas as pd 
import os
import sys
import gc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns 
plt.style.use('ggplot')
import lightgbm as lgb
import xgboost as xgb
import time
import datetime
plt.style.use('seaborn')

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold,KFold, train_test_split
import warnings
from six.moves import urllib
from scipy.stats import norm, skew
#from chi import ChiSquare

from tqdm import tqdm_notebook as tqdm
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

import warnings
warnings.filterwarnings("ignore")

import logging

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s %(message)s')


class YeoJohnson:
    """
    Computing Yeo-Johnson transofrmation, which is an extension of Box-Cox transformation
    but can handle both positive and negative values.
    References:
    Weisberg, S. (2001). Yeo-Johnson Power Transformations.
    Department of Applied Statistics, University of Minnesota. Retrieved June, 1, 2003.
    https://www.stat.umn.edu/arc/yjpower.pdf
    Adapted from CRAN - Package VGAM
    """
    def fit(self, y, lmbda, derivative=0, epsilon=np.finfo(np.float).eps, inverse=False):
        """
        :param y: The variable to be transformed (numeric array).
        :param lmbda: The function's Lambda value (numeric value or array).
        :param derivative: The derivative with respect to lambda
        (non-negative integer; default: ordinary function evaluation).
        :param epsilon: The lambda's tolerance (positive value).
        :param inverse: The inverse transformation option (logical value).
        :return: The Yeo-Johnson transformation or its inverse, or its derivatives with respect to lambda, of y.
        """
        # Validate arguments
        self.__validate(y, lmbda, derivative, epsilon, inverse)

        # initialise
        y = np.array(y, dtype=float)
        result = y
        if not (isinstance(lmbda, list) or isinstance(lmbda, np.ndarray)):
            lmbda, y = np.broadcast_arrays(lmbda, y)
            lmbda = np.array(lmbda, dtype=float)
        l0 = np.abs(lmbda) > epsilon
        l2 = np.abs(lmbda - 2) > epsilon

        # Inverse
        with warnings.catch_warnings():  # suppress warnings
            warnings.simplefilter("ignore")
            if inverse is True:
                mask = np.where(((y >= 0) & l0) == True)
                result[mask] = np.power(np.multiply(y[mask], lmbda[mask]) + 1, 1 / lmbda[mask]) - 1

                mask = np.where(((y >= 0) & ~l0) == True)
                result[mask] = np.expm1(y[mask])

                mask = np.where(((y < 0) & l2) == True)
                result[mask] = 1 - np.power(np.multiply(-(2 - lmbda[mask]), y[mask]) + 1, 1 / (2 - lmbda[mask]))

                mask = np.where(((y < 0) & ~l2) == True)
                result[mask] = -np.expm1(-y[mask])

            # Derivative
            else:
                if derivative == 0:
                    mask = np.where(((y >= 0) & l0) == True)
                    result[mask] = np.divide(np.power(y[mask] + 1, lmbda[mask]) - 1, lmbda[mask])

                    mask = np.where(((y >= 0) & ~l0) == True)
                    result[mask] = np.log1p(y[mask])

                    mask = np.where(((y < 0) & l2) == True)
                    result[mask] = np.divide(-(np.power(-y[mask] + 1, 2 - lmbda[mask]) - 1), 2 - lmbda[mask])

                    mask = np.where(((y < 0) & ~l2) == True)
                    result[mask] = -np.log1p(-y[mask])

                # Not Derivative
                else:
                    p = self.fit(y, lmbda, derivative=derivative - 1, epsilon=epsilon, inverse=inverse)

                    mask = np.where(((y >= 0) & l0) == True)
                    result[mask] = np.divide(np.multiply(np.power(y[mask] + 1, lmbda[mask]),
                                                         np.power(np.log1p(y[mask]), derivative)) -
                                             np.multiply(derivative, p[mask]), lmbda[mask])

                    mask = np.where(((y >= 0) & ~l0) == True)
                    result[mask] = np.divide(np.power(np.log1p(y[mask]), derivative + 1), derivative + 1)

                    mask = np.where(((y < 0) & l2) == True)
                    result[mask] = np.divide(-(np.multiply(np.power(-y[mask] + 1, 2 - lmbda[mask]),
                                                                    np.power(-np.log1p(-y[mask]), derivative)) -
                                                        np.multiply(derivative, p[mask])), 2 - lmbda[mask])

                    mask = np.where(((y < 0) & ~l2) == True)
                    result[mask] = np.divide(np.power(-np.log1p(-y[mask]), derivative + 1), derivative + 1)

        return result

    @staticmethod
    def __validate(y, lmbda, derivative, epsilon, inverse):
        try:
            if not isinstance(y, (list, np.ndarray, pd.Series)):
                raise Exception("Argument 'y' must be a list!")
            if not isinstance(lmbda, (int, float, np.int, np.float)):
                if not isinstance(lmbda, (list, np.ndarray, pd.Series)) or len(lmbda) != len(y):
                    raise Exception("Argument 'lmbda' must be a number "
                                    "or a list, which its length matches 'y' argument!")
            if not isinstance(derivative, (int, float, np.int, np.float)) or derivative < 0:
                raise Exception("Argument 'derivative' must be a non-negative integer!")
            if not isinstance(epsilon, (int, float, np.int, np.float)) or epsilon <= 0:
                raise Exception("Argument 'epsilon' must be a positive number!")
            if not isinstance(inverse, bool):
                raise Exception("Argument 'inverse' must be boolean!")
            if inverse is True and derivative != 0:
                raise Exception("Argument 'derivative' must be zero "
                                "when argument 'inverse' is 'True'!")
        except ():
            sys.exit()
            
            
def inverse_yeo_johnson(y, yj_lambda=0.7):
    yj_trans = YeoJohnson()
    return yj_trans.fit(y, lmbda=yj_lambda, inverse=True)

            
            
