
# Test file, can be ignored.
from os import listdir

print("done")
import pandas as pd
from time import perf_counter

pd.options.display.max_columns = 6

from mapie.regression import MapieRegressor
from mapie.classification import MapieClassifier
from mapie.metrics import regression_coverage_score

from sklearn import model_selection
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, RepeatedKFold
from sklearn.ensemble import (GradientBoostingClassifier, HistGradientBoostingClassifier)
from sklearn.ensemble import (GradientBoostingRegressor, HistGradientBoostingRegressor)
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.experimental import enable_iterative_imputer
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeClassifier
from matplotlib.colors import ListedColormap

from sklearn.metrics import roc_auc_score
from sklearn.metrics import RocCurveDisplay

from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer

import numpy as np
from numpy import mean
from numpy import absolute
from numpy import sqrt

import xgboost as xgb
from xgboost import XGBRegressor
from xgboost import XGBClassifier

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import scale
from sklearn.metrics import mean_squared_error
# import the diamonds dataset
df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/diamonds.csv")

# Create a train, calibration and test set
X = df.drop(["price"], axis=1)
# drop categorical variables for brevity
X = X.drop(["cut", "color", "clarity"], axis=1)
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_cal, X_test, y_cal, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)