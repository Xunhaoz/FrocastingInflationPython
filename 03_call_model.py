import copy

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from tqdm import trange
from functions.utils import roll_prod

n_windows = 180

from sklearn.ensemble import ExtraTreesRegressor, BaggingRegressor, RandomForestRegressor, \
    HistGradientBoostingRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.svm import SVR, NuSVR, LinearSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import RANSACRegressor, LinearRegression, HuberRegressor, SGDRegressor, \
    PassiveAggressiveRegressor, BayesianRidge, ARDRegression, TweedieRegressor, ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.cross_decomposition import PLSRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


def prep_data():
    data = pd.read_csv("data/data.csv", index_col=0)
    data['dummy'] = 0
    data.loc['2008-11-01', 'dummy'] = 1
    pca = PCA(n_components=4).fit_transform(data)
    X = np.concatenate([data.values, pca], axis=1)
    y = data[['CPIAUCSL']].values[3:]
    X_lag = []
    for i in range(X.shape[0] - 3):
        X_lag.append(X[i:i + 4].flatten())
    return np.array(X_lag), y


models = [
    LinearRegression(),
    ExtraTreesRegressor(n_jobs=-1),
    BaggingRegressor(n_jobs=-1),
    RandomForestRegressor(n_jobs=-1),
    HistGradientBoostingRegressor(),
    GradientBoostingRegressor(),
    AdaBoostRegressor(),
    DecisionTreeRegressor(),
    ExtraTreeRegressor(),
    KNeighborsRegressor(n_jobs=-1),
    RANSACRegressor(),
    HuberRegressor(),
    SGDRegressor(),
    PassiveAggressiveRegressor(),
    BayesianRidge(),
    ElasticNet(),
    KernelRidge(),
    PLSRegression(),
    GaussianProcessRegressor(),
    XGBRegressor(),
    LGBMRegressor(n_jobs=-1, verbose=False),
    CatBoostRegressor(verbose=-1),
    MLPRegressor(max_iter=1000),
    SVR(max_iter=1000),
    NuSVR(max_iter=1000),
    LinearSVR(max_iter=1000),
]

for model in models:
    try:
        y_predict_list = []
        X_lag, y = prep_data()
        for w in trange(1, 13):
            X_train, X_test, y_train, y_test = X_lag[:-w][:-180], X_lag[:-w][-180:], y[w:][:-180], y[w:][-180:]
            model_copy = copy.copy(model)
            model_copy.fit(X_train, y_train)
            y_predict_list.append(model_copy.predict(X_test))

        predict_arr = np.array(y_predict_list).T

        for period in [3, 6, 12]:
            acc_predict = np.vstack([
                np.diag(predict_arr[w:w + period, :period])
                for w in range(predict_arr.shape[0] - period + 1)
            ])
            acc_temp = np.full(n_windows, np.nan)
            acc_temp[period - 1:] = np.cumprod(acc_predict + 1, axis=1)[:, -1] - 1
            predict_arr = np.concatenate([predict_arr, acc_temp.reshape((-1, 1))], axis=1)
        rw = pd.read_csv("forecasts/rw.csv", index_col=0)
        predict_result = pd.DataFrame(predict_arr, index=rw.index, columns=rw.columns)
        predict_result.to_csv(f"forecasts/{model.__class__.__name__}.csv")
    except Exception as e:
        continue