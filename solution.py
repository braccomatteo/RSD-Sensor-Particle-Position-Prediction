import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


df_dev=pd.read_csv('C:\\Users\\matte\\Desktop\\DSL\\Project\\development.csv',sep= ',', low_memory=False)
df_eval = pd.read_csv('C:\\Users\\matte\\Desktop\\DSL\\Project\\evaluation.csv',sep= ',', low_memory=False)
list_max_corr = []
df_dev_coord = df_dev[['x', 'y']]
df_dev_train = df_dev.drop(columns=['x', 'y'])

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score,mean_absolute_error, mean_squared_error

features_eliminated = []
for i in [0, 7, 12, 16, 17]:
    features_eliminated  = features_eliminated + [f"pmax[{i}]",f"negpmax[{i}]",f"area[{i}]",f"tmax[{i}]",f"rms[{i}]"]
for i in [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 13, 14]:
    features_eliminated  = features_eliminated + [f"area[{i}]",f"tmax[{i}]", f"rms[{i}]"]
for i in [15]:
    features_eliminated  = features_eliminated + [f"area[{i}]",f"tmax[{i}]",f"rms[{i}]", f"negpmax[{i}]"]
"""for i in [8]:
    features_eliminated  = features_eliminated + [f"area[{i}]", f"rms[{i}]"]
for i in [2]:
    features_eliminated  = features_eliminated + [f"area[{i}]", f"rms[{i}]", f"negpmax[{i}]"]"""
df_dev_drop_train = df_dev_train.drop(columns = features_eliminated)
"""for i in [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 13, 14, 15]:
    df_dev_drop_train[[f"pmax[{i}]"]] = np.log(df_dev_drop_train[[f"pmax[{i}]"]])"""
fetures_names = df_dev_drop_train.columns
X = df_dev_drop_train.values
X.shape
y = df_dev_coord.values
from sklearn.model_selection import train_test_split
"""X_train, X_valid, y_train, y_valid = train_test_split(X, y, shuffle=True, random_state=42, test_size = 0.15)"""

from sklearn.model_selection import ParameterGrid
params = {
    "max_depth": [48],
    "n_estimators": [750],
    "random_state": [42],
    "max_features": ["sqrt"],
    "criterion": ["friedman_mse"]
}
for config in ParameterGrid(params):
    print("\n" + "\n")
    regr_multi = MultiOutputRegressor(
    ExtraTreesRegressor(**config)
    )
    regr_multi.fit(X, y)
    """y_pred = regr_multi.predict(X)
    eucl_dist = np.sum(np.sum(((y_valid - y_pred) )*(2), axis=1)*(1/2), axis=0) * 1/(y_valid.size / 2)"""
    """print(config, "\t ---> " ,eucl_dist)"""
    """for i in [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 13, 14, 15]:
        df_eval[[f"pmax[{i}]"]] = np.log(df_eval[[f"pmax[{i}]"]])"""
    X_test = df_eval.drop(columns = features_eliminated + ['Id'])
    y_eva_pred = regr_multi.predict(X_test.values)
    y_eva_pred = pd.DataFrame(y_eva_pred, columns=['x', 'y'])
    y_eva_pred['index'] = range(len(y_eva_pred))
    y_eva_pred['xy'] = y_eva_pred.apply(lambda row: '|'.join(map(str, row[['x', 'y']])), axis=1)
    y_eva_pred = y_eva_pred.drop(['x', 'y', 'index'], axis=1)
    csv_name = "output_" + str(config.get("max_depth")) +"" + str(config.get("n_estimators")) +"" + str(config.get("max_features")) + str(config.get("criterion")) + ".csv"
    y_eva_pred.to_csv(csv_name, index_label="Id", header=['Predicted'])
    for el in sorted(zip(fetures_names, regr_multi.estimators_[0].feature_importances_), key=lambda x: x[1], reverse=True):
        print(el)
    print("\n")
    for el in sorted(zip(fetures_names, regr_multi.estimators_[1].feature_importances_), key=lambda x: x[1], reverse=True):
        print(el)