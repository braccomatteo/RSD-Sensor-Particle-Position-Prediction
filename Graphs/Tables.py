import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import os
os.listdir('/kaggle/input')

df_dev=pd.read_csv('/kaggle/input/january-project/development.csv',sep= ',', low_memory=False)
df_eval = pd.read_csv('/kaggle/input/january-project/evaluation.csv',sep= ',', low_memory=False)
list_max_corr = []
df_dev_coord = df_dev[['x', 'y']]
df_dev_train = df_dev.drop(columns=['x', 'y'])

features_eliminated = []
for i in [0, 7, 12, 16, 17]:
    features_eliminated  = features_eliminated + [f"pmax[{i}]",f"negpmax[{i}]",f"area[{i}]",f"tmax[{i}]",f"rms[{i}]"]
for i in [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 13, 14]:
    features_eliminated  = features_eliminated + [f"area[{i}]",f"tmax[{i}]", f"rms[{i}]"]
for i in [15]:
    features_eliminated  = features_eliminated + [f"area[{i}]",f"tmax[{i}]",f"rms[{i}]", f"negpmax[{i}]"]
df_dev_drop_train = df_dev_train.drop(columns = features_eliminated)

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, BaggingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score,mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
X = df_dev_drop_train.values
X.shape
y = df_dev_coord.values
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, shuffle=True, random_state=42, test_size = 0.20)

regr_lin = MultiOutputRegressor(
    LinearRegression( fit_intercept=True)
)

regr_lin.fit(X_train, y_train)
y_pred = regr_lin.predict(X_valid)
r2_adjusted = 1 - (1-regr_lin.score(X_valid, y_valid))*(len(y_valid)-1)/(len(y_valid)-X_valid.shape[1]-1)
r2_s = r2_score(y_valid, y_pred)
mae = mean_absolute_error(y_valid, y_pred)
mse = mean_squared_error(y_valid, y_pred)
print(r2_adjusted, r2_s, mae, mse)

regr_dt = MultiOutputRegressor(
    DecisionTreeRegressor()
)

regr_dt.fit(X_train, y_train)
y_pred = regr_dt.predict(X_valid)
r2_adjusted = 1 - (1-regr_dt.score(X_valid, y_valid))*(len(y_valid)-1)/(len(y_valid)-X_valid.shape[1]-1)
r2_s = r2_score(y_valid, y_pred)
mae = mean_absolute_error(y_valid, y_pred)
mse = mean_squared_error(y_valid, y_pred)
print(r2_adjusted, r2_s, mae, mse)

regr_rf = MultiOutputRegressor(
    RandomForestRegressor()
)

regr_rf.fit(X_train, y_train)
y_pred = regr_rf.predict(X_valid)
r2_adjusted = 1 - (1-regr_rf.score(X_valid, y_valid))*(len(y_valid)-1)/(len(y_valid)-X_valid.shape[1]-1)
r2_s = r2_score(y_valid, y_pred)
mae = mean_absolute_error(y_valid, y_pred)
mse = mean_squared_error(y_valid, y_pred)
print(r2_adjusted, r2_s, mae, mse)

regr_et = MultiOutputRegressor(
    ExtraTreesRegressor()
)

regr_et.fit(X_train, y_train)
y_pred = regr_et.predict(X_valid)
r2_adjusted = 1 - (1-regr_et.score(X_valid, y_valid))*(len(y_valid)-1)/(len(y_valid)-X_valid.shape[1]-1)
r2_s = r2_score(y_valid, y_pred)
mae = mean_absolute_error(y_valid, y_pred)
mse = mean_squared_error(y_valid, y_pred)
print(r2_adjusted, r2_s, mae, mse)

regr_b = MultiOutputRegressor(
    BaggingRegressor()
)

regr_b.fit(X_train, y_train)
y_pred = regr_b.predict(X_valid)
r2_adjusted = 1 - (1-regr_b.score(X_valid, y_valid))*(len(y_valid)-1)/(len(y_valid)-X_valid.shape[1]-1)
r2_s = r2_score(y_valid, y_pred)
mae = mean_absolute_error(y_valid, y_pred)
mse = mean_squared_error(y_valid, y_pred)
print(r2_adjusted, r2_s, mae, mse)