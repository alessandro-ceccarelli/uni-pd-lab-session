#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  3 10:00:14 2022

@author: alessandro
"""
import pandas as pd
from matplotlib import pyplot
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_boston
from xgboost import XGBRegressor, plot_importance
from sklearn.metrics import mean_absolute_percentage_error as mape


boston = load_boston()

X = pd.DataFrame(boston.data, columns = boston.feature_names)
y = boston.target

# fit model no training data
model = XGBRegressor()
model.fit(X[:400], y[:400])
  
# display error
print(f"Mean MAPE: {mape(y, model.predict(X))}")

importance_df = pd.DataFrame(model.feature_importances_, boston.feature_names).sort_values(by=0, ascending=False)

plt.figure(figsize=(15,10))
plt.scatter(model.predict(X), y)
plt.show()

plt.figure(figsize=(15,10))
plt.barh(importance_df.index, importance_df[0])
plt.show()

print(boston["DESCR"])