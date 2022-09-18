#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 11:13:41 2022

@author: alessandro
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import DecisionBoundaryDisplay

# Parameters
n_classes = 3
plot_colors = "ryb"
plot_step = 0.02

# Load data
iris = load_iris(as_frame=True)
iris_df = iris.data

# Split data
X = iris_df.filter(["sepal length (cm)", "petal width (cm)"])
y = iris.target

# Train
clf = DecisionTreeClassifier().fit(X, y)

# Plot the decision boundary
plt.gcf().set_size_inches(20, 10)
ax = plt.subplot(2, 3, 1)
plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
DecisionBoundaryDisplay.from_estimator(
    clf,
    X,
    cmap=plt.cm.RdYlBu,
    response_method="predict",
    ax=ax,
    xlabel=X.columns[0],
    ylabel=X.columns[1],
)

# Plot the training points
for i, color in zip(range(n_classes), plot_colors):

    idx = np.where(y == i)

    plt.scatter(
        X.loc[idx].iloc[:, 0],
        X.loc[idx].iloc[:, 1],
        c=color,
        label=iris.target_names[i],
        cmap=plt.cm.RdYlBu,
        edgecolor="black",
        s=50,
    )

plt.suptitle("Decision surface of decision trees trained on pairs of features")
plt.legend(loc="lower right", borderpad=0, handletextpad=0)
_ = plt.axis("tight")
