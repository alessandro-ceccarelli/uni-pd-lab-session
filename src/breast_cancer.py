#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 11:45:47 2022

@author: alessandro
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import DecisionBoundaryDisplay


def filter_high_confidence_predictions(X: pd.DataFrame, threshold: float = 0.9):
    print(f"Treshold is set at {threshold}")
    return X.query(
        f"prediction_0 > {threshold} | prediction_1 > {threshold}"
    ).reset_index(drop=True)


def scatter_plot(
    X: pd.DataFrame,
    model,
    is_prediction_plot: bool = False,
    add_decision_boundary: bool = False,
):

    cm_bright = ListedColormap(["#FF0000", "#0000FF"])
    plt.gcf().set_size_inches(20, 10)
    ax = plt.subplot(2, 3, 1)
    plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

    if add_decision_boundary:

        DecisionBoundaryDisplay.from_estimator(
            model,
            X.filter(selected_features_l),
            axis=1,
            cmap=plt.cm.RdBu,
            response_method="predict_proba",
            alpha=0.9,
            eps=0.05,
            ax=ax,
            xlabel=X.columns[0],
            ylabel=X.columns[1],
        )

    # Plot the training points
    iterations = n_classes + 1 if is_prediction_plot else n_classes
    for i, color in zip(range(iterations), plot_colors):

        is_wrong_prediction_class = i == (iterations - 1)

        idx = (
            np.where(X.point_prediction == i)
            if is_prediction_plot
            else np.where(X.target == i)
        )

        plt.scatter(
            X.loc[idx].iloc[:, 0],
            X.loc[idx].iloc[:, 1],
            c=color,
            label="wrong prediction"
            if is_wrong_prediction_class & is_prediction_plot
            else breast_cancer.target_names[i],
            marker="*"
            if is_wrong_prediction_class
            & is_prediction_plot
            & (not add_decision_boundary)
            else "o",
            cmap=cm_bright,
            edgecolor="black",
            s=55 if not add_decision_boundary else 25,
        )

    plt.title("Predicted points" if is_prediction_plot else "Raw data points")
    plt.legend(loc="lower right", borderpad=0, handletextpad=0)
    plt.show()


# Parameters
n_classes = 2
plot_colors = "ryb"
plot_step = 0.02

# Load data
breast_cancer = load_breast_cancer(as_frame=True)
canc_df = breast_cancer.data

# Split data
selected_features_l = ["mean perimeter", "mean texture"]
X, y = canc_df.filter(selected_features_l), breast_cancer.target

# Train and predict
clf = DecisionTreeClassifier().fit(X, y)
# clf = RandomForestClassifier().fit(X, y)
# clf = KNeighborsClassifier().fit(X, y)

X[["prediction_0", "prediction_1"]] = clf.predict_proba(X)
X = X.assign(target=y, point_prediction=clf.predict(X.filter(selected_features_l)))

X["point_prediction"] = np.where(
    X.target != X.point_prediction, max(X.target) + 1, X.point_prediction
)

# Raw data plot
scatter_plot(X, model=clf, is_prediction_plot=False)

# Prediction plot
scatter_plot(X, model=clf, is_prediction_plot=True)

# High-confidence prediction plot
scatter_plot(
    X.pipe(filter_high_confidence_predictions), model=clf, is_prediction_plot=True
)

# Decision boundary plot
scatter_plot(
    X.pipe(filter_high_confidence_predictions),
    model=clf,
    is_prediction_plot=False,
    add_decision_boundary=True,
)

# Outlier detection
outlier_detector = GaussianMixture(n_components=3, covariance_type="full").fit(
    X.filter(selected_features_l)
)

X["outlier_score"] = outlier_detector.score_samples(X.filter(selected_features_l))
boundary = np.quantile(X["outlier_score"], 0.01)
X["Is anomaly?"] = np.where(X["outlier_score"] < boundary, "Anomaly", "Normal")

# Decision boundary plot
scatter_plot(
    X, model=outlier_detector, is_prediction_plot=False, add_decision_boundary=True
)

sns.scatterplot(
    data=X, x=selected_features_l[0], y=selected_features_l[1], hue="Is anomaly?"
)
