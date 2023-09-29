import pandas as pd

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import seaborn as sn

from fields import *
from experiments import prepare_per_breast_df

def show_correlation_matrix(df):
    correlations = df[INPUT_FEATURE_COLS].corr()
    sn.heatmap(correlations, annot=True)
    plt.title("Pearson correlation coefficients between input variables")
    plt.show()


def principal_components(df):

    scaler = StandardScaler()
    pca = PCA()

    scaled_features = scaler.fit_transform(df)
    pricipal_components = pca.fit_transform(scaled_features)
    print(f"explained variance: {pca.explained_variance_}")
    print(f"explained variance ratio: {pca.explained_variance_ratio_}")

    # Plot initialisation
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(principal_components[:, 0], principal_components[:, 1], principal_components[:, 2], cmap="Set2_r", s=60)
    # make simple, bare axis lines through space:
    xAxisLine = ((min(principal_components[:, 0]), max(principal_components[:, 0])), (0, 0), (0, 0))
    ax.plot(xAxisLine[0], xAxisLine[1], xAxisLine[2], 'r')
    yAxisLine = ((0, 0), (min(principal_components[:, 1]), max(principal_components[:, 1])), (0, 0))
    ax.plot(yAxisLine[0], yAxisLine[1], yAxisLine[2], 'r')
    zAxisLine = ((0, 0), (0, 0), (min(principal_components[:, 2]), max(principal_components[:, 2])))
    ax.plot(zAxisLine[0], zAxisLine[1], zAxisLine[2], 'r')
    # label the axes
    ax.set_xlabel(f"PC1: {pca.explained_variance_ratio_[0] * 100:.2f}% explained variance")
    ax.set_ylabel(f"PC2: {pca.explained_variance_ratio_[1] * 100:.2f}% explained variance")
    ax.set_zlabel(f"PC3: {pca.explained_variance_ratio_[2] * 100:.2f}% explained variance")
    ax.set_title("PCA on the input features")
    plt.show()

def show_boxplot(df):
    for (columnName, columnData) in df.iteritems():
        print(f"{columnName}")
        print(f"Min: {columnData.min()}")
        print(f"Q1: {columnData.quantile(q=0.25)}")
        print(f"Q2: {columnData.quantile(q=0.5)}")
        print(f"Q3: {columnData.quantile(q=0.75)}")
        print(f"Max: {columnData.max()}")
    df.boxplot(rot=90)
    plt.title("Raw measurements boxplot")
    plt.show()

if __name__ == "__main__":
    breast_reduction_df = pd.read_csv("./data/RUMC Reduction Mammaplasties - MS 8-18-2021.csv")
    breast_reduction_df = breast_reduction_df.set_index(UNIQUE_ID_COL)
    show_boxplot(breast_reduction_df[INPUT_FEATURE_COLS[:-1]])
    # breast_reduction_df = prepare_per_breast_df(breast_reduction_df)
    # show_correlation_matrix(breast_reduction_df[INPUT_FEATURE_COLS])
    # show_boxplot(breast_reduction_df[TRAIN_FEATURE_COLS])
    # plot_principal_components(pca, pricipal_components)
