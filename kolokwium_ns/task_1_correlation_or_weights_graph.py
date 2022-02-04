import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def load_sklearn_dataset_as_dataframe(dataset):
    data = np.c_[dataset.data, dataset.target]
    columns = np.append(dataset.feature_names, ["target"])
    return pd.DataFrame(data, columns=columns)


def plot_columns_correlation(corr_matrix_):
    fig, ax = plt.subplots(1, 1)
    indexes = np.arange(len(corr_matrix_.columns))

    ax.bar(indexes, corr_matrix_.values[-1, :])
    ax.set_xticks(indexes)
    ax.set_xticklabels(corr_matrix_.columns, rotation=0)
    ax.set_title('Wykres korelacji')
    ax.set_ylabel('współczynniki korelacji')
    ax.set_xlabel('cechy')
    fig.tight_layout()
    plt.show()


def generate_weights_plots(weights, independent_features):
    fig, ax = plt.subplots(1, 1)
    x = np.arange(len(independent_features))
    ax.bar(x, weights)
    ax.set_xticks(x)
    ax.set_xticklabels(independent_features, rotation=90)
    fig.tight_layout()
    plt.show()


df = load_sklearn_dataset_as_dataframe(load_diabetes())
df = df[['sex', 'bmi', 's2', 's4', 's6', 'target']]

corr_matrix = df.corr()
corr_matrix = corr_matrix.drop(columns='target')
plot_columns_correlation(corr_matrix)

x = df.values[:, :-1]
y = df.values[:, -1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)
linReg = LinearRegression()
linReg.fit(x_train, y_train)
coefs = linReg.coef_

generate_weights_plots(coefs, ['sex', 'bmi', 's2', 's4', 's6'])
