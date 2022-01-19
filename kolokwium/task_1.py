import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression


def load_sklearn_dataset_as_dataframe(dataset):
    data = np.c_[dataset.data, dataset.target]
    columns = np.append(dataset.feature_names, ["target"])
    return pd.DataFrame(data, columns=columns)


df = load_sklearn_dataset_as_dataframe(load_diabetes())

mask = df['sex'].values > 0

data_1_sex = df.values[mask, :]
data_2_sex = df.values[~mask, :]

x_1 = data_1_sex[:, :-1]
y_1 = data_1_sex[:, -1]

x_2 = data_2_sex[:, :-1]
y_2 = data_2_sex[:, -1]

linReg_1 = LinearRegression()
linReg_1.fit(x_1, y_1)

linReg_2 = LinearRegression()
linReg_2.fit(x_2, y_2)

print('Model 1:')
print(linReg_1.coef_)

print('\nModel 2:')
print(linReg_2.coef_)
