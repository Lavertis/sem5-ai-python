import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split


def train_model(x_train, y_train):
    linReg = LinearRegression()
    linReg.fit(x_train, y_train)
    return linReg


def test_model(x, y):
    # podzielenie zbioru na testowy i treningowy
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)
    linReg = train_model(x_train, y_train)
    y_pred = linReg.predict(x_test)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    return mape


bh_data = pd.read_csv('practice_lab_2.csv', sep=';')
bh_columns = bh_data.columns.to_list()
bh_values = bh_data.values

# podzielenie zbioru na wejście i wyjście
x = bh_values[:, :-1]  # for all but last column
y = bh_values[:, -1]  # for last column

mapes = []
for i in range(10):
    mape = test_model(x, y)
    mapes.append(mape)

mapes_mean = np.mean(mapes)
print(round(np.mean(mapes), 3))
