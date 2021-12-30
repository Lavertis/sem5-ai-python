import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split


def approximation_of_a_noisy_function():
    x = np.arange(-3, 3, 0.1).reshape((-1, 1))
    y = np.tanh(x) + np.random.randn(x.shape[0], x.shape[1]) * 0.2
    y_pred = LinearRegression().fit(x, y).predict(x)

    plt.scatter(x, y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(x, y_pred)
    plt.legend(['F(x) - aproksymująca', 'f(x) - aproksymowana zaszumiona'])
    plt.show()


def train_model(x_train, y_train):
    linReg = LinearRegression()
    linReg.fit(x_train, y_train)
    return linReg


def plot_linear_regression_results(y_test, y_pred):
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.scatter(y_test, y_pred)
    plt.plot([min_val, max_val], [min_val, max_val])
    plt.xlabel('y_test')
    plt.ylabel('y_pred')
    plt.show()


def calculate_regression_metric(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    print(f'mse: {round(mse, 3)}')
    print(f'mae: {round(mae, 3)}')
    print(f'mape: {round(mape, 3)}')


def generate_boxplot(y_train):
    plt.boxplot(y_train)
    plt.title("Medianowa wartosc mieszkania")
    plt.show()


def remove_outliers(x_train, y_train, value):
    outliers = np.abs((y_train - y_train.mean()) / y_train.std()) > value
    x_train_no_outliers = x_train[~outliers, :]
    y_train_no_outliers = y_train[~outliers]
    return x_train_no_outliers, y_train_no_outliers


def replace_outliers_with_mean(x_train, y_train, value):
    outliers = np.abs((y_train - y_train.mean()) / y_train.std()) > value
    y_train_mean = y_train.copy()
    y_train_mean[outliers] = y_train.mean()
    return x_train, y_train_mean


def generate_weights_plots(weights, independent_features):
    fig, ax = plt.subplots(1, 1)
    x = np.arange(len(independent_features))
    ax.bar(x, weights)
    ax.set_xticks(x)
    ax.set_xticklabels(independent_features, rotation=90)
    fig.tight_layout()
    plt.show()


def generate_new_features(x):
    new_data = np.stack([x[:, 4] / x[:, 7],
                         x[:, 4] / x[:, 5],
                         x[:, 4] * x[:, 3],
                         x[:, 4] / x[:, -1]], axis=-1)
    x_additional = np.concatenate([x, new_data], axis=-1)
    return x_additional


def boston_housing():
    bh_data = pd.read_csv('practice_lab_2.csv', sep=';')
    bh_columns = bh_data.columns.to_list()
    bh_values = bh_data.values

    # podzielenie zbioru na wejście i wyjście
    x = bh_values[:, :-1]  # for all but last column
    y = bh_values[:, -1]  # for last column

    # podzielenie zbioru na testowy i treningowy
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=221, shuffle=False)

    # dokonanie predykcji za pomocą regresji liniowej
    linReg = train_model(x_train, y_train)
    y_pred = linReg.predict(x_test)

    # wykres wyników
    plot_linear_regression_results(y_test, y_pred)

    # metryka wyników regresji
    print('Początkowa metryka wyników regresji')
    calculate_regression_metric(y_test, y_pred)

    # wykres pudełkowy dla cechy zależnej
    generate_boxplot(y_train)

    # usunięcie cech odstających lub zastąpienie ich średnią
    x_train, y_train = remove_outliers(x_train, y_train, 2.5)
    # x_train, y_train = replace_outliers_with_mean(x_train, y_train, 2.5)

    print('\nMetryka wyników regresji po usunięciu outlierów')
    linReg = train_model(x_train, y_train)
    y_pred = linReg.predict(x_test)
    calculate_regression_metric(y_test, y_pred)

    # wykres pudełkowy dla cechy zależnej po zajęciu się outlierami
    generate_boxplot(y_train)

    # wykres słupkowy poszczególnych wag cech niezależnych
    generate_weights_plots(linReg.coef_, bh_columns[:-1])

    # generacja nowych cech
    print('\nMetryka wyników regresji po generacji nowych cech')
    x_additional = generate_new_features(x)
    x_train, x_test, y_train, y_test = train_test_split(x_additional, y, test_size=0.2, random_state=221, shuffle=False)
    linReg = train_model(x_train, y_train)
    y_pred = linReg.predict(x_test)
    calculate_regression_metric(y_test, y_pred)


boston_housing()
