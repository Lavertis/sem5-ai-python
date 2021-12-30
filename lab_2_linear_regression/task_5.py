import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split


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


bunch = load_diabetes(as_frame=True)
df = bunch.data
columns = df.columns.to_list()
values = df.values

# podzielenie zbioru na wejście i wyjście
x = values
y = np.array(bunch.target)

mape_vanilla = []
mape_no_outliers = []
mape_no_outliers_additional_x = []
for _ in range(10):
    # podzielenie zbioru na testowy i treningowy
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)

    # dokonanie predykcji za pomocą regresji liniowej
    linReg = train_model(x_train, y_train)
    y_pred = linReg.predict(x_test)

    # wykres wyników
    plot_linear_regression_results(y_test, y_pred)

    # metryka wyników regresji
    mape = mean_absolute_percentage_error(y_test, y_pred)
    mape_vanilla.append(mape)

    # usunięcie cech odstających lub zastąpienie ich średnią
    x_train, y_train = remove_outliers(x_train, y_train, 1.5)
    # x_train, y_train = replace_outliers_with_mean(x_train, y_train, 2.5)
    linReg = train_model(x_train, y_train)
    y_pred = linReg.predict(x_test)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    mape_no_outliers.append(mape)

    # wykres słupkowy poszczególnych wag cech niezależnych
    generate_weights_plots(linReg.coef_, columns)

    # generacja nowych cech
    x_additional = generate_new_features(x)
    x_train, x_test, y_train, y_test = train_test_split(x_additional, y, test_size=0.2, random_state=221, shuffle=False)
    linReg = train_model(x_train, y_train)
    y_pred = linReg.predict(x_test)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    mape_no_outliers_additional_x.append(mape)

print(f'mape vanilla: {round(np.mean(mape_vanilla), 3)}')
print(f'mape no outliers: {round(np.mean(mape_no_outliers), 3)}')
print(f'mape no outliers with additional x: {round(np.mean(mape_no_outliers_additional_x), 3)}')
