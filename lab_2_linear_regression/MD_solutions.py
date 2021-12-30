import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split


def sol_2_1(path: str, sep: str = ';') -> pd.DataFrame:
    data = pd.read_csv(path, sep=sep)

    features = list(data.columns)
    features_count = len(features)
    fig, ax = plt.subplots(1, features_count, figsize=(50, 10))
    for i in range(features_count):
        ax[i].scatter(data.values[:, i], data.values[:, -1])
        ax[i].set_xlabel("x")
        ax[i].set_ylabel("y")
        fig.tight_layout()
    # fig.show()

    return data.corr()


def sol_2_2(path: str, sep: str = ';', iterations: int = 10):
    data = pd.read_csv(path, sep=sep)
    x = data.values[:, :-1]
    y = data.values[:, -1]

    linReg = LinearRegression()
    mpes = []
    for i in range(iterations):
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, shuffle=True)
        linReg.fit(x_train, y_train)
        y_pred = linReg.predict(x_test)
        mpes.append(mean_absolute_percentage_error(y_test, y_pred))
    print(mpes)


if __name__ == "__main__":
    path = "practice_lab_2.csv"

    corr = sol_2_1(path)

    sol_2_2(path, iterations=100)
