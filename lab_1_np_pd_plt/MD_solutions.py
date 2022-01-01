# -*- coding: utf-8 -*-

"""
Solutions for LAB 1
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

np.set_printoptions(precision=3, suppress=True)


# Ex 1.2 1
def sol_1_2_1(verbose: bool = False) -> np.ndarray:
    data_csv = pd.read_csv("practice_lab_1.csv", sep=';')

    data_p = data_csv.values[::2, :]
    data_np = data_csv.values[1::2, :]

    res1 = data_p - data_np

    # the same in one line
    # res2 = data_csv.values[::2, :] - data_csv.values[1::2, :]

    if verbose:
        print("\n====  1.2 1  ====\n")
        print(">>>> Data (even rows):")
        print(data_p[:5, :])
        print(data_p.shape)
        print(">>>> Data (odd rows):")
        print(data_np[:5, :])
        print(data_np.shape)
        print(">>>> Final result:")
        print(res1[:5, :])
        print(res1.shape)

    return res1


# Ex 1.2 2
def sol_1_2_2(verbose: bool = False) -> np.ndarray:
    data_csv = pd.read_csv("practice_lab_1.csv", sep=';')

    res = (data_csv.values - data_csv.values.mean()) / \
          (data_csv.values.std() + np.spacing(data_csv.values.std()))

    if verbose:
        print("\n====  1.2 2  ====\n")
        print(">>>> Mean:")
        print(data_csv.values.mean())
        print(">>>> Std dev:")
        print(data_csv.values.std())
        print(">>>> Spacing std dev:")
        print(np.spacing(data_csv.values.std()))

    return res


# Ex 1.2 3
def sol_1_2_3(verbose: bool = False) -> np.ndarray:
    data_csv = pd.read_csv("practice_lab_1.csv", sep=';')

    res = (data_csv.values - data_csv.values.mean(axis=0)) / \
          (data_csv.values.std(axis=0) + np.spacing(data_csv.values.std(axis=0)))

    if verbose:
        print("\n====  1.2 3  ====\n")
        print(">>>> Mean:")
        print(data_csv.values.mean(axis=0))
        print(">>>> Std dev:")
        print(data_csv.values.std(axis=0))
        print(">>>> Spacing + std dev:")
        print(data_csv.values.std(axis=0) +
              np.spacing(data_csv.values.std(axis=0)))

    return res


# Ex 1.2 4
def sol_1_2_4(verbose: bool = False) -> np.ndarray:
    data_csv = pd.read_csv("practice_lab_1.csv", sep=';')

    res = data_csv.values.mean(
        axis=0) / (data_csv.values.std(axis=0) + np.spacing(data_csv.values.std(axis=0)))

    if verbose:
        print("\n====  1.2 4  ====\n")
        print(">>>> Mean:")
        print(data_csv.values.mean(axis=0))
        print(">>>> Std dev:")
        print(data_csv.values.std(axis=0))
        print(">>>> Spacing + std dev:")
        print(data_csv.values.std(axis=0) +
              np.spacing(data_csv.values.std(axis=0)))

    return res


# Ex 1.2 5
def sol_1_2_5(verbose: bool = False) -> int:
    if verbose:
        print("\n====  1.2 5  ====\n")

    data = sol_1_2_4(verbose)

    return np.argmax(data)


# Ex 1.3 3
def sol_1_3_3(verbose: bool = False) -> None:
    x = np.arange(-5, 5, 0.1)
    x = np.append(x, 5)

    y = 1 / (1 + np.exp(-1 * x))

    plt.clf()
    plt.plot(x, y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("Function 3")

    if verbose:
        print("\n====  1.3 3  ====\n")
        print(">>>> X:")
        print(x)
        print(">>>> Y:")
        print(y)


# Ex 1.3 5
def sol_1_3_5(verbose: bool = False) -> None:
    x = np.arange(-5, 5, 0.1)
    x = np.append(x, 5)

    mask = x <= 0

    x1 = x[mask]
    x2 = x[~mask]

    y1 = x1
    y2 = np.exp(x2) - 1
    y = np.append(y1, y2)

    plt.clf()
    plt.plot(x, y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("Function 5")

    if verbose:
        print("\n====  1.3 5  ====\n")
        print(">>>> X:")
        print(x)
        print(">>>> X1:")
        print(x1)
        print(">>>> X2:")
        print(x2)
        print(">>>> Y:")
        print(y)


# Ex 1.4
def sol_1_4(verbose: bool = False) -> None:
    data_csv = pd.read_csv("practice_lab_1.csv", sep=';')

    data_corr = data_csv.corr()

    rows = data_csv.values.shape[0]
    cols = data_csv.values.shape[1]
    x = np.arange(0, rows, 1)

    norm = (data_csv.values - data_csv.values.mean(axis=0)) / \
           (data_csv.values.std(axis=0) + np.spacing(data_csv.values.std(axis=0)))

    plt.clf()
    fig, el = plt.subplots(cols, cols, figsize=(15, 15))
    for i in range(cols):
        for j in range(cols):
            el[i, j].scatter(x, norm[:, i])
            el[i, j].scatter(x, norm[:, j])
            el[i, j].set_title(str(data_corr.iloc[i, j])[:6])

    fig.tight_layout()
    plt.show()

    if verbose:
        print("\n====  1.4  ====\n")


if __name__ == '__main__':
    res_1_2_1 = sol_1_2_1(True)
    res_1_2_2 = sol_1_2_2(True)
    res_1_2_3 = sol_1_2_3(True)
    res_1_2_4 = sol_1_2_4(True)
    res_1_2_5 = sol_1_2_5(True)

    # Run only one from functions below
    # sol_1_3_3(True)
    # sol_1_3_5(True)

    # sol_1_4(True)
