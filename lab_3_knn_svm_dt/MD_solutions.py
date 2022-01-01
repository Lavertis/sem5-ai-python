import numpy as np
import pandas as pd


def sol_3_2(data: pd.DataFrame, column: str, val_1: str):
    mask = data[column].values == val_1
    data[column].values[mask] = 1
    data[column].values[~mask] = 0
    return data


def __helper_3_4(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    y_true	y_pred  y_t+y_t+y_p
    0	    0	    0		        TN
    0	    1	    1		        FP
    1	    0	    2		        FN
    1	    1	    3		        TP

    """
    tmp = y_true + y_true + y_pred
    tn = (tmp == 0).sum()
    fp = (tmp == 1).sum()
    fn = (tmp == 2).sum()
    tp = (tmp == 3).sum()

    return {"TN": tn, "FP": fp, "FN": fn, "TP": tp}


def sol_3_4_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    m = __helper_3_4(y_true, y_pred)
    return (m["TP"] + m["TN"]) / (m["TP"] + m["TN"] + m["FP"] + m["FN"])


def sol_3_4_sensivity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    m = __helper_3_4(y_true, y_pred)
    return m["TP"] / (m["TP"] + m["FN"])


def sol_3_4_precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    m = __helper_3_4(y_true, y_pred)
    return m["TP"] / (m["TP"] + m["FP"])


def sol_3_4_specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    m = __helper_3_4(y_true, y_pred)
    return m["TN"] / (m["TN"] + m["FP"])


def sol_3_4_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    sens = sol_3_4_sensivity(y_true, y_pred)
    prec = sol_3_4_precision(y_true, y_pred)
    return 2 * sens * prec / (sens + prec)


if __name__ == "__main__":
    path = "../../data/practice_lab_3.csv"

    data = pd.read_csv(path, sep=';')

    print(data)

    col_names = ["Gender", "Married", "Education",
                 "Self_Employed", "Loan_Status"]
    col_values_1 = ["Male", "Yes", "Graduate", "Yes", "Y"]

    for i in range(len(col_names)):
        data = sol_3_2(data, col_names[i], col_values_1[i])

    print(data)

    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    print(sol_3_4_accuracy(y_true, y_pred))
    print(sol_3_4_precision(y_true, y_pred))
    print(sol_3_4_sensivity(y_true, y_pred))
    print(sol_3_4_specificity(y_true, y_pred))
    print(sol_3_4_f1(y_true, y_pred))
