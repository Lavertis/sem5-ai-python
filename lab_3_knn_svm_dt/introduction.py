import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as kNN
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.svm import SVC as SVM
from sklearn.tree import DecisionTreeClassifier as DT, plot_tree


def get_readable_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
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


def change_qualitative_feature_of_2_types_to_01(data_frame, column_name, feat_0_name):
    mask = data_frame[column_name].values == feat_0_name
    data_frame[column_name].values[mask] = 1
    data_frame[column_name].values[~mask] = 0
    return data_frame


def change_qualitative_feature_of_more_than_2_types_to_numbers(data_frame, column_name, target_column_name):
    # preserve target and later append to the end
    target = data_frame[target_column_name]
    data_frame = data_frame.drop(columns=[target_column_name])

    cat_feature = pd.Categorical(data_frame[column_name])
    one_hot = pd.get_dummies(cat_feature)
    data_frame = pd.concat([data_frame, one_hot, target], axis=1)
    data_frame = data_frame.drop(columns=[column_name])
    return data_frame


def get_x_y_from_dataframe(data_frame, target_column_name):
    y_col_name_ = data_frame.columns[-1]
    x_col_names_ = np.array(data_frame.columns[:-1])
    y_ = data_frame[target_column_name].values.astype(np.float)
    x_ = data_frame.drop(columns=[target_column_name]).values.astype(np.float)
    return x_, y_, x_col_names_, y_col_name_


def scale_data(scaler, x_train_, x_test_):
    scaler.fit(x_train_)
    x_train_ = scaler.transform(x_train_)
    x_test_ = scaler.transform(x_test_)
    return x_train_, x_test_


def test_models(models, x_train_, x_test_, y_train_, y_test_):
    for model in models:
        model.fit(x_train_, y_train_)
        y_pred = model.predict(x_test_)
        score_f1 = round(f1_score(y_test_, y_pred), 3)
        cm = confusion_matrix(y_test_, y_pred)
        readable_cm = get_readable_confusion_matrix(y_test, y_pred)
        print(model)
        print(cm)
        print(readable_cm)
        print(f'f1_score: {score_f1}', end='\n\n')


def plot_decision_tree(x_train_, y_train_, x_col_names_, target_class_names_):
    model = DT(max_depth=3)
    model.fit(x_train_, y_train_)
    plt.figure(figsize=(20, 10))
    plot_tree(model, feature_names=x_col_names_, class_names=target_class_names_, fontsize=20)
    plt.show()


df = pd.read_csv('practice_lab_3.csv', sep=';')
columns = list(df.columns)

# zamiana binarnych cech jakościowych na 0 i 1
df = change_qualitative_feature_of_2_types_to_01(df, 'Gender', 'Female')
df = change_qualitative_feature_of_2_types_to_01(df, 'Married', 'No')
df = change_qualitative_feature_of_2_types_to_01(df, 'Education', 'Not Graduate')
df = change_qualitative_feature_of_2_types_to_01(df, 'Self_Employed', 'No')
df = change_qualitative_feature_of_2_types_to_01(df, 'Loan_Status', 'N')

# zamiana niebinarnych cech jakościowych na 0...n
df = change_qualitative_feature_of_more_than_2_types_to_numbers(df, 'Property_Area', 'Loan_Status')

# podzielenie zbioru na x i y
x, y, x_col_names, y_col_name = get_x_y_from_dataframe(df, 'Loan_Status')

# podzielenie zbioru na testowy i treningowy
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)

# testowanie modeli
print('==================== Vanilla models ====================')
models = [kNN(), SVM(), DT()]
test_models(models, x_train, x_test, y_train, y_test)

# testowanie modeli po przeskalowaniu danych
print('==================== Models after StandardScaler ====================')
x_train_scaled, x_test_scaled = scale_data(StandardScaler(), x_train, x_test)
models = [kNN(), SVM(), DT()]
test_models(models, x_train_scaled, x_test_scaled, y_train, y_test)

print('==================== Models after MinMaxScaler ====================')
x_train_scaled, x_test_scaled = scale_data(MinMaxScaler(), x_train, x_test)
models = [kNN(), SVM(), DT()]
test_models(models, x_train_scaled, x_test_scaled, y_train, y_test)

print('==================== Models after RobustScaler ====================')
x_train_scaled, x_test_scaled = scale_data(RobustScaler(), x_train, x_test)
models = [kNN(), SVM(), DT()]
test_models(models, x_train_scaled, x_test_scaled, y_train, y_test)

# rysowanie drzewa decyzyjnego
plot_decision_tree(x_train_scaled, y_train, x_col_names, ['No', 'Yes'])

print('==================== Models with custom attributes ====================')
x_train_scaled, x_test_scaled = scale_data(StandardScaler(), x_train, x_test)
models = [kNN(n_neighbors=3, weights='uniform'), kNN(weights='uniform'), kNN(n_neighbors=10, weights='uniform'),
          kNN(n_neighbors=3, weights='distance'), kNN(weights='distance'), kNN(n_neighbors=10, weights='distance'),
          SVM(kernel='linear'), SVM(kernel='poly'), SVM(kernel='rbf')]
test_models(models, x_train_scaled, x_test_scaled, y_train, y_test)
