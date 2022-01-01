import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as kNN
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def change_qualitative_feature_of_2_types_to_01(data_frame, column_name, feat_0_name):
    mask = data_frame[column_name].values == feat_0_name
    data_frame[column_name].values[mask] = 1
    data_frame[column_name].values[~mask] = 0
    return data_frame


class PrincipalComponents:
    def __init__(self, cumulated_variance_percentage):
        self.pca = PCA(cumulated_variance_percentage)

    def fit(self, x_, y_=None):
        self.pca.fit(x_)
        return self

    def transform(self, x_, y_=None):
        return self.pca.transform(x_)

    def fit_transform(self, x_, y_=None):
        return self.pca.fit_transform(x_)


def remove_outliers(x_train, y_train, value):
    outliers = np.abs((y_train - y_train.mean()) / y_train.std()) > value
    x_train_no_outliers = x_train[~outliers, :]
    y_train_no_outliers = y_train[~outliers]
    return x_train_no_outliers, y_train_no_outliers


class OutliersRemover:
    def __init__(self, threshold):
        self.threshold = threshold

    def fit(self, *_):
        return self

    def transform(self, x_train_):
        def func(x_):
            outliers = np.abs((x_ - x_.mean()) / x_.std()) > self.threshold
            x_[outliers] = x_.mean()

        x_train_cpy = x_train_.copy()
        np.apply_along_axis(func1d=func, axis=0, arr=x_train_cpy)
        return x_train_cpy

    def fit_transform(self, x_train_, _):
        return self.transform(x_train_)


df = pd.read_csv("voice_extracted_features.csv", sep=',')

# zamiana female i male na 0 i 1
df = change_qualitative_feature_of_2_types_to_01(df, 'label', 'female')

# podział na nazwy kolumn i wartości kolumn
column_names = list(df.columns)
values = df.values.astype(np.float)

# podział na x i y
x = values[:, :-1]
y = values[:, -1]

# podział zbioru na treningowy i testowy
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)

pipeline1 = Pipeline([
    ['outliers_remover', OutliersRemover(2.0)],
    ['scaler', StandardScaler()],
    ['transformer', PrincipalComponents(0.95)],
    ['classifier', kNN()]
])

pipeline2 = Pipeline([
    ['scaler', StandardScaler()],
    ['transformer', PrincipalComponents(0.95)],
    ['classifier', kNN()]
])

pipeline1.fit(x_train, y_train)
y_pred1 = pipeline1.predict(x_test)

pipeline2.fit(x_train, y_train)
y_pred2 = pipeline2.predict(x_test)

score_f1_1 = f1_score(y_test, y_pred1)
score_f1_2 = f1_score(y_test, y_pred2)

print(f'Without outliers: {round(score_f1_1, 6)}')
print(f'With outliers: {round(score_f1_2, 6)}')
