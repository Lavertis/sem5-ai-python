import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import FastICA
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as kNN
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC as SVM
from sklearn.tree import DecisionTreeClassifier as DT


def change_qualitative_feature_of_2_types_to_01(data_frame, column_name, feat_0_name):
    mask = data_frame[column_name].values == feat_0_name
    data_frame[column_name].values[mask] = 1
    data_frame[column_name].values[~mask] = 0
    return data_frame


def scale_data(scaler, x_train_, x_test_):
    scaler.fit(x_train_)
    x_train_ = scaler.transform(x_train_)
    x_test_ = scaler.transform(x_test_)
    return x_train_, x_test_


def get_number_needed_for_variance_percentage(x_train_, percentage):
    pca_transformer_ = PCA()
    pca_transformer_.fit(x_train_)
    variances_ = pca_transformer_.explained_variance_ratio_
    cumulated_variances_ = variances_.cumsum()
    pca_num_ = (cumulated_variances_ < percentage).sum() + 1
    return pca_num_


def plot_cumulated_variances(variances_, cumulated_variances_):
    plt.scatter(np.arange(variances_.shape[0]), cumulated_variances_)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.show()


def plot_2_first_principal_components(x_train_, y_train_):
    x_pcaed = PCA(2).fit_transform(x_train_)
    fig, ax = plt.subplots(1, 1)
    females = y_train_ == 1
    ax.scatter(x_pcaed[females, 0], x_pcaed[females, 1], label='female')
    ax.scatter(x_pcaed[~females, 0], x_pcaed[~females, 1], label='male')
    ax.legend()
    plt.show()


def do_test(classifier, x_train_, x_test_, y_train_, y_test_):
    model = classifier
    model.fit(x_train_, y_train_)
    _y_pred = model.predict(x_test_)
    cm = confusion_matrix(y_test_, _y_pred)
    print(model)
    # print(cm)
    print(round(np.diag(cm).sum() / cm.sum(), 3), end='\n\n')


def do_three_tests(_x_train, _x_test, _y_train, _y_test):
    models = [kNN(), SVM(), DT(max_depth=10)]
    for model in models:
        do_test(model, _x_train, _x_test, _y_train, _y_test)


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
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)

# normalizacja danych
x_train_scaled, x_test_scaled = scale_data(StandardScaler(), x_train, x_test)

# check results with models
print("========== Original ==========")
do_three_tests(x_train_scaled, x_test_scaled, y_train, y_test)

# ==================== PCA ====================
pca_transformer = PCA()
pca_transformer.fit(x_train_scaled)
variances = pca_transformer.explained_variance_ratio_
cumulated_variances = variances.cumsum()
PC_num = (cumulated_variances < 0.95).sum() + 1

# get number needed for percentage
num_095_percentage = get_number_needed_for_variance_percentage(x_train_scaled, 0.95)

# plot cumulated variances
plot_cumulated_variances(variances, cumulated_variances)

# plot 2 first principal components
plot_2_first_principal_components(x_train_scaled, y_train)

# check results with models
pca_transformer = PCA(0.95)
pca_transformer.fit(x_train_scaled)
x_train_transformed = pca_transformer.transform(x_train_scaled)
x_test_transformed = pca_transformer.transform(x_test_scaled)
print("========== PCA ==========")
do_three_tests(x_train_transformed, x_test_transformed, y_train, y_test)

# ==================== ICA ====================
x_ica, y_ica = load_digits(return_X_y=True)

# podział zbioru na treningowy i testowy
x_train_ica, x_test_ica, y_train_ica, y_test_ica = train_test_split(x_ica, y_ica, test_size=0.2, shuffle=False)

# normalizacja danych
x_train_ica_scaled, x_test_ica_scaled = scale_data(StandardScaler(), x_train_ica, x_test_ica)

# użycie ICA
ica_transformer = FastICA(n_components=num_095_percentage, random_state=0, max_iter=300)
ica_transformer.fit(x_train_ica_scaled)
x_train_ica_transformed = ica_transformer.transform(x_train_ica_scaled)
x_test_ica_transformed = ica_transformer.transform(x_test_ica_scaled)

# check results with models
print("========== FastICA ==========")
do_three_tests(x_train_ica_transformed, x_test_ica_transformed, y_train_ica, y_test_ica)

# ==================== Pipeline ====================
print("========== Pipeline ==========")
pipe = Pipeline([['transformer', PCA(9)],
                 ['scaler', StandardScaler()],
                 ['classifier', kNN(weights='distance')]])
pipe.fit(x_train, y_train)
y_pred = pipe.predict(x_test)
score_f1 = round(f1_score(y_test, y_pred), 3)
print(score_f1)
