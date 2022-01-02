import pandas as pd
from sklearn.decomposition import PCA, FastICA
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC as SVM


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


def do_test(classifier, x_train_, x_test_, y_train_, y_test_):
    model = classifier
    model.fit(x_train_, y_train_)
    y_pred_ = model.predict(x_test_)
    score_f1 = round(f1_score(y_test_, y_pred_), 3)
    print(model)
    print(score_f1, end='\n\n')


def do_three_tests(x_train_, x_test_, y_train_, y_test_):
    models = [SVM(kernel='linear'), SVM(kernel='poly'), SVM(kernel='rbf')]
    for model in models:
        do_test(model, x_train_, x_test_, y_train_, y_test_)


df = pd.read_csv('credit.asc', sep=' ')
x = df.values[:, :-1]
y = df.values[:, -1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)
x_train, x_test = scale_data(StandardScaler(), x_train, x_test)

PC_num_075 = get_number_needed_for_variance_percentage(x_train, 0.75)
PC_num_080 = get_number_needed_for_variance_percentage(x_train, 0.8)

print('==================== Original data ====================')
do_three_tests(x_train, x_test, y_train, y_test)

transformers = [PCA(PC_num_075), PCA(PC_num_080), FastICA(PC_num_075), FastICA(PC_num_080)]
for transformer in transformers:
    transformer.fit(x_train)
    x_train_transformed = transformer.transform(x_train)
    x_test_transformed = transformer.transform(x_test)
    print(f'========== {transformer} ==========')
    do_three_tests(x_train_transformed, x_test_transformed, y_train, y_test)
