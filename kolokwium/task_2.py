import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.svm import SVC as SVM


def do_test(classifier, x_train_, x_test_, y_train_, y_test_):
    model = classifier
    model.fit(x_train_, y_train_)
    y_pred_ = model.predict(x_test_)
    score_f1 = round(f1_score(y_test_, y_pred_), 3)
    print(model)
    print('F1 score:', score_f1, end='\n\n')


def do_three_tests(x_train_, x_test_, y_train_, y_test_):
    models = [SVM(kernel='linear'), SVM(kernel='sigmoid')
              # , SVM(kernel='precomputed') # przy obecnym formacie danych nie możemy użyć tego modelu
              ]
    for model in models:
        do_test(model, x_train_, x_test_, y_train_, y_test_)


df = pd.read_csv('credit.asc', sep=' ')
x = df.values[:, :-1]
y = df.values[:, -1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)

print('==================== Original data ====================')
do_three_tests(x_train, x_test, y_train, y_test)  # bez skalowania robi się strasznie długo

scalers = [MinMaxScaler(), StandardScaler(), RobustScaler()]
for scaler in scalers:
    scaler.fit(x_train)
    x_train_scaled = scaler.transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    print(f'========== {scaler} ==========')
    do_three_tests(x_train_scaled, x_test_scaled, y_train, y_test)
