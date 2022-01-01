import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as kNN
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC as SVM
from sklearn.tree import DecisionTreeClassifier as DT


def do_test(model_):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)
    pipeline = Pipeline([
        ['scaler', StandardScaler()],
        ['transformer', PCA(0.95)],
        ['classifier', model_]
    ])

    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict(x_test)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    return cm


df = pd.read_csv('voice_extracted_features.csv', sep=',')

# podział danych na x i y
x = df.values[:, :-1]
# x = x.astype(np.float)
y = df.values[:, -1]

# zamiana jakościowych cech binarnych na 0 i 1
mask = y == 'female'
y[mask] = 0
y[~mask] = 1
y = y.astype(np.int)

confusion_matrices = []
models = [kNN(), SVM(), DT()]
tests_number = 30
for model in models:
    for _ in range(tests_number):
        confusion_matrices.append(do_test(model))

confusion_matrix_mean = np.array([[0, 0], [0, 0]])
for matrix in confusion_matrices:
    confusion_matrix_mean[0, 0] += matrix[0, 0]
    confusion_matrix_mean[0, 1] += matrix[0, 1]
    confusion_matrix_mean[1, 0] += matrix[1, 0]
    confusion_matrix_mean[1, 1] += matrix[1, 1]

confusion_matrix_mean[0, 0] = confusion_matrix_mean[0, 0] / (tests_number * len(models))
confusion_matrix_mean[0, 1] = confusion_matrix_mean[0, 1] / (tests_number * len(models))
confusion_matrix_mean[1, 0] = confusion_matrix_mean[1, 0] / (tests_number * len(models))
confusion_matrix_mean[1, 1] = confusion_matrix_mean[1, 1] / (tests_number * len(models))

print()
print(confusion_matrix_mean)
print('Minimalnie lepiej wykrywa mężczyzn')
