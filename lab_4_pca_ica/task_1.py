import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as kNN
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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

# podział danych na zbiór treningowy i testowy
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)

# normalizacja danych
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# PCA dla wykresu 2 składowych głównych
pca_transformer = PCA(2)
pca_transformer.fit(x_train)
x_train_pcaed = pca_transformer.transform(x_train)

# wykres 2 składowych głównych
fig, ax = plt.subplots(1, 1)
females = y_train == 0
ax.scatter(x_train_pcaed[females, 0], x_train_pcaed[females, 1], label='female')
ax.scatter(x_train_pcaed[~females, 0], x_train_pcaed[~females, 1], label='male')
ax.legend()
plt.show()

# wykres wyjaśnionej wariancji
pca_transformer = PCA()
pca_transformer.fit(x_train)
variances = pca_transformer.explained_variance_ratio_
variances_cumulated = variances.cumsum()
plt.scatter(np.arange(variances.shape[0]), variances_cumulated)
plt.yticks(np.arange(0, 1.1, 0.1))
plt.show()

# PCA dla klasyfikacji
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)
PC_num = (variances_cumulated < 0.95).sum() + 1

pipeline = Pipeline([
    ['scaler', StandardScaler()],
    ['transformer', PCA(0.95)],
    ['classifier', kNN(weights='distance')]
])

pipeline.fit(x_train, y_train)
y_pred = pipeline.predict(x_test)
score_f1 = round(f1_score(y_test, y_pred), 3)
print(score_f1)
