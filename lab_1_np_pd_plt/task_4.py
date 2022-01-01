import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('practice_lab_1.csv', sep=';')
corr_matrix = df.corr()
values = df.values
size = values.shape[0]

scaler = StandardScaler()
normalized = scaler.fit_transform(values)

fig, ax = plt.subplots(7, 7, figsize=(15, 15))
indexes = np.arange(0, values.shape[0], 1)

for row in range(7):
    for col in range(7):
        col_1 = normalized[:, row]
        col_2 = normalized[:, col]
        ax[row, col].scatter(indexes, col_1, s=5)
        ax[row, col].scatter(indexes, col_2, s=5)
        ax[row, col].set_title(round(corr_matrix.values[row, col], 3))

fig.tight_layout()
plt.show()
