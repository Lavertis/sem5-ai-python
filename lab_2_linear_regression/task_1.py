import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv('practice_lab_2.csv', sep=';')
corr_matrix = df.corr()
columns = df.columns
values = df.values
x = values[:, :-1]
y = values[:, -1]

fig, ax = plt.subplots(13, 1, figsize=(4, 30))
for i in range(13):
    indexes = np.arange(0, x.shape[0], 1)
    ax[i].scatter(x[:, i], y, s=5)
    # ax[i].scatter(indexes, x[:, i], s=5)
    # ax[i].scatter(indexes, y, s=5)
    ax[i].set_title(round(corr_matrix.values[i, -1], 3))

fig.tight_layout()
fig.show()
