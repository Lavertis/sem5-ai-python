import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import load_diabetes


def load_sklearn_dataset_as_dataframe(dataset):
    data = np.c_[dataset.data, dataset.target]
    columns = np.append(dataset.feature_names, ["target"])
    return pd.DataFrame(data, columns=columns)


df = load_sklearn_dataset_as_dataframe(load_diabetes())
df = df[['sex', 'age', 's1', 's2', 's3', 'target']]

corr_matrix = df.corr()
corr_matrix = corr_matrix.drop(columns='target')

fig, ax = plt.subplots(1, 1)
indexes = np.arange(len(corr_matrix.columns))

ax.bar(indexes, corr_matrix.values[-1, :])
ax.set_xticks(indexes)
ax.set_xticklabels(corr_matrix.columns, rotation=0)
ax.set_title('Wykres korelacji')
ax.set_ylabel('współczynniki korelacji')
ax.set_xlabel('cechy')
fig.tight_layout()
plt.show()
