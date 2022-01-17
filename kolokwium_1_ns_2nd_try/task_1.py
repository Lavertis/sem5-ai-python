import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import load_diabetes

data = load_diabetes(as_frame=True)

x_df = data.data
x_df = x_df[['sex', 'age', 's1', 's2', 's3']]
columns = list(x_df.columns)

y_df = pd.DataFrame({'target': data.target})
concatenated = pd.concat([x_df, y_df], axis=1)

corr_matrix = concatenated.corr()
corr_matrix = corr_matrix.drop('target', axis=1)

fig = plt.figure(figsize=(10, 7))
# creating the bar plot
plt.bar(columns, corr_matrix.values[-1, :], color='maroon')
plt.xlabel("Cechy niezależne")
plt.ylabel("Współczynik korelacji")
plt.title("Korelacje w formie wykresu kolumnowego")
plt.show()
