import pandas as pd

# data_excel = pd.read_excel("dane_w_ekselu.xlsx")

# jakikolwiek rodzaj pliku - trzeba tylko podać odpowiedni separator
data = pd.read_csv("practice_lab_1.csv", sep=';')
column_names = list(data.columns)
column_values = data.values

# wybranie danych kolumn z dataframe
specific_columns = data[["kolumna 1", "kolumna 2"]]
