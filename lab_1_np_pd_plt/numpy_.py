import numpy as np

# ==================== ODWOŁYWANIE SIĘ DO ELEMENTÓW W TABLICY NUMPY ====================
arr = np.array([[2, 3, 5, 1],
                [5, 1, 2, 8],
                [5, 1, 6, -1]])

single_element = arr[0, 3]  # [wiersz, kolumna]
single_element_from_end = arr[-3, -1]  # z minusem numerujemy od 1 a nie od 0
column = arr[:, 2]  # pobranie całej kolumny
row = arr[2, :]  # pobranie całego wiersza

# pobranie zakresu kolumn z krokiem
res = arr[:, 1:3:1]  # [pierwsza_kolumna(włącznie): ostatnia_kolumna(wyłącznie) : krok]
# pobranie zakresu wierszy z krokiem
res = arr[0:2:1, :]  # [pierwszy_wiersz(włącznie): ostatni_wiersz(wyłącznie) : krok]
# pobranie części całej tablicy z wyborem wierszy i kolumn
res = arr[0:2, 1:3]

res = arr[:, [1, 2, 3]]  # pobranie wybranych kolumn z tablicy
res = arr[[0, 1, 2], :]  # pobranie wybranych wierszy z tablicy

mask = arr < 2  # utworzenie maski
res = arr[mask]  # pobranie danych gdzie true w masce jako 1 wymiarowa tablica
res = arr[~mask]  # pobranie danych gdzie false w masce jako 1 wymiarowa tablica

# ==================== OPERACJE NA TABLICACH NUMPY ====================
arr1 = np.array([1, 2, 3, 4, 5, 6])
arr2 = np.array([7, 8, 9, 10, 11, 12])

# TABLICA <OPERACJA> TABLICA
res = arr1 + arr2  # dodawanie
res = arr1 - arr2  # odejmowanie
res = arr1 * arr2  # mnożenie
res = arr1 / arr2  # dzielenie

# TABLICA <OPERACJA> LICZBA
res = arr1 + 2  # dodawanie
res = arr1 - 2  # odejmowanie
res = arr1 * 2  # mnożenie
res = arr1 / 2  # dzielenie
res = arr1 ** 3  # podnoszenie do potęgi

# ==================== SUMA/ILOCZYN/ŚREDNIA ====================
res = arr.sum()  # suma elementów w tablicy
res = arr.prod()  # iloczyn elementów w tablicy
res = arr.mean()  # średnia elementów w tablicy
res = arr.std()  # odchylenie standardowe elementów w tablicy

# [axis_0, axis_1]
sum_in_each_column = arr.sum(axis=0)  # suma w każdej kolumnie
prod_in_each_column = arr.prod(axis=0)  # iloczyn w każdej kolumnie
mean_in_each_column = arr.mean(axis=0)  # średnia w każdej kolumnie
std_in_each_column = arr.std(axis=0)  # odchylenie standardowe w każdej kolumnie

sum_in_each_row = arr.sum(axis=1)  # suma w każdym wierszu
prod_in_each_row = arr.prod(axis=1)  # iloczyn w każdym wierszu
mean_in_each_row = arr.mean(axis=1)  # średnia w każdym wierszu
std_in_each_row = arr.std(axis=1)  # odchylenie standardowe w każdym wierszu

# ==================== ATRYBUTY TABLICY ====================
res = arr.ndim  # liczba wymiarów tablicy
res = arr.shape[0]  # rozmiar danego wymiaru
res = arr.T  # macierz transponowana
arr.sort(axis=0)  # posortowanie elementów w kolumnach
arr.sort(axis=1)  # posortowanie elementów w wierszach

# Reverse the sorted 1D array
# [wiersze, kolumny]
reverse_array = arr[::-1, ::-1]  # odwrócone wiersze i kolumny
reverse_array = arr[::1, ::-1]  # odwrócone kolumny
reverse_array = arr[::-1, ::1]  # odwrócone wiersze
