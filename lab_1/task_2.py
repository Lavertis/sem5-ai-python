import numpy as np
import pandas as pd

data = pd.read_csv('practice_lab_1.csv', sep=';')
values = data.values
columns = np.array(data.columns)

# ============================== SUBTASK 1 ==============================
# Znajdź tablicę dwuwymiarową, która będzie wynikiem różnicy dwóch tablic: pierwsza
# będzie zawierała wszystkie kolumny oraz parzyste wiersze tablicy danych
# wejściowych, druga – wszystkie kolumny i nieparzyste wiersze tablicy danych
# wejściowych. Numeracja wierszy oraz kolumn zaczyna się od 0!
arr1 = values[::2, :]
arr2 = values[1::2, :]
task_1_res = arr1 - arr2

# ============================== SUBTASK 2 ==============================
# Przekształć dane w sposób następujący: od każdej wartości odejmij średnią obliczoną
# dla całej tablicy oraz podziel przez odchylenie standardowe wyznaczone dla całej
# tablicy. Podpowiedź: skorzystaj z metod mean oraz std.
task_2_res = (values - values.mean()) / (values.std() + np.spacing(values))

# ============================== SUBTASK 3 ==============================
# Wykonaj zadanie z podpunktu 2 dla oddzielnych kolumn pierwotnej tablicy, czyli
# wyznaczając średnią oraz odchylenie standardowe dla oddzielnych kolumn.
# Podpowiedź: aby uniknąć dzielenia przez zero, do wyniku wyrażenia arr.std(axis=0)
# dodaj wynik funkcji np.spacing(arr.std(axis=0)).
task_3_res = (values - values.mean(axis=0)) / (values.std(axis=0) + np.spacing(values.std(axis=0)))

# ============================== SUBTASK 4 ==============================
# Dla każdej kolumny pierwotnej tablicy policz współczynnik zmienności, definiowany
# jako stosunek odchylenia standardowego do średniej, zabezpiecz się przed dzieleniem
# przez 0 podobnie do poprzedniego punktu.
task_4_res = values.std(axis=0) / (values.mean(axis=0) + np.spacing(values.mean(axis=0)))

# ============================== SUBTASK 5 ==============================
# Znajdź kolumnę o największym współczynniku zmienności.
task_5_res = np.argmax(task_4_res)

# ============================== SUBTASK 6 ==============================
# Dla każdej kolumny pierwotnej tablicy policz liczbę elementów o wartości większej,
# niż średnia tej kolumny.
task_6_res = np.count_nonzero(values > values.mean(axis=0))

# ============================== SUBTASK 7 ==============================
# Znajdź nazwy kolumn w których znajduje się wartość maksymalna. Podpowiedź: listę
# stringów można również przekształcić na tablicę numpy, po czym można będzie dla niej
# zastosować maskę.
task_7_res = columns[values.max(axis=0) == values.max()]

# ============================== SUBTASK 8 ==============================
# Znajdź nazwy kolumn w których jest najwięcej elementów o wartości 0. Podpowiedź:
# wartości w tablicy wartości logicznych można sumować, zakładając, że zawiera ona
# liczby całkowite, rzutowanie będzie wykonane automatycznie.
task_8_res = np.count_nonzero(values == 0)

# ============================== SUBTASK 9 ==============================
# Znajdź nazwy kolumn w których suma elementów na pozycjach parzystych jest większa
# od sumy elementów na pozycjach nieparzystych. Wyświetl ich nazwy, postaraj się
# nie korzystać z pętli.
task_9_res = columns[np.sum(values[::2, :], axis=0) > np.sum(values[1::2, :], axis=0)]
