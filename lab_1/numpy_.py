import numpy as np

# odwoływanie się do elementów w tablicy numpy ndarray
arr = np.array([[2, 3, 5, 1],
                [5, 1, 2, 8],
                [5, 1, 6, -1]])
pojedynczy_element = arr[0, 3]  # x, y

# z minusem numerujemy od 1 a nie od 0
pojedynczy_element_od_konca = arr[-3, -1]  # x, y
