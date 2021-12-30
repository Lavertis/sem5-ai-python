import numpy as np
from matplotlib import pyplot as plt


# ==================== 2 WYKRESY NAŁOŻONE NA SIEBIE ====================
def two_plots_on_top_of_each_other():
    x = np.arange(0, 10, 0.1)  # generacja punktów - start(włącznie), koniec(wyłącznie), krok
    y = np.sin(x ** 2 - 5 * x + 3)  # generacja wartości dla punktów
    plt.scatter(x, y)  # utworzenie wykresu punktowego
    plt.plot(x, y)  # utworzenie wykresu ciągłego
    plt.xlabel('x')  # dodanie opisu osi x
    plt.ylabel('y')  # dodanie opisu osi y
    plt.show()


# ==================== 2 WYKRESY OBOK SIEBIE ====================
def two_plots_next_to_each_other():
    x = np.arange(0, 10, 0.1)  # generacja punktów - start(włącznie), koniec(wyłącznie), krok
    y = np.sin(x ** 2 - 5 * x + 3)  # generacja wartości dla punktów
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].plot(x, y)  # utworzenie wykresu ciągłego
    ax[0].set_xlabel("x")  # dodanie opisu osi x dla pierwszego wykresu
    ax[0].set_ylabel("y")  # dodanie opisu osi y dla pierwszego wykresu
    ax[1].scatter(x, y)  # utworzenie wykresu punktowego
    ax[1].set_xlabel("x")  # dodanie opisu osi x dla drugiego wykresu
    ax[1].set_ylabel("y")  # dodanie opisu osi y dla drugiego wykresu
    fig.tight_layout()  # dobre upakowanie elementów
    plt.show()


def plot_grid():
    x = np.arange(0, 10, 0.1)  # generacja punktów - start(włącznie), koniec(wyłącznie), krok
    y = np.sin(x ** 2 - 5 * x + 3)  # generacja wartości dla punktów
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    ax[0, 0].scatter(x, y)  # utworzenie wykresu punktowego w 0 rzędzie i w 0 kolumnie
    ax[0, 1].plot(x, y)  # utworzenie wykresu ciągłego w 0 rzędzie i 1 kolumnie
    ax[1, 0].hist(y)  # utworzenie wykresu histogramu w 1 rzędzie i 0 kolumnie
    ax[1, 1].boxplot(y)  # utworzenie wykresu pudełkowego w 1 rzędzie i 1 kolumnie
    plt.show()


two_plots_on_top_of_each_other()
two_plots_next_to_each_other()
plot_grid()
