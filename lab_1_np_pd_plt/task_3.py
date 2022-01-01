import matplotlib.pyplot as plt
import numpy as np


def func_1():
    x = np.arange(-5, 5.001, 0.01)
    y = np.tanh(x)
    plt.plot(x, y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("Function 1")
    plt.show()


def func_2():
    x = np.arange(-5, 5.001, 0.01)
    y = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    plt.plot(x, y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("Function 2")
    plt.show()


def func_3():
    x = np.arange(-5, 5.001, 0.01)
    y = 1 / (1 + np.exp(-x))
    plt.plot(x, y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("Function 3")
    plt.show()


def func_4():
    x1 = np.arange(-5, 0.001, 0.01)
    x2 = np.arange(0, 5.001, 0.01)
    y1 = np.zeros(x1.shape)
    y2 = x2
    plt.plot(x1, y1)
    plt.plot(x2, y2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("Function 4")
    plt.show()


def func_5():
    x1 = np.arange(-5, 0.001, 0.01)
    x2 = np.arange(0, 5.001, 0.01)
    y1 = np.exp(x1) - 1
    y2 = x2
    plt.plot(x1, y1)
    plt.plot(x2, y2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("Function 5")
    plt.show()


func_5()
