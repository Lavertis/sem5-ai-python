import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GaussianNoise
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


def encode_one_hot(a):
    a = pd.Categorical(a)
    a = pd.get_dummies(a).values
    return a


def scale_data(scaler, x_train_, x_test_):
    scaler.fit(x_train_)
    x_train_ = scaler.transform(x_train_)
    x_test_ = scaler.transform(x_test_)
    return x_train_, x_test_


def plot_learning_history(model_, epoch_cnt_):
    history = model_.history.history
    floss_train = history['loss']
    floss_test = history['val_loss']
    acc_train = history['accuracy']
    acc_test = history['val_accuracy']
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    epochs = np.arange(0, epoch_cnt_)
    ax[0].plot(epochs, floss_train, label='floss_train')
    ax[0].plot(epochs, floss_test, label='floss_test')
    ax[0].set_title('Funkcje strat')
    ax[0].legend()
    ax[1].set_title('Dokładności')
    ax[1].plot(epochs, acc_train, label='acc_train')
    ax[1].plot(epochs, acc_test, label='acc_test')
    ax[1].legend()
    plt.show()


x, y = load_iris(return_X_y=True)

# y one hot encoding
y = encode_one_hot(y)

# # określenie wielkości wejścia i wyjścia
input_num = x.shape[1]
class_num = y.shape[1]

# podział zbioru na testowy i treningowy
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=69, shuffle=True)

# normalizacja danych — skalowanie
x_train, x_test = scale_data(StandardScaler(), x_train, x_test)

# określenie hiperparametrów
neuron_num = 64
noise = [0, 0.1, 0.2, 0.3]
learning_rate = 0.001
epoch_cnt = 250

# określenie bloków warstw i ich argumentów
block = [Dense, GaussianNoise]
args = [(neuron_num, 'selu'), (noise[2],)]

# utworzenie modelu
model = Sequential()
model.add(Dense(neuron_num, activation='relu', input_dim=input_num))

# dodanie kilku bloków wcześniej zdefiniowanych warstw razem z argumentami
repeat_num = 2
for i in range(repeat_num):
    for layer, arg in zip(block, args):
        model.add(layer(*arg))

# dodanie warstwy wyjściowej
model.add(Dense(class_num, activation='sigmoid'))

# kompilacja modelu
model.compile(optimizer=Adam(learning_rate),
              loss='binary_crossentropy',
              metrics=['accuracy', 'Recall', 'Precision'])

# uczenie modelu
model.fit(x_train, y_train, batch_size=15, epochs=epoch_cnt,
          validation_data=(x_test, y_test), verbose=2)

# predykcja
y_pred = model.predict(x_test)
y_pred[y_pred >= 0.5] = 1
y_pred[y_pred < 0.5] = 0

# metryka ogólna
y_test_single = y_test.argmax(axis=1)
y_pred_single = y_pred.argmax(axis=1)
score_f1 = f1_score(y_test_single, y_pred_single, average='weighted')
print('F1 score:', round(score_f1, 4))

# wykres historii uczenia
plot_learning_history(model, epoch_cnt)
