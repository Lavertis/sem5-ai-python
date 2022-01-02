import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def encode_one_hot(a):
    a = pd.Categorical(a)
    a = pd.get_dummies(a).values
    return a


def scale_data(scaler, x_train_, x_test_):
    scaler.fit(x_train_)
    x_train_ = scaler.transform(x_train_)
    x_test_ = scaler.transform(x_test_)
    return x_train_, x_test_


def create_model(input_num_, class_num_):
    model_ = Sequential()

    model_.add(Dense(64, activation='relu', input_dim=input_num_))
    model_.add(Dense(64, activation='relu'))
    model_.add(Dense(64, activation='relu'))
    model_.add(Dense(64, activation='relu'))
    model_.add(Dense(class_num_, activation='softmax'))

    model_.summary()
    # plot_model(model_, to_file="my_model.png")

    learning_rate = 0.001
    model_.compile(optimizer=Adam(learning_rate),
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])
    return model_


def print_result_metrics(y_test_, y_pred_):
    y_pred_[y_pred_ >= 0.5] = 1
    y_pred_[y_pred_ < 0.5] = 0
    class_num_ = y_test_.shape[1]
    for i in range(class_num_):
        print(f'\n========== {i + 1} class ==========')
        print(confusion_matrix(y_test_[:, i], y_pred_[:, i]))
        print(f'Accuracy score: {round(accuracy_score(y_test_[:, i], y_pred_[:, i]), 3)}')
        print(f'F1 score: {round(f1_score(y_test_[:, i], y_pred_[:, i]), 3)}')


def print_general_result_metrics_for_all_classes(y_test_, y_pred_):
    y_pred_single_hot = y_pred_.argmax(axis=1)
    y_test_single_hot = y_test_.argmax(axis=1)
    score_f1 = f1_score(y_test_single_hot, y_pred_single_hot, average='weighted')
    print('\nGeneral F1 score:', round(score_f1, 3))


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


data = load_digits()
x = data.data
y = data.target

# one hot encoding
y = encode_one_hot(y)

# podział zbioru na treningowy i testowy
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)

# normalizacja danych — skalowanie
x_train, x_test = scale_data(StandardScaler(), x_train, x_test)

# pobranie ilości cech na wejściu i klas na wyjściu
input_num = x.shape[1]
class_num = y.shape[1]

# utworzenie modelu z warstwami dense
model = create_model(input_num, class_num)

# uczenie modelu
epoch_cnt = 25
model.fit(x_train, y_train, batch_size=16, epochs=epoch_cnt,
          validation_data=(x_test, y_test), verbose=2)

# predykcja
y_pred = model.predict(x_test)

# oddzielne metryki dla każdej klasy wyjściowej
print_result_metrics(y_test, y_pred)

# metryka f1_score dla całości
print_general_result_metrics_for_all_classes(y_test, y_pred)

# wykres historii uczenia
plot_learning_history(model, epoch_cnt)
