import numpy as np
import pandas as pd
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from matplotlib import pyplot as plt
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler


def encode_one_hot(a):
    a = pd.Categorical(a)
    a = pd.get_dummies(a).values
    return a


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


def perform_cross_validation(model_):
    x_train_, x_test_, y_train_, y_test_ = train_test_split(x, y, test_size=0.2, shuffle=True)
    accs_ = []
    scaler = StandardScaler()
    epoch_cnt_ = 15
    weights = model_.get_weights()

    for train_index, test_index in KFold(5).split(x_train_):
        x_train_cv = x_train_[train_index, :]
        x_test_cv = x_train_[test_index, :]
        y_train_cv = y_train_[train_index, :]
        y_test_cv = y_train_[test_index, :]

        # normalizacja danych — skalowanie
        x_train_cv = scaler.fit_transform(x_train_cv)
        x_test_cv = scaler.transform(x_test_cv)

        # uczenie modelu
        model_.set_weights(weights)
        model_.fit(x_train_cv, y_train_cv, batch_size=16, epochs=epoch_cnt_,
                   validation_data=(x_test_cv, y_test_cv), verbose=2)

        # predykcja
        y_pred_ = model_.predict(x_test_cv).argmax(axis=1)
        y_test_cv = y_test_cv.argmax(axis=1)
        accs_.append(accuracy_score(y_test_cv, y_pred_))
    plot_learning_history(model_, epoch_cnt_)
    return accs_


data = load_digits()
x = data.data
y = data.target

# one hot encoding
y = encode_one_hot(y)

# pobranie ilości cech na wejściu i klas na wyjściu
input_num = x.shape[1]
class_num = y.shape[1]

# utworzenie modelu
model = create_model(input_num, class_num)
accs = perform_cross_validation(model)
acc_mean = np.array(accs).mean()
print("\nMean accuracy:", round(acc_mean, 4))
