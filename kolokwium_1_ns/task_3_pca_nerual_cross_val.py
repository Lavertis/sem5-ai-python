import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GaussianNoise
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1
from tensorflow.keras.regularizers import l2


def encode_one_hot(a):
    a = pd.Categorical(a)
    a = pd.get_dummies(a).values
    return a


def replace_outliers_with_mean(y_train_, value):
    outliers = np.abs((y_train_ - y_train_.mean()) / y_train_.std()) > value
    y_train_mean = y_train_.copy()
    y_train_mean[outliers] = y_train_.mean()
    return y_train_mean


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


def create_model_01(input_num_, class_num_):
    model_ = Sequential()
    neuron_num = 64

    model_.add(Dense(neuron_num, activation='relu', input_dim=input_num_))
    model_.add(Dense(neuron_num, activation='relu'))
    model_.add(Dense(neuron_num, activation='relu'))
    model_.add(Dense(class_num_, activation='softmax'))

    model_.summary()
    # plot_model(model_, to_file="my_model.png")

    learning_rate = 0.0001
    model_.compile(optimizer=Adam(learning_rate),
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])
    return model_


def create_model_02(input_num_, class_num_):
    model_ = Sequential()
    neuron_num = 128

    model_.add(Dense(neuron_num, activation='relu', input_dim=input_num_))
    model_.add(Dropout(0.5))
    model_.add(GaussianNoise(0.1))
    model_.add(Dense(neuron_num, activation='relu', kernel_regularizer=l2(0.01)))
    model_.add(GaussianNoise(0.1))
    model_.add(Dense(neuron_num, activation='relu'))
    model_.add(Dense(class_num_, activation='softmax'))

    model_.summary()
    # plot_model(model_, to_file="my_model.png")

    learning_rate = 0.0001
    model_.compile(optimizer=Adam(learning_rate),
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])
    return model_


def create_model_03(input_num_, class_num_):
    model_ = Sequential()
    neuron_num = 256

    model_.add(Dense(neuron_num, activation='relu', input_dim=input_num_))
    model_.add(Dense(neuron_num, activation='relu', kernel_regularizer=l1(0.1)))
    model_.add(GaussianNoise(0.1))
    model_.add(Dense(neuron_num, activation='relu'))
    model_.add(Dense(class_num_, activation='softmax'))

    model_.summary()
    # plot_model(model_, to_file="my_model.png")

    learning_rate = 0.0001
    model_.compile(optimizer=Adam(learning_rate),
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])
    return model_


def test_three_models(x_train_, y_train_):
    # określenie wielkości wejścia i wyjścia
    input_num = x_train_.shape[1]
    class_num = y.shape[1]

    # tworzenie modelów
    model_01 = create_model_01(input_num, class_num)
    model_02 = create_model_02(input_num, class_num)
    model_03 = create_model_03(input_num, class_num)

    models = [model_01, model_02, model_03]
    accs = []
    for idx, model in enumerate(models):
        # uczenie modelu
        model.fit(x_train_, y_train_, batch_size=5, epochs=50, verbose=2)

        accs += (f'Model{idx + 1}', perform_cross_validation(model, x_train_, y_train_))

    print(accs)


def perform_cross_validation(model_, x_train_, y_train_):
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
        # x_train_cv = scaler.fit_transform(x_train_cv)
        # x_test_cv = scaler.transform(x_test_cv)

        # uczenie modelu
        model_.set_weights(weights)
        model_.fit(x_train_cv, y_train_cv, batch_size=16, epochs=epoch_cnt_,
                   validation_data=(x_test_cv, y_test_cv), verbose=2)

        # predykcja
        y_pred_ = model_.predict(x_test_cv).argmax(axis=1)
        y_test_cv = y_test_cv.argmax(axis=1)
        accs_.append(accuracy_score(y_test_cv, y_pred_))
    # plot_learning_history(model_, epoch_cnt_)
    return round(np.array(accs_).mean(), 4)


# wczytanie danych
data = pd.read_csv('accent.csv', sep=',')

# podział danych na x i y
x = data.values[:, 1:]
y = data.values[:, 0]

# one-hot encoding klasy wynikowej
y = encode_one_hot(y)

# podział zbioru na treningowy i testowy
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)

# zastąpienie wartości odstających średnią
y_train = replace_outliers_with_mean(y_train, 3)

# normalizacja danych — skalowanie
x_train, x_test = scale_data(StandardScaler(), x_train, x_test)

# redukcja wymiarów PCA
pca_transformer = PCA(0.95)
x_train = pca_transformer.fit_transform(x_train)
x_test = pca_transformer.transform(x_test)

test_three_models(x_train, y_train)
