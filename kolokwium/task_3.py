import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, GaussianNoise
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1, l2


def encode_one_hot(a):
    a = pd.Categorical(a)
    a = pd.get_dummies(a).values
    return a


def replace_outliers_with_mean(x_train_, value):
    x_train_mean = x_train_.copy()
    for i in range(x_train_.shape[1]):
        mean = x_train_[:, i].mean()
        outliers = np.abs((x_train_[:, i] - mean) / x_train_[:, i].std()) > value
        x_train_mean[:, i][outliers] = mean
        print(outliers.sum())

    return x_train_mean


def scale_data(scaler, x_train_, x_test_):
    scaler.fit(x_train_)
    x_train_ = scaler.transform(x_train_)
    x_test_ = scaler.transform(x_test_)
    return x_train_, x_test_


def create_model_01(input_num_, class_num_):
    model_ = Sequential()
    neuron_num = 64

    model_.add(Dense(neuron_num, activation='relu', input_dim=input_num_))
    model_.add(Dense(neuron_num, activation='relu'))
    model_.add(Dense(neuron_num, activation='relu'))
    model_.add(Dense(class_num_, activation='softmax'))

    model_.summary()
    # plot_model(model_, to_file="my_model.png")

    learning_rate = 0.001
    model_.compile(optimizer=Adam(learning_rate),
                   loss='categorical_crossentropy',
                   metrics=['categorical_accuracy'])
    return model_


def create_model_02(input_num_, class_num_):
    model_ = Sequential()
    neuron_num = 64

    model_.add(Dense(neuron_num, activation='relu', input_dim=input_num_))
    model_.add(Dropout(0.5))
    model_.add(GaussianNoise(0.1))
    model_.add(Dense(neuron_num, activation='relu', kernel_regularizer=l2(0.01)))
    model_.add(GaussianNoise(0.1))
    model_.add(Dense(neuron_num, activation='relu'))
    model_.add(Dense(class_num_, activation='softmax'))

    model_.summary()
    # plot_model(model_, to_file="my_model.png")

    learning_rate = 0.001
    model_.compile(optimizer=Adam(learning_rate),
                   loss='categorical_crossentropy',
                   metrics=['categorical_accuracy'])
    return model_


def create_model_03(input_num_, class_num_):
    model_ = Sequential()
    neuron_num = 128

    model_.add(Dense(neuron_num, activation='relu', input_dim=input_num_))
    model_.add(Dense(neuron_num, activation='relu', kernel_regularizer=l1(0.1)))
    model_.add(GaussianNoise(0.1))
    model_.add(Dense(neuron_num, activation='relu'))
    model_.add(Dense(class_num_, activation='softmax'))

    model_.summary()
    # plot_model(model_, to_file="my_model.png")

    learning_rate = 0.001
    model_.compile(optimizer=Adam(learning_rate),
                   loss='categorical_crossentropy',
                   metrics=['categorical_accuracy'])
    return model_


def test_three_models(x_train_, y_train_):
    # określenie wielkości wejścia i wyjścia
    input_num = x_train_.shape[1]
    class_num = y_train_.shape[1]

    # tworzenie modelów
    model_01 = create_model_01(input_num, class_num)
    model_02 = create_model_02(input_num, class_num)
    model_03 = create_model_03(input_num, class_num)

    models = [model_01, model_02, model_03]
    accs = []
    for idx, model in enumerate(models):
        accs += (f'Model{idx + 1}', perform_cross_validation(model, x_train_, y_train_))

    print(accs)


def perform_cross_validation(model_, x_train_, y_train_):
    accs_ = []
    epoch_cnt_ = 10
    weights = model_.get_weights()

    for train_index, test_index in KFold(5).split(x_train_):
        x_train_cv = x_train_[train_index, :]
        x_test_cv = x_train_[test_index, :]
        y_train_cv = y_train_[train_index, :]
        y_test_cv = y_train_[test_index, :]

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
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2, shuffle=True)

# zamiana outlierów na średnią
x_train = replace_outliers_with_mean(x_train, 3)

# skalowanie
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

test_three_models(x_train, y_train)
# Model 1 jest najlepszy
