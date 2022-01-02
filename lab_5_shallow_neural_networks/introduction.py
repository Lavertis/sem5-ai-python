import numpy as np
import pandas as pd
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils import plot_model
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler


def load_sklearn_dataset_as_dataframe(dataset):
    data = np.c_[dataset.data, dataset.target]
    columns = np.append(dataset.feature_names, ["target"])
    return pd.DataFrame(data, columns=columns)


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
    model_.add(Dense(class_num_, activation='softmax'))

    model_.summary()
    plot_model(model_, to_file="my_model.png")

    learning_rate = 0.0001
    model_.compile(optimizer=Adam(learning_rate),
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])
    return model_


def print_result_metrics(y_test_, y_pred_):
    class_num_ = y_test_.shape[1]
    for i in range(class_num_):
        print(f'\n========== {i + 1} class ==========')
        print(confusion_matrix(y_test_[:, i], y_pred_[:, i]))
        print(f'Accuracy score: {round(accuracy_score(y_test_[:, i], y_pred_[:, i]), 3)}')
        print(f'F1 score: {round(f1_score(y_test_[:, i], y_pred_[:, i]), 3)}')


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


df = load_sklearn_dataset_as_dataframe(load_iris())

# podział na x i y
x = df.values[:, :-1]
y = df.values[:, -1]

# y one-hot encoding
y = encode_one_hot(y)

# określenie wielkości wejścia i wyjścia
input_num = x.shape[1]
class_num = y.shape[1]

# utworzenie modelu
model = create_model(input_num, class_num)

# podzielenie zbioru na treningowy i testowy
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)

# normalizacja danych — skalowanie
x_train, x_test = scale_data(StandardScaler(), x_train, x_test)

# uczenie modelu
epoch_cnt = 150
model.fit(x_train, y_train, batch_size=5, epochs=epoch_cnt,
          validation_data=(x_test, y_test), verbose=2)

# predykcja
y_pred = model.predict(x_test)

# ustalenie, która klasa najbardziej prawdopodobna
y_pred[y_pred >= 0.5] = 1
y_pred[y_pred < 0.5] = 0

# (opcjonalnie) zredukowanie wyniku do 1 wymiarowej tablicy
y_pred_one_dim = y_pred.argmax(axis=1)

# metryki — ocena modelu
print_result_metrics(y_test, y_pred)

# wykres historii uczenia modelu
plot_learning_history(model, epoch_cnt)

# cross validation
accuracies = perform_cross_validation(model)
acc = np.array(accuracies).mean()

print(f'Cross validation accuracy: {round(acc, 4)}')
