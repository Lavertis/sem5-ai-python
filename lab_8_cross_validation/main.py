import numpy as np
import pandas as pd
from keras.layers import Dense
from keras.models import Sequential
from sklearn.decomposition import PCA, FastICA
from sklearn.metrics import multilabel_confusion_matrix  # to jest nowe
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as kNN
from sklearn.tree import DecisionTreeClassifier as DT
from tensorflow.keras.optimizers import Adam


def reduce_pca(data, variance):
    pca = PCA(variance)
    return pca.fit_transform(data)


def reduce_ica(data, components):
    ica = FastICA(n_components=components)
    return ica.fit_transform(data)


def read_data(path: str, separator: str, verbose: int = 0) -> tuple:
    df = pd.read_csv(path, sep=separator)
    x = df.values[:, 1:].astype(np.float64)
    y = df.values[:, 0]

    y = pd.get_dummies(pd.Categorical(y)).values

    if verbose > 0:
        print(df)
        print(type(df.iloc[14, 10]))
        print(type(df.iloc[14, 0]))
        print(x)
        print(x.shape)
        print(y)
        print(y.shape)

    return x, y


def create_model_01(input_count, output_count):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=input_count))
    model.add(Dense(output_count, activation='softmax', input_dim=input_count))
    return model


def train_val_model_01(model, x_train, y_train, x_val, y_val, verbose: int = 0):
    learning_rate = 0.001
    model.compile(
        optimizer=Adam(learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    model.fit(x_train, y_train, validation_data=(x_val, y_val),
              epochs=250, batch_size=15, verbose=verbose)
    return model


def cross_validate(model, x_train, y_train, fold_count: int = 5, verbose: int = 0):
    res = []
    weights = model.get_weights()

    for train_index, val_index in KFold(fold_count).split(x_train):
        x_train_cv = x_train[train_index, :]
        x_val_cv = x_train[val_index, :]
        y_train_cv = y_train[train_index, :]
        y_val_cv = y_train[val_index, :]

        model.set_weights(weights)
        model = train_val_model_01(model, x_train_cv, y_train_cv, x_val_cv, y_val_cv, verbose)

        his_acc = model.history.history['accuracy']
        his_val_acc = model.history.history['val_accuracy']
        his_val_los = model.history.history['val_loss']

        train = [np.max(his_acc),
                 np.argmax(his_acc),
                 np.max(his_val_acc),
                 np.argmax(his_val_acc),
                 np.min(his_val_los),
                 np.argmin(his_val_los)]
        res.append(train)

    return res


if __name__ == "__main__":
    # Wczytanie danych z pliku csv
    path = "accent-mfcc-data-1.csv"
    # x, y = read_data(path, ',', 0)
    x, y = read_data(path, separator=',', verbose=0)

    # Czy nie ma brakujących danych
    # Elementy odstające (Listing 2.6)
    # Skalowanie (Listing 3.5 + zadania)

    # Redukcja wymiarów
    x_pca_09 = reduce_pca(x, 0.9)
    x_pca_08 = reduce_pca(x, 0.8)

    x_ica_7 = reduce_ica(x, 7)  # x_pca_09.shape[1]
    x_ica_4 = reduce_ica(x, 4)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    # x_train, x_test, y_train, y_test = train_test_split(x_pca_08, y, test_size=0.2)

    # Testowanie klasycznych algorytmów
    models = [kNN(), DT(max_depth=9)]
    for model in models:
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        cm = multilabel_confusion_matrix(y_test, y_pred)
        print(model)
        print(cm)

    # Pierwszy model sieci neuronowej
    input_cnt = x_train.shape[1]
    output_cnt = y_train.shape[1]
    model = create_model_01(input_cnt, output_cnt)
    model.summary()

    # nauka
    model = train_val_model_01(model, x_train, y_train, x_test, y_test, verbose=1)

    # informacje o procesie uczenia
    his_acc = model.history.history['accuracy']
    his_val_acc = model.history.history['val_accuracy']
    print(np.max(his_acc))
    print(np.max(his_val_acc))

    # predykcja
    y_pred = model.predict(x_test)
    mask = y_pred > 0.5
    y_pred[mask] = 1
    y_pred[~mask] = 0

    # ocena predykcji
    cm = multilabel_confusion_matrix(y_test, y_pred)
    print(model)
    print(cm)

    # Walidacja krzyżowa
    model = create_model_01(input_cnt, output_cnt)
    res = cross_validate(model, x_train, y_train, fold_count=5, verbose=0)
    for line in res:
        print(line)

    npres = np.array(res)
    print(np.average(npres[:, 2]))

    y_pred = model.predict(x_test)
    mask = y_pred > 0.5
    y_pred[mask] = 1
    y_pred[~mask] = 0
    cm = multilabel_confusion_matrix(y_test, y_pred)
    print(model)
    print(cm)
