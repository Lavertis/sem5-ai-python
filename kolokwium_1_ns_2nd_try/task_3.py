import numpy as np
import pandas as pd
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler


def create_model_1(input_dim_, class_num_):
    model = Sequential()

    model.add(Dense(64, input_dim=input_dim_))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(class_num_, activation='softmax'))

    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['categorical_accuracy'])
    return model


def create_model_2(input_dim_, class_num_):
    model = Sequential()

    model.add(Dense(128, input_dim=input_dim_))
    model.add(Dense(128, activation='selu'))
    model.add(Dense(128, activation='selu'))
    model.add(Dense(128, activation='selu'))
    model.add(Dense(class_num_, activation='softmax'))

    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['categorical_accuracy'])
    return model


def create_model_3(input_dim_, class_num_):
    model = Sequential()

    model.add(Dense(128, input_dim=input_dim_))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='selu'))
    model.add(Dense(class_num_, activation='softmax'))

    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['categorical_accuracy'])
    return model


def test_three_models(x_train_, y_train_):
    # określenie wielkości wejścia i wyjścia
    input_num = x_train_.shape[1]
    class_num = y.shape[1]

    # tworzenie modelów
    model_1 = create_model_1(input_num, class_num)
    model_2 = create_model_2(input_num, class_num)
    model_3 = create_model_3(input_num, class_num)

    models = [model_1, model_2, model_3]
    accs = []
    for idx, model in enumerate(models):
        accs += (f'Model{idx + 1}', perform_cross_validation(model, x_train_, y_train_))

    print(accs)


def perform_cross_validation(model_, x_train_, y_train_):
    accs_ = []
    epoch_cnt_ = 30
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


data = pd.read_csv('../kolokwium_1_ns/accent.csv', sep=',')

y = data.values[:, 0]
x = data.values[:, 1:]

y = pd.get_dummies(pd.Categorical(y)).values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2, shuffle=True)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

pca_transformer = PCA(0.95)
x_train = pca_transformer.fit_transform(x_train)
x_test = pca_transformer.transform(x_test)

test_three_models(x_train, y_train)
