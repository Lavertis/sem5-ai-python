import numpy as np
import pandas as pd
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


def change_qualitative_feature_of_2_types_to_01(data_frame, column_name, feat_0_name):
    mask = data_frame[column_name].values == feat_0_name
    data_frame[column_name].values[mask] = 1
    data_frame[column_name].values[~mask] = 0
    return data_frame


def load_sklearn_dataset_as_dataframe(dataset):
    data = np.c_[dataset.data, dataset.target]
    columns = np.append(dataset.feature_names, ["target"])
    return pd.DataFrame(data, columns=columns)


def encode_one_hot(a):
    a = pd.Categorical(a)
    a = pd.get_dummies(a).values
    return a


def change_qualitative_feature_of_more_than_2_types_to_numbers(data_frame, column_name):
    cat_feature = pd.Categorical(data_frame[column_name])
    one_hot = pd.get_dummies(cat_feature)
    data_frame = pd.concat([data_frame, one_hot], axis=1)
    data_frame = data_frame.drop(columns=[column_name])
    return data_frame


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


df = load_sklearn_dataset_as_dataframe(fetch_openml("titanic", version=1, as_frame=True))

df = df.drop(['name', 'cabin', 'home.dest', 'ticket', 'boat', 'body'], axis=1)

df = change_qualitative_feature_of_2_types_to_01(df, 'sex', 'female')
df = change_qualitative_feature_of_more_than_2_types_to_numbers(df, 'embarked')
df = change_qualitative_feature_of_more_than_2_types_to_numbers(df, 'parch')
df = change_qualitative_feature_of_more_than_2_types_to_numbers(df, 'sibsp')
df = change_qualitative_feature_of_more_than_2_types_to_numbers(df, 'pclass')

df = df.apply(pd.to_numeric, errors='coerce')
df = df.dropna()

x = df.values[:, :-1]
y = df.values[:, -1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)

input_num = x.shape[1]
class_num = 1

neuron_num = 64
epoch_cnt = 25

model = Sequential()
model.add(Dense(neuron_num, activation='relu', input_dim=input_num))

model.add(Dense(class_num, activation='sigmoid'))

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=16, epochs=epoch_cnt,
          validation_data=(x_test, y_test), verbose=2)

y_pred = model.predict(x_test)

y_pred[y_pred >= 0.5] = 1
y_pred[y_pred < 0.5] = 0

score = f1_score(y_test, y_pred)
print(score)

plot_learning_history(model, epoch_cnt)
