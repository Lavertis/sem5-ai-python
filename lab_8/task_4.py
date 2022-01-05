import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Input, Model
from keras.datasets import mnist
from keras.layers import Dense, Lambda, Flatten, Conv2D, AveragePooling2D
from keras.utils import plot_model
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score


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


# f(x) = x * tanh(ln(1 + e^x))
def Mish(x):
    tanh = tf.math.tanh(tf.math.log(tf.add(tf.exp(x), 1)))
    x = tf.multiply(x, tanh)
    return x


# wczytanie danych
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# dodanie 4 wymiaru
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# one-hot encoding
y_train = pd.get_dummies(pd.Categorical(y_train)).values
y_test = pd.get_dummies(pd.Categorical(y_test)).values

# określenie wartości hiperparametrów
act_func = 'relu'
filter_cnt = 32
kernel_size = (3, 3)
class_cnt = y_train.shape[1]
epochs_cnt = 10
dense_net_cnt = 2

# utworzenie warstw
output_tensor = input_tensor = Input(x_train.shape[1:])

output_tensor = Lambda(Mish)(output_tensor)

output_tensor = Conv2D(filter_cnt, kernel_size, activation=act_func, padding='same', )(output_tensor)
output_tensor = AveragePooling2D(2, 2)(output_tensor)

output_tensor = Lambda(Mish)(output_tensor)

output_tensor = Conv2D(filter_cnt, kernel_size, activation=act_func, padding='same', )(output_tensor)
output_tensor = AveragePooling2D(2, 2)(output_tensor)
output_tensor = Flatten()(output_tensor)
output_tensor = Dense(class_cnt, activation='softmax')(output_tensor)

# utworzenie modelu
ANN = Model(inputs=input_tensor, outputs=output_tensor)

# kompilacja modelu
ANN.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

# generacja schematu sieci
plot_model(ANN, to_file='task_4_model.png', show_shapes=True)

# uczenie modelu
ANN.fit(x_train, y_train, epochs=epochs_cnt, validation_data=(x_test, y_test), verbose=2)

# predykcja
y_pred = ANN.predict(x_test)
y_pred_single = y_pred.argmax(axis=1)
y_pred = pd.get_dummies(pd.Categorical(y_pred_single)).values

# określenie jakości modelu
score_f1 = round(f1_score(y_test, y_pred, average='weighted'), 4)
print('F1 score:', score_f1)

plot_learning_history(ANN, epochs_cnt)
