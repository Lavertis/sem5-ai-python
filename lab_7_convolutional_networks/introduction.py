import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


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


# wczytanie danych
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# dodanie kanału koloru (1 wymiaru) do x_train i x_test
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# określenie wielkości wejścia i wyjścia
input_num = x_train.shape[1:]  # bez sample size, keras dodaje to sam
class_cnt = np.unique(y_train).shape[0]  # wielkość tablicy unikalnych wartości

filter_cnt = 32
neuron_cnt = 32
learning_rate = 0.0001
act_func = 'relu'
kernel_size = (3, 3)
pooling_size = (2, 2)
conv_rule = 'same'
epochs_cnt = 10

# utworzenie modelu
model = Sequential()

# dodanie warstw do modelu
model.add(Conv2D(input_shape=input_num,
                 filters=filter_cnt,
                 kernel_size=kernel_size,
                 padding=conv_rule, activation=act_func))
model.add(MaxPooling2D(pool_size=pooling_size))
model.add(Flatten())
model.add(Dense(class_cnt, activation='softmax'))

# kompilacja modelu
model.compile(optimizer=Adam(learning_rate),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# uczenie modelu
model.fit(x_train, y_train, epochs=epochs_cnt,
          validation_data=(x_test, y_test), verbose=2)

# predykcja
y_pred = model.predict(x_test)
y_pred = y_pred.argmax(axis=1)

# f1 score
score = round(f1_score(y_test, y_pred, average='weighted'), 3)

# podsumowanie modelu
model.summary()

# wykres historii uczenia
plot_learning_history(model, class_cnt)
