import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix
from tensorflow.keras import Sequential
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


def generate_confusion_matrices(y_test_, y_pred_):
    class_num_ = y_test_.shape[1]
    for i in range(class_num_):
        print(f'\n========== {i + 1} class ==========')
        print(confusion_matrix(y_test_[:, i], y_pred_[:, i]))


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


(x_train, y_train), (x_test, y_test) = cifar10.load_data()

y_test = np.squeeze(y_test)
y_train = np.squeeze(y_train)

# określenie wielkości wejścia i wyjścia
input_num = x_train.shape[1:]  # bez sample size, keras dodaje to sam
class_cnt = np.unique(y_train).shape[0]  # wielkość tablicy unikalnych wartości

filter_cnt = 32
neuron_cnt = 64
learning_rate = 0.001
act_func = 'relu'
kernel_size = (3, 3)
pooling_size = (2, 2)
conv_rule = 'same'
epochs_cnt = 20

# utworzenie modelu
model = Sequential()

# dodanie warstw do modelu
model.add(Conv2D(input_shape=input_num,
                 filters=filter_cnt,
                 kernel_size=kernel_size,
                 padding=conv_rule, activation=act_func))

model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3), activation=act_func, kernel_initializer='he_uniform'))

model.add(Dropout(0.3))

model.add(Conv2D(64, (3, 3), activation=act_func, kernel_initializer='he_uniform'))

model.add(AveragePooling2D((2, 2)))

model.add(Conv2D(64, (3, 3), activation=act_func, kernel_initializer='he_uniform'))

model.add(Conv2D(64, (3, 3), activation=act_func, kernel_initializer='he_uniform'))

model.add(MaxPooling2D((2, 2)))

model.add(Flatten())

model.add(Dropout(0.3))

model.add(Dense(256, activation=act_func, kernel_initializer='he_uniform'))

model.add(Dense(neuron_cnt, kernel_regularizer=l2(0.01)))

model.add(Dense(128, activation=act_func, kernel_initializer='he_uniform'))

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

y_pred_single = y_pred.argmax(axis=1)
y_pred = pd.get_dummies(pd.Categorical(y_pred_single)).values
y_test = pd.get_dummies(pd.Categorical(y_test)).values

# confusion matrices
generate_confusion_matrices(y_test, y_pred)

# f1 score
score_f1 = round(f1_score(y_test, y_pred, average='weighted'), 3)
print('F1 score:', score_f1)

# podsumowanie modelu
model.summary()

# wykres historii uczenia
plot_learning_history(model, epochs_cnt)
