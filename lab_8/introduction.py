import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Input, Model
from keras.datasets import mnist
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, concatenate
from keras.layers import Lambda
from keras.utils.vis_utils import plot_model


def ReLOGU(tensor):
    mask = tensor >= 1
    tensor = tf.where(mask, tensor, 1)
    tensor = tf.math.log(tensor)
    return tensor


def add_inception_module(input_tensor):
    act_func = 'relu'
    paths = [
        [Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation=act_func)],
        [Conv2D(filters=96, kernel_size=(1, 1), padding='same', activation=act_func),
         Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation=act_func)],
        [Conv2D(filters=16, kernel_size=(1, 1), padding='same', activation=act_func),
         Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation=act_func)],
        [MaxPooling2D(pool_size=(3, 3), strides=1, padding='same'),
         Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation=act_func)]
    ]
    for_concat = []
    for path in paths:
        tmp_sensor = input_tensor
        for layer in path:
            tmp_sensor = layer(tmp_sensor)
        for_concat.append(tmp_sensor)

    return concatenate(for_concat)


# wczytanie danych
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# dodanie 4 wymiaru
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# one-hot encoding
y_train = pd.get_dummies(pd.Categorical(y_train)).values
y_test = pd.get_dummies(pd.Categorical(y_test)).values

# określenie wartości hiperparametrów
filter_cnt = 32
kernel_size = (3, 3)
act_func = 'selu'
class_cnt = y_train.shape[1]
epochs_cnt = 2

# ========================================== SIEĆ O LINIOWEJ STRUKTURZE ==========================================

# ========================= ZASTOSOWANIE INTERFEJSU FUNKCYJNEGO =========================
# utworzenie warstw
output_tensor = input_tensor = Input(x_train.shape[1:])
output_tensor = Conv2D(filter_cnt, kernel_size, activation=act_func)(output_tensor)
output_tensor = MaxPooling2D(2, 2)(output_tensor)
output_tensor = Conv2D(filter_cnt, kernel_size, activation=act_func)(output_tensor)
output_tensor = MaxPooling2D(2, 2)(output_tensor)
output_tensor = Conv2D(filter_cnt, kernel_size, activation=act_func)(output_tensor)
output_tensor = GlobalAveragePooling2D()(output_tensor)
output_tensor = Dense(class_cnt, activation='softmax')(output_tensor)
# =======================================================================================

# ========== ZASTOSOWANIE INTERFEJSU FUNKCYJNEGO Z WARSTWAMI ZEBRANYMI DO LISTY ==========
# utworzenie warstw
layers = [Conv2D(filter_cnt, kernel_size, activation=act_func),
          MaxPooling2D(2, 2),
          Conv2D(filter_cnt, kernel_size, activation=act_func),
          MaxPooling2D(2, 2),
          Conv2D(filter_cnt, kernel_size, activation=act_func),
          GlobalAveragePooling2D(),
          Dense(class_cnt, activation='softmax')]

input_tensor = Input(x_train.shape[1:])
output_tensor = input_tensor
for layer in layers:
    output_tensor = layer(output_tensor)

# utworzenie modelu
ANN = Model(inputs=input_tensor, outputs=output_tensor)

# kompilacja modelu
ANN.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
# ========================================================================================

# ======================================== SIEĆ O ROZGAŁĘZIONEJ STRUKTURZE ========================================
# utworzenie warstw
input_tensor = Input(x_train.shape[1:])
output_tensor = input_tensor
inception_module_cnt = 2
for i in range(inception_module_cnt):
    output_tensor = add_inception_module(output_tensor)
output_tensor = GlobalAveragePooling2D()(output_tensor)
output_tensor = Dense(class_cnt, activation='softmax')(output_tensor)

# utworzenie modelu
ANN = Model(inputs=input_tensor, outputs=output_tensor)

# kompilacja modelu
ANN.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

# generacja schematu sieci
plot_model(ANN, to_file='introduction_model.png', show_shapes=True)

# ================================================ WARSTWY LAMBDA ================================================
# utworzenie warstw
output_tensor = input_tensor = Input(x_train.shape[1:])
inception_module_cnt = 2
for i in range(inception_module_cnt):
    output_tensor = add_inception_module(output_tensor)
output_tensor = Conv2D(32, (3, 3))(output_tensor)
output_tensor = Lambda(ReLOGU)(output_tensor)
output_tensor = GlobalAveragePooling2D()(output_tensor)
output_tensor = Dense(class_cnt, activation='softmax')(output_tensor)

# utworzenie modelu
ANN = Model(inputs=input_tensor, outputs=output_tensor)

# kompilacja modelu
ANN.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

# # uczenie modelu
# ANN.fit(x_train, y_train, epochs=epochs_cnt, validation_data=(x_test, y_test), verbose=2)
#
# # predykcja
# y_pred = ANN.predict(x_test)
# y_pred_single = y_pred.argmax(axis=1)
# y_pred = pd.get_dummies(pd.Categorical(y_pred_single)).values
#
# # określenie jakości modelu
# score_f1 = round(f1_score(y_test, y_pred, average='weighted'), 4)
# print('F1 score:', score_f1)
