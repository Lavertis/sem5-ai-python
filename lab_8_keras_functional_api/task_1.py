import pandas as pd
from keras import Input, Model
from keras.datasets import mnist
from keras.layers import Dense, Reshape, BatchNormalization, Average
from keras.utils import plot_model

# wczytanie danych
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# dodanie 4 wymiaru
# x_train = np.expand_dims(x_train, axis=-1)
# x_test = np.expand_dims(x_test, axis=-1)

# one-hot encoding
y_train = pd.get_dummies(pd.Categorical(y_train)).values
y_test = pd.get_dummies(pd.Categorical(y_test)).values

# utworzenie warstw
output_tensor = input_tensor = Input(x_train.shape[1:])
output_tensor = Reshape((784,))(output_tensor)
output_tensor = BatchNormalization()(output_tensor)

paths = [
    [Dense(512), Dense(128), Dense(64), Dense(16), Dense(10)],
    [Dense(512), Dense(64), Dense(10)],
    [Dense(512), Dense(64), Dense(10)],
    [Dense(512), Dense(64), Dense(10)],
    [Dense(512), Dense(64), Dense(10)],
]

for_avg = []
for path in paths:
    tmp_sensor = output_tensor
    for layer in path:
        tmp_sensor = layer(tmp_sensor)
    for_avg.append(tmp_sensor)

output_tensor = Average()(for_avg)

# utworzenie modelu
ANN = Model(inputs=input_tensor, outputs=output_tensor)

# kompilacja modelu
ANN.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

# generacja schematu sieci
plot_model(ANN, to_file='task_1_model.png', show_shapes=True)
