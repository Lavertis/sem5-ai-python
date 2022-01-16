import numpy as np
from keras.datasets import fashion_mnist
from keras.layers import Conv2D, MaxPool2D, Input, UpSampling2D, GaussianNoise, Dense, GlobalAveragePooling2D, \
    Reshape, BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from matplotlib import pyplot as plt

# wczytanie danych
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# dodanie 4 wymiaru
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# zmniejszenie zbioru treningowego
samples = 60000
x_train = x_train[:samples, :, :]
y_train = y_train[:samples]
# x_test = x_test[:samples, :, :]
# y_test = y_test[:samples]

# skalowanie
x_train_scaled = (x_train / 255).copy()
x_test_scaled = (x_test / 255).copy()

# hiperparametry
act_func = 'selu'
aec_dim_num = 2
lrng_rate = 0.0001

# warstwy enkodera
encoder_layers = [
    GaussianNoise(1),
    BatchNormalization(),
    Conv2D(32, (7, 7), padding='same', activation=act_func),
    MaxPool2D(2, 2),
    BatchNormalization(),
    Conv2D(64, (5, 5), padding='same', activation=act_func),
    MaxPool2D(2, 2),
    BatchNormalization(),
    Conv2D(128, (3, 3), padding='same', activation=act_func),
    GlobalAveragePooling2D(),
    Dense(aec_dim_num, activation='tanh')
]

# warstwy dekodera
decoder_layers = [
    Dense(128, activation=act_func),
    BatchNormalization(),
    Reshape((1, 1, 128)),
    UpSampling2D((7, 7)),
    Conv2D(128, (3, 3), padding='same', activation=act_func),
    BatchNormalization(),
    UpSampling2D((2, 2)),
    Conv2D(64, (5, 5), padding='same', activation=act_func),
    BatchNormalization(),
    UpSampling2D((2, 2)),
    Conv2D(32, (7, 7), padding='same', activation=act_func),
    BatchNormalization(),
    Conv2D(1, (3, 3), padding='same', activation='sigmoid')
]

# utworzenie warstwy wejściowej do auto-enkodera i enkodera
tensor = input_aec = input_encoder = Input(x_train_scaled.shape[1:])

# dodanie warstw do enkodera
for layer in encoder_layers:
    tensor = layer(tensor)

output_encoder = tensor  # warstwa wyjściowa enkodera
dec_tensor = input_decoder = Input(output_encoder.shape[1:])  # warstwa wejściowa dekodera

# dodanie warstw do dekodera
for layer in decoder_layers:
    tensor = layer(tensor)
    dec_tensor = layer(dec_tensor)

output_aec = tensor  # zapisanie warstwy wyjściowej auto-enkodera
output_decoder = dec_tensor  # zapisanie warstwy wyjściowej dekodera

# utworzenie auto-enkodera, enkodera oraz dekodera
autoencoder = Model(inputs=input_aec, outputs=output_aec)
encoder = Model(inputs=input_encoder, outputs=output_encoder)
decoder = Model(inputs=input_decoder, outputs=output_decoder)

# kompilacja i uczenie modelu
autoencoder.compile(optimizer=Adam(lrng_rate), loss='binary_crossentropy', metrics=['mean_squared_error'])
autoencoder.fit(x=x_train_scaled, y=x_train_scaled, epochs=10, batch_size=256,
                validation_data=(x_test_scaled, x_test_scaled), verbose=2)

# wprowadzenie szumu do danych
test_photos = x_train_scaled[10:20, ...].copy()
noisy_test_photos = test_photos.copy()
mask = np.random.randn(*test_photos.shape)
white = mask > 1
black = mask < -1

noisy_test_photos[white] = 1
noisy_test_photos[black] = 0


# funkcja wyświetlająca grupy obrazów
def show_pictures(arrs):
    arr_cnt = arrs.shape[0]
    fig, axes = plt.subplots(1, arr_cnt, figsize=(5 * arr_cnt, arr_cnt))
    for axis, pic in zip(axes, arrs):
        axis.imshow(pic.squeeze(), cmap='gray')

    plt.show()


# wyświetlenie obrazów
cleaned_images = autoencoder.predict(noisy_test_photos) * 255
show_pictures(test_photos)
show_pictures(noisy_test_photos)
show_pictures(cleaned_images)

# reprezentacja zbioru uczącego w przestrzeni parametrów ukrytych
fig, ax = plt.subplots(1, 1, figsize=(20, 16))
for i in range(10):
    digits = y_train == i
    needed_imgs = x_train_scaled[digits, ...]

    preds = encoder.predict(needed_imgs)
    ax.scatter(preds[:, 0], preds[:, 1])

ax.legend(list(range(10)))
plt.show()

# wizualizacja przestrzeni ukrytych parametrów
num = 15
limit = 0.6
step = limit * 2 / num
fig, ax = plt.subplots(num, num, figsize=(20, 16))
X_vals = np.arange(-limit, limit, step)
Y_vals = np.arange(-limit, limit, step)
for i, x in enumerate(X_vals):
    for j, y in enumerate(Y_vals):
        test_in = np.array([[x, y]])
        output = decoder.predict(x=test_in)
        output = np.squeeze(output)
        ax[-j - 1, i].imshow(output, cmap='jet')
        ax[-j - 1, i].axis('off')
plt.show()
