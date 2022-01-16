import numpy as np
from keras.datasets import fashion_mnist
from keras.layers import Conv2D, MaxPool2D, Input, UpSampling2D, GaussianNoise
from keras.models import Model
from keras.optimizers import Adam
from matplotlib import pyplot as plt

# wczytanie danych
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# zmniejszenie zbioru treningowego
samples = 10000
x_train = x_train[:samples, :, :]
y_train = y_train[:samples]
# x_test = x_test[:samples, :, :]
# y_test = y_test[:samples]

# dodanie 4 wymiaru
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# skalowanie
x_train_scaled = (x_train / 255).copy()
x_test_scaled = (x_test / 255).copy()

# hiperparametry
act_func = 'selu'
lrng_rate = 0.0001

# warstwy enkodera
encoder_layers = [
    GaussianNoise(1),
    Conv2D(32, (3, 3), padding='same', activation=act_func),
    MaxPool2D(2, 2),
    Conv2D(64, (3, 3), padding='same', activation=act_func),
    MaxPool2D(2, 2),
    Conv2D(128, (3, 3), padding='same', activation=act_func)
]

# warstwy dekodera
decoder_layers = [
    UpSampling2D((2, 2)),
    Conv2D(32, (3, 3), padding='same', activation=act_func),
    UpSampling2D((2, 2)),
    Conv2D(32, (3, 3), padding='same', activation=act_func),
    Conv2D(1, (3, 3), padding='same', activation='sigmoid')
]

# utworzenie warstwy wejściowej auto-enkodera
tensor = autoencoder_input = Input(x_train_scaled.shape[1:])

# dodanie warstw do auto-enkodera
for layer in encoder_layers + decoder_layers:
    tensor = layer(tensor)

# utworzenie modelu, kompilacja oraz uczenie
autoencoder = Model(inputs=autoencoder_input, outputs=tensor)
autoencoder.compile(optimizer=Adam(lrng_rate), loss='binary_crossentropy', metrics=['mean_squared_error'])
autoencoder.fit(x=x_train_scaled, y=x_train_scaled, epochs=20, batch_size=256,
                validation_data=(x_test_scaled, x_test_scaled), verbose=2)

# # wprowadzenie szumu do zdjęć — wersja 1
# test_photos = x_train[10:20, ...].copy()
# noisy_test_photos = test_photos.copy()
# mask = np.random.randn(*test_photos.shape)
# white = mask > 1
# black = mask < -1
#
# noisy_test_photos[white] = 255
# noisy_test_photos[black] = 0
# noisy_test_photos = noisy_test_photos / 255

# wprowadzenie szumu do zdjęć — wersja 2
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
