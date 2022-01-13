import numpy as np
from keras.datasets import mnist
from keras.layers import Conv2D, MaxPool2D, Input, UpSampling2D, GaussianNoise
from keras.models import Model
from keras.optimizers import Adam
from matplotlib import pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data()

samples = 10000
x_train = x_train[:samples, :, :]
x_test = x_test[:samples, :, :]
y_train = y_train[:samples]
y_test = y_test[:samples]

x_train = np.expand_dims(x_train, axis=-1)
x_train_scaled = (x_train / 255).copy()

act_func = 'selu'

encoder_layers = [
    GaussianNoise(1),
    Conv2D(32, (3, 3), padding='same', activation=act_func),
    MaxPool2D(2, 2),
    Conv2D(64, (3, 3), padding='same', activation=act_func),
    MaxPool2D(2, 2),
    Conv2D(128, (3, 3), padding='same', activation=act_func)
]

decoder_layers = [
    UpSampling2D((2, 2)),
    Conv2D(32, (3, 3), padding='same', activation=act_func),
    UpSampling2D((2, 2)),
    Conv2D(32, (3, 3), padding='same', activation=act_func),
    Conv2D(1, (3, 3), padding='same', activation='sigmoid')
]

lrng_rate = 0.0001
tensor = autoencoder_input = Input(x_train_scaled.shape[1:])

for layer in encoder_layers + decoder_layers:
    tensor = layer(tensor)

autoencoder = Model(inputs=autoencoder_input, outputs=tensor)
autoencoder.compile(optimizer=Adam(lrng_rate), loss='binary_crossentropy')
autoencoder.fit(x=x_train_scaled, y=x_train_scaled, epochs=30, batch_size=256, verbose=2)

# # wersja 1
# test_photos = x_train[10:20, ...].copy()
# noisy_test_photos = test_photos.copy()
# mask = np.random.randn(*test_photos.shape)
# white = mask > 1
# black = mask < -1
#
# noisy_test_photos[white] = 255
# noisy_test_photos[black] = 0
# noisy_test_photos = noisy_test_photos / 255

# wersja 2
test_photos = x_train_scaled[10:20, ...].copy()
noisy_test_photos = test_photos.copy()
mask = np.random.randn(*test_photos.shape)
white = mask > 1
black = mask < -1

noisy_test_photos[white] = 1
noisy_test_photos[black] = 0


def show_pictures(arrs):
    arr_cnt = arrs.shape[0]
    fig, axes = plt.subplots(1, arr_cnt, figsize=(5 * arr_cnt, arr_cnt))
    for axis, pic in zip(axes, arrs):
        axis.imshow(pic.squeeze(), cmap='gray')

    plt.show()


cleaned_images = autoencoder.predict(noisy_test_photos) * 255
show_pictures(test_photos)
show_pictures(noisy_test_photos)
show_pictures(cleaned_images)
