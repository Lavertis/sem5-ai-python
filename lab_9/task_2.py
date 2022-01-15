import numpy as np
from keras.datasets import mnist
from keras.layers import Input, GaussianNoise, Dense, Reshape, Flatten
from keras.models import Model
from keras.optimizers import Adam
from matplotlib import pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train_scaled = (x_train / 255).copy()

encoder_layers = [
    GaussianNoise(1),
    Flatten(),
    Dense(784, activation='relu'),
    Dense(512, activation='relu'),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
]

decoder_layers = [
    Dense(256, activation='relu'),
    Dense(512, activation='relu'),
    Dense(784, activation='sigmoid'),
    Reshape(x_train_scaled.shape[1:])
]

lrng_rate = 0.0001
tensor = autoencoder_input = Input(x_train_scaled.shape[1:])

for layer in encoder_layers + decoder_layers:
    tensor = layer(tensor)

autoencoder = Model(inputs=autoencoder_input, outputs=tensor)
autoencoder.compile(optimizer=Adam(lrng_rate), loss='binary_crossentropy')
autoencoder.fit(x=x_train_scaled, y=x_train_scaled, epochs=30, batch_size=256, verbose=2)

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
