import numpy as np
from keras.datasets import fashion_mnist
from keras.layers import Conv2D, MaxPool2D, Input, UpSampling2D, GaussianNoise, Dense
from keras.models import Model
from keras.optimizers import Adam
from matplotlib import pyplot as plt

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

samples = 10000
x_train = x_train[:samples, :, :]
x_test = x_test[:samples, :, :]
y_train = y_train[:samples]
y_test = y_test[:samples]

x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

x_train_scaled = (x_train / 255).copy()
x_test_scaled = (x_test / 255).copy()

act_func = 'selu'
aec_dim_num = 2

encoder_layers = [
    GaussianNoise(1),
    Conv2D(32, (3, 3), padding='same', activation=act_func),
    MaxPool2D(2, 2),
    Conv2D(64, (3, 3), padding='same', activation=act_func),
    MaxPool2D(2, 2),
    Conv2D(128, (3, 3), padding='same', activation=act_func),
    Dense(aec_dim_num, activation='tanh')
]

decoder_layers = [
    UpSampling2D((2, 2)),
    Conv2D(32, (3, 3), padding='same', activation=act_func),
    UpSampling2D((2, 2)),
    Conv2D(32, (3, 3), padding='same', activation=act_func),
    Conv2D(1, (3, 3), padding='same', activation='sigmoid')
]

lrng_rate = 0.0001
tensor = input_aec = input_encoder = Input(x_train_scaled.shape[1:])

for layer in encoder_layers:
    tensor = layer(tensor)

output_encoder = tensor
dec_tensor = input_decoder = Input(output_encoder.shape[1:])

for layer in decoder_layers:
    tensor = layer(tensor)
    dec_tensor = layer(dec_tensor)

output_aec = tensor
output_decoder = dec_tensor

autoencoder = Model(inputs=input_aec, outputs=tensor)
encoder = Model(inputs=input_encoder, outputs=output_encoder)

autoencoder.compile(optimizer=Adam(lrng_rate), loss='binary_crossentropy', metrics=['mean_squared_error'])
autoencoder.fit(x=x_train_scaled, y=x_train_scaled, epochs=30, batch_size=256,
                validation_data=(x_test_scaled, x_test_scaled), verbose=2)

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

fig, ax = plt.subplots(1, 1, figsize=(20, 16))
for i in range(10):
    digits = y_train == i
    needed_imgs = x_train_scaled[digits, ...]

    preds = encoder.predict(needed_imgs)
    ax.scatter(preds[:, 0], preds[:, 1])

ax.legend(list(range(10)))
plt.show()
