import matplotlib.pyplot as plt
import tkinter
# import PIL
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import math
import os

from tensorflow.python.keras.models import Sequential, Model, load_model      #  try tf.keras.models import Sequential in the future
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import InputLayer, Input
from tensorflow.python.keras.layers import Reshape, MaxPooling2D, BatchNormalization, LeakyReLU, Dropout
from tensorflow.python.keras.layers import Conv2D, Conv2DTranspose, Dense, Flatten, Activation, UpSampling2D
from tensorflow.python.keras.optimizers import Adam, RMSprop
from tensorflow.examples.tutorials.mnist import input_data
from IPython import display


data = input_data.read_data_sets('data/MNIST', one_hot=True)


# data params
img_size = 28
img_size_flat = img_size * img_size
img_shape = (img_size, img_size)
latent_size = 128

# for reshaping in Keras
num_channels = 1
img_size_full = (img_size, img_size, num_channels)
num_classes = 1
batch_size = 64
smooth = 0.1
mi = -1.0
mx = 1.0


# helper functions for plotting images, examples of errors


def plot_generated_images(generator):
    noise = np.random.uniform(mi,mx,size=[9 ,latent_size])
    images = generator.predict(noise)
    assert len(images) == 9

    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].reshape(img_shape), cmap='Greys_r')
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()


def plot_loss(losses):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.figure(figsize=(10, 8))
    plt.plot(losses["disc"], label='discriminator loss')
    plt.plot(losses["gen"], label='generator loss')
    plt.legend()
    plt.show()


def trainable_model(net, val):
    net.trainable = val
    for layer in net.layers:
        layer.trainable = val
    return


def generate_fake_dataset(gen):
    num = int(batch_size)
    z = []
    for _ in range(num):
        z.append(np.random.normal(mi, mx, size=(latent_size, )))
    z = np.array(z)
    fake_images = np.array(gen.predict(x=z, batch_size=num))
    fake_images = fake_images.reshape((num, img_size_flat))
    z = np.array(z)
    return z, fake_images


def create_data_batch(genb, noise=False):

    if not noise:
        z, fake_images = generate_fake_dataset(genb)
        batch_indices = np.random.randint(0, len(data.train.images), size=int(batch_size))
        #print(batch_indices)
        data_batch = []
        data_batch_labels = []

        for i in range(len(batch_indices)):
            data_batch.append(data.train.images[batch_indices[i]])
            data_batch_labels.append(1)
        for i in range(len(batch_indices), 2*len(batch_indices)):
            data_batch.append(fake_images[i-len(batch_indices)])
            data_batch_labels.append(0)

        data_batch = np.array(data_batch)
        data_batch_labels = np.array(data_batch_labels)
        np.squeeze(data_batch)
        return z, data_batch, data_batch_labels

    elif noise:
        z = []
        data_batch_labels = []
        for _ in range(batch_size*2):
            z.append(np.random.normal(mi, mx, size=(latent_size,)))
            data_batch_labels.append(1)
        data_batch = np.array(z)
        data_batch_labels = np.array(data_batch_labels)
        return data_batch, data_batch_labels


# creating stacked model

optimizer_g = Adam(lr=1e-3, decay=3e-8)
optimizer_d = Adam(lr=2e-3, decay=6e-8)
metrics = ['accuracy']
loss_ge = 'binary_crossentropy'
loss_de = 'binary_crossentropy'


# first the generator
gen = Sequential()
dropout = 0.4
channel_depth = 128
dim = 7
gen.add(Dense(dim * dim * channel_depth, input_dim=latent_size))
gen.add(BatchNormalization(momentum=0.9))
gen.add(LeakyReLU(alpha=0.2))
gen.add(Reshape((dim, dim, channel_depth)))
gen.add(Dropout(dropout))

gen.add(UpSampling2D())
gen.add(Conv2DTranspose(int(channel_depth / 4), 5, padding='same'))
gen.add(BatchNormalization(momentum=0.9))
gen.add(LeakyReLU(alpha=0.2))

gen.add(UpSampling2D())
gen.add(Conv2DTranspose(int(channel_depth / 8), 5, padding='same'))
gen.add(BatchNormalization(momentum=0.9))
gen.add(LeakyReLU(alpha=0.2))
'''
gen.add(Conv2DTranspose(int(channel_depth / 8), 5, padding='same'))
gen.add(BatchNormalization(momentum=0.9))
gen.add(Activation('relu'))
'''

# Out: 28 x 28 x 1
gen.add(Conv2DTranspose(1, 5, padding='same'))
gen.add(Activation('tanh'))


# now the discriminator
channel_depth = 16
dropout = 0.4

D = Sequential()
D.add(InputLayer(input_shape=(img_size_flat, )))
D.add(Reshape(img_size_full))

D.add(Conv2D(channel_depth * 1, 5, strides=2, padding='same'))
D.add(BatchNormalization(momentum=0.9))
D.add(LeakyReLU(alpha=0.2))
D.add(Dropout(dropout))


D.add(Conv2D(channel_depth * 2, 5, strides=2, padding='same'))
D.add(LeakyReLU(alpha=0.2))
D.add(Dropout(dropout))

'''
D.add(Conv2D(channel_depth * 4, 5, strides=2, padding='same'))
D.add(LeakyReLU(alpha=0.2))
D.add(Dropout(dropout))

D.add(Conv2D(channel_depth * 8, 5, strides=1, padding='same'))
D.add(LeakyReLU(alpha=0.2))
D.add(Dropout(dropout))
'''

D.add(Flatten())
D.add(Dense(1))
D.add(Activation('sigmoid'))
# D.summary()

# DISC model
disc = Sequential()
disc.add(D)
disc.compile(loss=loss_de, optimizer=optimizer_d,
                metrics=metrics)

# GAN model
gan = Sequential()
gan.add(gen)
gan.add(D)
gan.compile(loss=loss_ge, optimizer=optimizer_g,
                metrics=metrics)
#gan.summary()
print("\n\n")


# training and evaluating


losses = {'gen': [], 'disc': []}


def batch_train_for_n(n_epoch, genb, discb, ganb, k=5):

    for i in tqdm(range(n_epoch)):
        Z, X, y = create_data_batch(genb=genb)
        for _ in range(k):
            loss_d, acc_d = discb.train_on_batch(x=X, y=y)
        if i % 100 == 0:
            losses['disc'].append(loss_d)

        Z, y = create_data_batch(noise=True, genb=genb)
        loss_g, acc_g = ganb.train_on_batch(x=Z, y=y)
        if i % 100 == 0:
            losses['gen'].append(loss_g)
            print()
            print("iteration: ", i, "   loss_d: ", loss_d, "   loss_g: ", loss_g)
            print("acc_d: ", acc_d, "   acc_g: ", acc_g)
            print()
        if i%500 == 0:
            plot_generated_images(generator=genb)

    plot_loss(losses)
    ganb.save(filepath='gan2__.h5')
    discb.save(filepath='disc2__.h5')
    genb.save(filepath='gen2__.h5')
    return genb

#genn = load_model('gen2__.h5')
#gann = load_model('gan2__.h5')
#discn = load_model('disc2__.h5')


gen1 = batch_train_for_n(n_epoch=3001, genb=gen, ganb=gan, discb=disc)
plot_generated_images(generator=gen1)








