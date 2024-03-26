import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
import os
from keras.layers import Dense, Flatten, Reshape, LeakyReLU, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import BatchNormalization, Activation, ZeroPadding2D

from keras.layers import Activation, BatchNormalization
from keras.layers import Conv2D, Conv2DTranspose
import cv2

width = 144
height = 144
channels = 1
shape = (width, height, channels)
noise_dim = 100

from keras.layers import Dense, Reshape, Conv2DTranspose, BatchNormalization, LeakyReLU, Activation
from keras.models import Sequential

from keras.layers import Dropout

def generator_model(noise_dim):
    model = Sequential()
    model.add(Dense(256 * 18 * 18, input_dim=noise_dim))
    model.add(Reshape((18, 18, 256)))

    model.add(Conv2DTranspose(128, kernel_size=3, strides=2, padding='same'))
    model.add(LeakyReLU(alpha=0.01))

    model.add(Conv2DTranspose(64, kernel_size=3, strides=1, padding='same'))
    model.add(LeakyReLU(alpha=0.01))
    
    model.add(Conv2DTranspose(32, kernel_size=3, strides=2, padding='same'))
    model.add(LeakyReLU(alpha=0.01))
    
    # model.add(Conv2DTranspose(16, kernel_size=3, strides=1, padding='same'))
    # model.add(LeakyReLU(alpha=0.01))

    # model.add(Conv2DTranspose(8, kernel_size=3, strides=1, padding='same'))
    # model.add(LeakyReLU(alpha=0.01))

    # 新しいConv2DTranspose層を追加
    # model.add(Conv2DTranspose(4, kernel_size=3, strides=2, padding='same'))
    # model.add(LeakyReLU(alpha=0.01))

    # もう一つのConv2DTranspose層を追加
    # model.add(Conv2DTranspose(2, kernel_size=3, strides=1, padding='same'))
    # model.add(LeakyReLU(alpha=0.01))
    
    # 最後のConv2DTranspose層とActivation層
    model.add(Conv2DTranspose(1, kernel_size=3, strides=2, padding='same'))
    model.add(Activation('tanh'))
    
    return model


def discriminator_model(shape):
    model = Sequential()

    model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=shape, padding="same"))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, kernel_size=3, strides=2, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    model.add(Conv2D(512, kernel_size=3, strides=2, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    model.add(Conv2D(1024, kernel_size=3, strides=2, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    # model.add(Conv2D(2048, kernel_size=3, strides=2, padding="same"))
    # model.add(LeakyReLU(alpha=0.2))
    # model.add(Dropout(0.25))

    # model.add(Conv2D(4096, kernel_size=3, strides=2, padding="same"))
    # model.add(BatchNormalization())
    # model.add(LeakyReLU(alpha=0.2))
    # model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    return model



def gan_model(generator, discriminator):
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    return model

discriminator = discriminator_model(shape)
discriminator.compile(loss='binary_crossentropy',
                      optimizer=Adam(lr=0.0001, beta_1=0.5),#0.0001→0.0002,
                      metrics=['accuracy'])

generator = generator_model(noise_dim)
generator.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0007, beta_1=0.5))#0.0003→0.0004

discriminator.trainable = False

gan = gan_model(generator, discriminator)
gan.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0007, beta_1=0.5))

losses = []
accuracies = []
iteration_checkpoints = []

def train(iterations, batch_size, sample_interval):
    X_train = load_images("./gray_144_face")
    X_train = X_train / 127.5 - 1.0
    X_train=np.expand_dims(X_train,axis=3)#グレースケールだから
    real_label = np.ones((batch_size, 1))  # 修正
    fake_label = np.zeros((batch_size, 1))  # 修正

    for iteration in range(iterations):
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        batch_images = X_train[idx]
        z = np.random.normal(0, 1, (batch_size, noise_dim))
        gene_imgs = generator.predict(z)

        d_loss_real = discriminator.train_on_batch(batch_images, real_label)
        d_loss_fake = discriminator.train_on_batch(gene_imgs, fake_label)
        d_loss, accuracy = 0.5 * np.add(d_loss_real, d_loss_fake)

        z = np.random.normal(0, 1, (batch_size, noise_dim))
        g_loss = gan.train_on_batch(z, real_label)


        if (iteration + 1) % sample_interval == 0:
            losses.append((d_loss, g_loss))
            accuracies.append(100.0 * accuracy)
            iteration_checkpoints.append(iteration + 1)

            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (iteration + 1, d_loss, 100.0 * accuracy, g_loss))
            save_images(generator, iteration + 1)



def save_images(generator, iteration, directory='144x144x_face_gray_images', image_grid_rows=4, image_grid_columns=4):
    if not os.path.exists(directory):
        os.makedirs(directory)

    z = np.random.normal(0, 1, (image_grid_rows * image_grid_columns, noise_dim))
    gene_imgs = generator.predict(z)
    gene_imgs = 0.5 * gene_imgs + 0.5
    fig, axs = plt.subplots(image_grid_rows, image_grid_columns, figsize=(4, 4), sharey=True, sharex=True)
    cnt = 0
    for row in range(image_grid_rows):
        for col in range(image_grid_columns):
            axs[row, col].imshow(gene_imgs[cnt, :, :, 0], cmap='gray')
            axs[row, col].axis('off')
            cnt += 1

    fig.savefig(f"{directory}/iteration_{iteration}.png")
    plt.close(fig)

def load_images(directory):#グレースケール用に変換する
    images = []
    for filename in os.listdir(directory):
        img = cv2.imread(os.path.join(directory, filename))
        if img is not None:  
            img = cv2.resize(img, (width, height))
            if len(img.shape) == 3:  # チャンネルが3の場合はグレースケールに変換
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            images.append(img)
        else:
            print(f"Warning: Failed to load image {filename}")
    if len(images) == 0:
        print("Error: No images loaded")
        return None
    else:
        return np.array(images)



iterations = 20000
batch_size = 128
sample_interval = 1  

train(iterations, batch_size, sample_interval)
generator.save('Face-gene.keras')
discriminator.save('Face-dis.keras')
gan.save('Face-gan.keras')


# X_train = load_images("./gray_144_face")
# if X_train is not None:
#     X_train = X_train / 127.5 - 1.0
#     real_label = np.ones((batch_size, 1))
#     fake_label = np.zeros((batch_size, 1))
#     train(iterations, batch_size, sample_interval)
#     generator.save('Face-gene.keras')
#     discriminator.save('Face-dis.keras')
#     gan.save('Face-gan.keras')
# else:
#     print("Error: No images loaded, training cannot proceed.")