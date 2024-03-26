from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers import LeakyReLU
from keras.layers import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np

width = 40
height = 40
channels = 3
shape = (width, height, channels)
noise_dim = 100


def generator_model(noise_dim):
    model = Sequential()
    model.add(Dense(10 * 10 * 256, activation="relu", input_dim=noise_dim))
    model.add(Reshape((10, 10, 256)))
    model.add(BatchNormalization(momentum=0.8))
    model.add(UpSampling2D())

    model.add(Conv2D(128, kernel_size=3, padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(UpSampling2D())


    model.add(Conv2D(64, kernel_size=3, padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(momentum=0.8))

    model.add(Conv2D(3, kernel_size=3, padding="same"))
    model.add(Activation("tanh"))
    return model



def discriminator_model(shape):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=shape, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
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
                      optimizer=Adam(lr=0.0001, beta_1=0.5),
                      metrics=['accuracy'])

generator = generator_model(noise_dim)
generator.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0003, beta_1=0.5))

discriminator.trainable = False

gan = gan_model(generator, discriminator)
gan.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0003, beta_1=0.5))



losses = []
accuracies = []
iteration_checkpoints = []



def train(iterations, batch_size, sample_interval):
    X_train = load_images("./resized_images_40")
    X_train = X_train / 127.5 - 1.0
    real_label = np.ones((batch_size, 1))  # 修正
    fake_label = np.zeros((batch_size, 1))  # 修正

    for iteration in range(iterations):
        idx_real = np.random.randint(0, X_train.shape[0] - 1, batch_size)  # 修正
        idx_fake = np.random.randint(0, X_train.shape[0] - 1, batch_size)  # 修正
        batch_images_real = X_train[idx_real]
        batch_images_fake = X_train[idx_fake]
        
        # ノイズの生成時に正しい次元数を使用する
        z = np.random.normal(0, 1, (batch_size, noise_dim))  # 修正
        gene_imgs = generator.predict(z)

        d_loss_real = discriminator.train_on_batch(batch_images_real, real_label)  # 修正
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


def save_images(generator, iteration, directory='face-gan_images', image_grid_rows=4, image_grid_columns=4):
    if not os.path.exists(directory):
        os.makedirs(directory)

    z = np.random.normal(0, 1, (image_grid_rows * image_grid_columns, noise_dim))
    gene_imgs = generator.predict(z)
    gene_imgs = 0.5 * gene_imgs + 0.5
    fig, axs = plt.subplots(image_grid_rows, image_grid_columns, figsize=(4, 4), sharey=True, sharex=True)
    cnt = 0
    for row in range(image_grid_rows):
        for col in range(image_grid_columns):
            axs[row, col].imshow(gene_imgs[cnt])
            axs[row, col].axis('off')
            cnt += 1

    fig.savefig(f"{directory}/iteration_{iteration}.png")
    plt.close(fig)


def load_images(directory):
    images = []
    for filename in os.listdir(directory):
        img = cv2.imread(os.path.join(directory, filename))
        if img is not None:  # 画像の読み込みが成功した場合のみ処理を続行
            img = cv2.resize(img, (width, height))
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

X_train = load_images("./resized_images_40")
if X_train is not None:
    X_train = X_train / 127.5 - 1.0
    real_label = np.ones((batch_size, 1))
    fake_label = np.zeros((batch_size, 1))
    train(iterations, batch_size, sample_interval)
    generator.save('40x40color-Face-gene.keras')
    discriminator.save('40x40color-Face-dis.keras')
    gan.save('40x40color-Face-gan.keras')
else:
    print("Error: No images loaded, training cannot proceed.")
