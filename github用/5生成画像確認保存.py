import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import os
import cv2
noise_dim=100
# 保存したGANモデルの読み込み
generator = load_model('40x40color-Face-gene.keras')

def generate_images(generator, save_directory, num_images=10):
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    noise = np.random.normal(0, 1, (num_images, noise_dim))
    generated_images = generator.predict(noise)

    for i in range(num_images):
        img = (generated_images[i] + 1) / 2  # 画像のスケーリングを元に戻す
        #img_resize = cv2.resize(img, (50, 50))  # 50x50にリサイズ

        plt.imshow(img)
        plt.axis('off')
        plt.savefig(f"{save_directory}/generated_image_{i+1}.png")
        plt.close()

# 画像を保存するディレクトリと生成する画像の枚数を指定して実行
generate_images(generator, "generated_images", num_images=40)
