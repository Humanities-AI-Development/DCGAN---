import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import os

noise_dim=100
def generate_and_save_images(generator, noise_dim, save_directory='generated_images', num_images=10):
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    
    for i in range(num_images):
        noise = np.random.normal(0, 1, (1, noise_dim))
        generated_image = generator.predict(noise)
        generated_image = 0.5 * generated_image + 0.5  # 画像のスケーリングを元に戻す
        generated_image = np.squeeze(generated_image)  # 不要な次元を削除
        plt.imshow(generated_image, cmap='gray')
        plt.axis('off')
        plt.savefig(f"{save_directory}/generated_image_{i+1}.png")
        plt.close()

# 生成器モデルをロードする
generator = load_model('Face-gene.keras')

# 10枚の画像を生成して保存する
generate_and_save_images(generator, noise_dim)
