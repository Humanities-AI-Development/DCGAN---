from PIL import Image
import os

def resize_images(input_folder, output_folder, target_size):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        try:
            # 画像を開く
            img_path = os.path.join(input_folder, filename)
            img = Image.open(img_path)
            # 画像を指定されたサイズにリサイズする
            resized_img = img.resize(target_size)
            # リサイズした画像を保存する
            output_path = os.path.join(output_folder, filename)
            resized_img.save(output_path)
            print(f"Resized image: {output_path}")
        except Exception as e:
            print(f"Error resizing image {filename}: {e}")

# 使用例
input_folder = './woman_images'  # 元の画像が保存されているフォルダ
output_folder = './resized_60'    # リサイズされた画像を保存するフォルダ
target_size=(60,60)# 目標の画像サイズ (幅, 高さ)
resize_images(input_folder, output_folder, target_size)
