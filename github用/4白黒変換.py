import cv2
import os

# 入力画像が保存されているフォルダパス
input_folder = ''

# 出力フォルダパス
output_folder = 'gray_60'

# 入力フォルダ内のすべての画像ファイルを処理
for filename in os.listdir(input_folder):
    # 画像ファイルのパスを取得
    input_path = os.path.join(input_folder, filename)
    
    # 画像をグレースケールで読み込む
    img = cv2.imread(input_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 出力ファイルのパスを指定
    output_path = os.path.join(output_folder, filename)
    
    # グレースケール画像を保存
    cv2.imwrite(output_path, gray_img)

print("変換が完了しました。")
