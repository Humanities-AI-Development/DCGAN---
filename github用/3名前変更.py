import os
import random
import string

# ディレクトリのパスを指定
directory = './resized_60'

# ディレクトリ内のファイルを取得
files = os.listdir(directory)

# ファイルごとにループ
for filename in files:
    # 拡張子を取得
    _, ext = os.path.splitext(filename)
    
    # 新しいファイル名を生成
    new_filename = ''.join(random.choices(string.digits, k=8)) + ext
    
    # 新しいファイル名のパスを作成
    new_filepath = os.path.join(directory, new_filename)
    
    # ファイル名を変更
    os.rename(os.path.join(directory, filename), new_filepath)
