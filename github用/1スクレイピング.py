import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs
import time

def is_valid_url(url):
    # URLが有効なHTTPまたはHTTPSスキームを持っているかどうかを確認する
    return url.startswith("http://") or url.startswith("https://")

def download_images(query, num_images):
    save_dir = './woman_images'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    headers = {'User-Agent': 'Mozilla/5.0'}
    downloaded = 0
    while downloaded < num_images:
        # キーワードを含むURLを構築
        url = f"https://www.google.com/search?q={query}&tbm=isch&start={downloaded}"
        # ヘッダーを設定してGoogleにリクエストを送信
        response = requests.get(url, headers=headers)
        # HTMLをパース
        soup = BeautifulSoup(response.text, 'html.parser')
        # 画像のURLを抽出
        image_urls = []
        for img in soup.find_all('img'):
            image_url = img.get('src')
            if image_url and is_valid_url(image_url):
                image_urls.append(image_url)
        # 画像をダウンロード
        for i, image_url in enumerate(image_urls):
            try:
                if downloaded >= num_images:
                    break
                image_data = requests.get(image_url).content
                with open(f'{save_dir}/{query}_{downloaded}.jpg', 'wb') as f:
                    f.write(image_data)
                print(f"Downloaded image {downloaded+1}/{num_images}")
                downloaded += 1
            except Exception as e:
                print(f"Error downloading image {downloaded+1}: {e}")
        time.sleep(1)  # Googleへの負荷を軽減するために1秒待つ

# 使用例
query = "有村架純　高画質"  # 検索キーワード
num_images = 100  # ダウンロードする画像の数
download_images(query, num_images)
