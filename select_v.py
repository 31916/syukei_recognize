import os
import shutil
import glob

yubimoji_dir = r'./do/data/yubimoji'
valuation_dir = r'./do/data/valuation'

# ディレクトリが存在する場合は削除
if os.path.exists(valuation_dir):
    shutil.rmtree(valuation_dir)  # ディレクトリを削除

# 必要に応じて再作成
os.makedirs(valuation_dir, exist_ok=True)

# ディレクトリ内のすべての動画ファイルを取得
video_paths_avi = glob.glob(os.path.join(yubimoji_dir, '*.avi'))
video_paths_mp4 = glob.glob(os.path.join(yubimoji_dir, '*.mp4'))

L =[]

for video_path in video_paths_avi:
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    if 'L' in video_name:
        L.append(video_path)
    
print(L)