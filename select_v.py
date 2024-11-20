import os
import shutil
import glob
import re  # 正規表現を使用

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

# ひらがなまたはカタカナの正規表現
kana_pattern = re.compile(r'[ぁ-んァ-ン]+')  # 連続したひらがな・カタカナを抽出

# 条件に一致する動画の情報を保管する2次元リスト
L = []
R = []

for video_path in video_paths_avi + video_paths_mp4:  # avi と mp4 を結合
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    # ファイル名からひらがなとカタカナのみ抽出
    kana_text = ''.join(kana_pattern.findall(video_name))
    if kana_text:  # ひらがなまたはカタカナが含まれている場合のみ
        if 'L' in video_name:  # 'L' を含むか判定
            L.append([video_path, kana_text])  # パスと抽出した文字列をリストに追加
        else:
            R.append([video_path, kana_text])  # パスと抽出した文字列をリストに追加

# 移動済みアイテムを追跡するためのリスト
moved_items = []

# LとRのひらがなカタカナ部分が一致するものをvaluationディレクトリに移動
for l_item in L:
    for r_item in R:
        if l_item[1] == r_item[1] and l_item not in moved_items and r_item not in moved_items:  
            # 一致する場合、両方のファイルをvaluationディレクトリに移動
            l_target = os.path.join(valuation_dir, os.path.basename(l_item[0]))
            r_target = os.path.join(valuation_dir, os.path.basename(r_item[0]))

            # ファイルの存在を確認してから移動
            if os.path.exists(l_item[0]):
                shutil.move(l_item[0], l_target)
            if os.path.exists(r_item[0]):
                shutil.move(r_item[0], r_target)

            # 移動済みのアイテムを追跡リストに追加
            moved_items.append(l_item)
            moved_items.append(r_item)
