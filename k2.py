import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model
from valuation import evaluate_model_with_visualization
from learn import clean_data, prepare_data
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# データ準備
output_dir = './do/data/output/model'
valuation_dir = './do/data/output/valuation'
data_dir = './do/data/output/hand_info'

# 保存ディレクトリ作成
os.makedirs(valuation_dir, exist_ok=True)

# モデルパスとテストデータインデックスを読み込み
with open(os.path.join(output_dir, 'fold_info.json'), 'r', encoding='utf-8') as f:
    fold_model_paths = json.load(f)

# ラベルエンコーダーの読み込み
with open(os.path.join(output_dir, 'label_encoder.json'), 'r', encoding='utf-8') as f:
    label_encoder_classes = json.load(f)

# データを再準備
from learn import prepare_data
X, Y, groups = prepare_data(data_dir)
Y_encoded = LabelEncoder().fit_transform(Y)
Y_one_hot = to_categorical(Y_encoded)

# 各foldの評価
for fold, (model_path, test_index) in enumerate(fold_model_paths):
    print(f"Evaluating fold {fold + 1}")

    # テストデータの準備
    X_test = clean_data(X[test_index])
    Y_test = Y_one_hot[test_index]

    # 評価用ディレクトリ
    fold_valuation_dir = os.path.join(valuation_dir, f'fold_{fold + 1}')
    
    # 必要なディレクトリがない場合、親ディレクトリから再帰的に作成
    os.makedirs(fold_valuation_dir, exist_ok=True)

    # モデル評価
    # fold_model_paths から各foldのモデルファイルをロード
    model_path = model_path  # 各foldごとのモデルファイルパスが保存されている

    # モデル評価の呼び出し
    evaluate_model_with_visualization(
        X=X_test,
        y=Y_test,
        subjects=groups[test_index],
        model_path=model_path,  # 各foldごとのモデルを評価に使用
        label_encoder_path=os.path.join(output_dir, 'label_encoder.json'),
        output_path=fold_valuation_dir  # 結果を保存するディレクトリ
    )

print("評価が完了しました。")
