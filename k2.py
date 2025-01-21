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
X, Y, groups = prepare_data(data_dir)
Y_encoded = LabelEncoder().fit_transform(Y)
Y_one_hot = to_categorical(Y_encoded)

# ラベルエンコーダーのインスタンスを作成
label_encoder = LabelEncoder()
label_encoder.fit(label_encoder_classes)

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

# 統合された混同行列を作成する関数
def integrate_confusion_matrices(fold_model_paths, X, Y, label_encoder_classes, output_dir):
    # 各foldで計算された混同行列を合計するための初期化
    total_cm = np.zeros((len(label_encoder_classes), len(label_encoder_classes)), dtype=int)
    
    # 各foldでの評価
    for fold, (model_path, test_index) in enumerate(fold_model_paths):
        print(f"Evaluating fold {fold + 1}")
        
        # テストデータの準備
        X_test = X[test_index]
        Y_test = Y[test_index]
        
        # モデルをロード
        model = load_model(model_path)
        
        # 予測を取得
        predictions = model.predict(X_test)
        predicted_labels = np.argmax(predictions, axis=1)
        
        # true_labelsを数値ラベルに変換
        true_labels = label_encoder.transform(Y_test)  # 文字列ラベルを数値ラベルに変換

        # 混同行列を計算
        cm = confusion_matrix(true_labels, predicted_labels)
        
        # 合計する
        total_cm += cm

    # 統合した混同行列をプロットして保存
    plt.figure(figsize=(14, 12))  # 画像サイズをさらに大きく調整
    sns.heatmap(total_cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder_classes, yticklabels=label_encoder_classes,
                cbar_kws={'label': 'Count'}, annot_kws={'size': 14}, linewidths=1, linecolor='black')  # セルの間に線を入れ、アノテーションの文字を大きく

    # タイトルとラベルのフォントサイズを変更
    plt.title("Integrated Confusion Matrix", fontsize=18)
    plt.xlabel("Predicted Labels", fontsize=16)
    plt.ylabel("True Labels", fontsize=16)

    # x軸（予測ラベル）を縦に表示
    plt.xticks(rotation=90, fontsize=7)  # x軸ラベルを縦向きにしてフォントサイズを設定
    plt.yticks(rotation=0, fontsize=7)  # y軸ラベルのフォントサイズを設定

    # ラベルとラベルの間隔を広げるための調整
    plt.subplots_adjust(left=0.1, right=0.2, top=0.2, bottom=0.1)  # プロットの余白を広げてラベル間隔を増やす

    # レイアウトを調整して画像を小さく
    plt.tight_layout()

    # 結果を画像として保存（高解像度で保存）
    plt.savefig(f"{output_dir}/integrated_confusion_matrix_wider_labels_vertical_predicted.png", dpi=300)  # 解像度を高く設定
    plt.close()



# 統合した混同行列を作成
integrate_confusion_matrices(fold_model_paths, X, Y, label_encoder_classes, valuation_dir)

print("評価が完了しました。")
