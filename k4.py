import os
import json
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from collections import defaultdict
from learn import prepare_data, clean_data

# ディレクトリの設定
output_dir = './do/data/output/model'
valuation_dir = './do/data/output/valuation'
data_dir = './do/data/output/hand_info'

os.makedirs(valuation_dir, exist_ok=True)

# モデルパスとテストデータインデックスを読み込み
with open(os.path.join(output_dir, 'fold_info.json'), 'r', encoding='utf-8') as f:
    fold_model_paths = json.load(f)

# データ準備（左右の差を無視）
X, Y, groups = prepare_data(data_dir)

# ラベルから左右識別（l/r）を削除
Y_normalized = np.array([label[:-1] if label[-1] in ['l', 'r'] else label for label in Y])

# ラベルをエンコード
label_encoder = LabelEncoder()
Y_encoded = label_encoder.fit_transform(Y_normalized)
Y_one_hot = to_categorical(Y_encoded)

# エンコーダを保存
with open(os.path.join(output_dir, 'normalized_label_encoder.json'), 'w', encoding='utf-8') as f:
    json.dump(label_encoder.classes_.tolist(), f, ensure_ascii=False, indent=4)

# ラベルマッピングの作成
original_labels = [f"{str(i).zfill(2)}l" for i in range(1, 60)] + [f"{str(i).zfill(2)}r" for i in range(1, 60)]
normalized_labels = [f"{str(i).zfill(2)}" for i in range(1, 60)] * 2
label_mapping = {orig: norm for orig, norm in zip(original_labels, normalized_labels)}

# デバッグ：ラベルマッピングの確認
print("ラベルマッピングの内容:", label_mapping)
print("エンコーダのラベル数:", len(label_encoder.classes_))
print("エンコーダのラベル一覧:", label_encoder.classes_)

# 評価結果を格納するリスト
all_results = []
all_top_10_predictions = []
label_accuracy = defaultdict(lambda: {"correct": 0, "total": 0, "accuracy": 0.0})

for fold, (model_path, test_index) in enumerate(fold_model_paths):
    print(f"Fold {fold + 1} の評価を開始します")

    # モデルの読み込み
    model = load_model(model_path)

    # テストデータの準備
    X_test = clean_data(X[test_index])
    Y_test = Y_one_hot[test_index]

    # モデルの予測
    predictions = model.predict(X_test)
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(Y_test, axis=1)

    # 各foldの評価結果を保存
    fold_valuation_dir = os.path.join(valuation_dir, f'fold_{fold + 1}')
    os.makedirs(fold_valuation_dir, exist_ok=True)

    fold_top_10_predictions = []

    # デバッグ：ラベルと予測内容を確認
    for true_label, pred in zip(true_labels[:10], predictions[:10]):  # 最初の10件を確認
        true_label_str = label_encoder.classes_[true_label]
        true_label_mapped = label_mapping.get(true_label_str, "")
        print(f"True Label: {true_label_str}, Mapped: {true_label_mapped}")

        top_10_indices = np.argsort(pred)[-10:][::-1]
        top_10_labels = [
            {"label": label_mapping.get(label_encoder.classes_[i], ""), "score": float(pred[i])}
            for i in top_10_indices if i < len(label_encoder.classes_)
        ]
        print("Top 10 Labels:", top_10_labels)

        fold_top_10_predictions.append({
            "true_label": true_label_mapped,
            "top_10_labels": top_10_labels
        })

        if top_10_labels and true_label_mapped == top_10_labels[0]["label"]:
            label_accuracy[true_label_mapped]["correct"] += 1
        label_accuracy[true_label_mapped]["total"] += 1

    all_top_10_predictions.extend(fold_top_10_predictions)

# 正確な統合結果を計算
for label, stats in label_accuracy.items():
    if stats["total"] > 0:
        stats["accuracy"] = stats["correct"] / stats["total"]

mean_accuracy = sum(stats["correct"] for stats in label_accuracy.values()) / \
                sum(stats["total"] for stats in label_accuracy.values())

rl_results = {
    "mean_accuracy": mean_accuracy,
    "label_accuracy": label_accuracy,
}

# 結果を保存
with open(os.path.join(valuation_dir, 'rl_results.json'), 'w', encoding='utf-8') as f:
    json.dump(rl_results, f, indent=4)

with open(os.path.join(valuation_dir, 'rl_top_10_predictions.json'), 'w', encoding='utf-8') as f:
    json.dump(all_top_10_predictions, f, indent=4)

print(f"統合された結果が {valuation_dir} に保存されました。")
