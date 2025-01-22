import os
import json
import numpy as np
from tensorflow.keras.models import load_model
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
label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(label_encoder_classes)
Y_encoded = label_encoder.transform(Y)
Y_one_hot = to_categorical(Y_encoded)

print("Registered labels:", label_encoder.classes_)

# 各foldの評価
for fold, (model_path, test_index) in enumerate(fold_model_paths):
    print(f"Evaluating fold {fold + 1}")

    # テストデータの準備
    X_test_clean, Y_test_clean = clean_data(X[test_index], Y[test_index])
    true_labels = Y[test_index]

    # 評価用ディレクトリ
    fold_valuation_dir = os.path.join(valuation_dir, f'fold_{fold + 1}')
    os.makedirs(fold_valuation_dir, exist_ok=True)

    # モデルをロード
    model = load_model(model_path)

    # モデルの予測
    predictions = model.predict(X_test_clean)

    # Top-10予測の保存
    top_10_predictions = []
    for i, probs in enumerate(predictions):
        top_indices = np.argsort(probs)[::-1][:10]
        top_labels = label_encoder.inverse_transform(top_indices)
        top_probs = probs[top_indices]
        top_10_predictions.append({
            "true_label": true_labels[i],
            "top_10": [
                {"label": top_labels[j], "probability": float(top_probs[j])}
                for j in range(len(top_labels))
            ]
        })

    top_10_save_path = os.path.join(fold_valuation_dir, 'top_10_predictions.json')
    with open(top_10_save_path, 'w', encoding='utf-8') as f:
        json.dump(top_10_predictions, f, indent=4)

    # クラスごとの精度計算
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels_encoded = label_encoder.transform(true_labels)
    correct_per_class = np.zeros(len(label_encoder.classes_), dtype=int)
    total_per_class = np.zeros(len(label_encoder.classes_), dtype=int)

    for true_label, pred_label in zip(true_labels_encoded, predicted_labels):
        total_per_class[true_label] += 1
        if true_label == pred_label:
            correct_per_class[true_label] += 1

    accuracy_per_label = {
        label_encoder.classes_[i]: {
            "correct": int(correct_per_class[i]),
            "total": int(total_per_class[i]),
            "accuracy": float(correct_per_class[i] / total_per_class[i] if total_per_class[i] > 0 else 0)
        }
        for i in range(len(label_encoder.classes_))
    }

    # 平均精度計算
    mean_accuracy = float(np.mean([v["accuracy"] for v in accuracy_per_label.values() if v["total"] > 0]))

    # 結果を保存
    results_save_path = os.path.join(fold_valuation_dir, 'results.json')
    with open(results_save_path, 'w', encoding='utf-8') as f:
        json.dump({
            "mean_accuracy": mean_accuracy,
            "accuracy_per_label": accuracy_per_label
        }, f, indent=4)

    print(f"Top-10 predictions saved to {top_10_save_path}")
    print(f"Results saved to {results_save_path}")

print("Evaluation completed.")
