import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from learn import prepare_data

# === データ準備 ===
sys.path.append("./do/matura_arbeit/learn.py")
data_dir = "./do/data/output/hand_info"
angles, labels, subjects = prepare_data(data_dir)

# ラベルエンコーディング
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
labels_one_hot = to_categorical(labels_encoded)

# ラベルを保存
with open("./do/data/output/model/label_encoder.json", "w") as f:
    json.dump(label_encoder.classes_.tolist(), f, indent=4)

# === ロジットを保存する関数 ===
def evaluate_model_with_logits(X, y, subjects, model_path, label_encoder_path, output_path):
    # モデルとラベルエンコーダのロード
    model = load_model(model_path)
    with open(label_encoder_path, "r") as f:
        label_encoder_classes = json.load(f)

    X = np.array(X)
    y = np.array(y)

    # **Softmax適用前のロジットを取得**
    logits = model(X)  # Softmax適用前の出力

    # Top-10 ロジットを取得
    top_10_predictions = [
        {
            "true_label": label_encoder_classes[true_label],
            "top_10_logits": [
                {"label": label_encoder_classes[i], "logit": float(logits[idx][i])}
                for i in np.argsort(logits[idx])[-10:][::-1]  # 上位10件
            ],
        }
        for idx, true_label in enumerate(np.argmax(y, axis=1))
    ]

    # **JSONに保存**
    with open(f"{output_path}/top_10_logits.json", "w") as f:
        json.dump(top_10_predictions, f, indent=4)

    print(f"Logits saved to {output_path}/top_10_logits.json")

# === 各FOLDの評価 ===
with open('./do/data/output/model/fold_info.json', 'r', encoding='utf-8') as f:
    fold_model_paths = json.load(f)

for fold, (model_path, test_index) in enumerate(fold_model_paths):
    print(f"Evaluating fold {fold + 1}")
    evaluate_model_with_logits(
        X=angles,
        y=labels_one_hot,
        subjects=subjects,
        model_path=model_path,
        label_encoder_path="./do/data/output/model/label_encoder.json",
        output_path=f"./do/data/output/valuation/fold_{fold + 1}"
    )
