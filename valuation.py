import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import load_model
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from learn import prepare_data

# 必要なデータの準備
sys.path.append("./do/matura_arbeit/learn.py")
data_dir = "./do/data/output/hand_info"
angles, labels, subjects = prepare_data(data_dir)

# ラベルエンコーディングの処理
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
labels_one_hot = to_categorical(labels_encoded)

with open("./do/data/output/model/label_encoder.json", "w") as f:
    json.dump(label_encoder.classes_.tolist(), f, indent=4)

#評価可視化する関数
def evaluate_model_with_visualization(
    X, y, subjects, model_path, label_encoder_path, output_path
):
    # モデルとエンコーダの読み込み
    model = load_model(model_path)
    with open(label_encoder_path, "r") as f:
        label_encoder_classes = json.load(f)
    
    X = np.array(X)
    y = np.array(y)

    # 予測の生成
    predictions = model.predict(X)
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(y, axis=1)

    # 混同行列の生成
    cm = confusion_matrix(true_labels, predicted_labels)

    # 混同行列のプロットと保存
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder_classes, yticklabels=label_encoder_classes)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.tight_layout()
    plt.savefig(f"{output_path}/confusion_matrix.png")
    plt.close()

    # 混同行列のサイズを取得
    num_classes = cm.shape[0]

    # インデックスが範囲内であるかを確認
    if len(label_encoder_classes) == num_classes:
        label_accuracy = {
            label: {
                "correct": int(cm[i, i]),
                "total": int(sum(cm[i])) ,
                "accuracy": float(cm[i, i] / sum(cm[i]) if sum(cm[i]) > 0 else 0),
            }
            for i, label in enumerate(label_encoder_classes)
        }
    else:
        print(f"Warning: Mismatch between number of classes in confusion matrix and label encoder. Classes in cm: {num_classes}, in encoder: {len(label_encoder_classes)}")

    # 各ラベルの精度計算
    label_accuracy = {
        label: {
            "correct": int(cm[i, i]),
            "total": int(sum(cm[i])) ,
            "accuracy": float(cm[i, i] / sum(cm[i]) if sum(cm[i]) > 0 else 0),
        }
        for i, label in enumerate(label_encoder_classes)
    }

    # ラベルごとの平均と分散を計算
    accuracies = [data["accuracy"] for data in label_accuracy.values()]
    mean_accuracy = float(np.mean(accuracies))
    variance_accuracy = float(np.var(accuracies))

    # Top-10 予測結果の準備
    top_10_predictions = [
        {
            "true_label": label_encoder_classes[true_label],
            "top_10_labels": [
                {"label": label_encoder_classes[i], "score": float(pred[i])}
                for i in np.argsort(pred)[-10:][::-1]
            ],
        }
        for true_label, pred in zip(true_labels, predictions)
    ]

    # 各ラベルごとのスコア平均と分散
    top_10_stats = {}
    for entry in top_10_predictions:
        for top_label in entry["top_10_labels"]:
            label = top_label["label"]
            score = top_label["score"]
            if label not in top_10_stats:
                top_10_stats[label] = []
            top_10_stats[label].append(score)

    top_10_mean_variance = {
        label: {
            "mean_score": float(np.mean(scores)),
            "variance_score": float(np.var(scores)),
            "count": len(scores),
        }
        for label, scores in top_10_stats.items()
    }

    # 結果の保存
    with open(f"{output_path}/results.json", "w") as f:
        json.dump({
            "mean_accuracy": mean_accuracy,
            "variance_accuracy": variance_accuracy,
            "label_accuracy": label_accuracy,
            "top_10_mean_variance": top_10_mean_variance,
        }, f, indent=4)

    # Top-10 結果の別ファイルへの保存
    with open(f"{output_path}/top_10_predictions.json", "w") as f:
        json.dump(top_10_predictions, f, indent=4)

    print(f"Confusion matrix saved to {output_path}/confusion_matrix.png")
    print(f"Results saved to {output_path}/results.json")
    print(f"Top-10 predictions saved to {output_path}/top_10_predictions.json")


# 修正: `trained_model.h5` ではなく、交差検証用のモデルファイルを使用
# fold_info.json からモデルパスを取得し、各foldのモデルを評価に使用
with open('./do/data/output/model/fold_info.json', 'r', encoding='utf-8') as f:
    fold_model_paths = json.load(f)

for fold, (model_path, test_index) in enumerate(fold_model_paths):
    print(f"Evaluating fold {fold + 1}")
    evaluate_model_with_visualization(
        X=angles,
        y=labels_one_hot,
        subjects=subjects,
        model_path=model_path,  # 各foldごとのモデルを使用
        label_encoder_path="./do/data/output/model/label_encoder.json",
        output_path=f"./do/data/output/valuation/fold_{fold + 1}"
    )
