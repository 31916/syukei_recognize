from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import load_model

# データの準備
sys.path.append("./do/matura_arbeit/learn.py")
from learn import prepare_data

data_dir = "./do/data/output/hand_info"  # データが格納されているディレクトリ
angles, labels, subjects = prepare_data(data_dir)

# ラベルのエンコーディング
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
labels_one_hot = to_categorical(labels_encoded)

# 必要に応じてエンコーダーを保存
with open("./do/data/output/label_encoder.json", "w") as f:
    json.dump(label_encoder.classes_.tolist(), f, indent=4)

def evaluate_model_with_visualization(
    X=angles,  # 準備されたデータ
    y=labels_one_hot,  # One-hot エンコーディングされたラベル
    subjects=subjects,  # 被験者の情報
    model_path="./do/data/output/trained_model.h5",  # 学習済みモデルのパス
    label_encoder_path="./do/data/output/label_encoder.json",  # ラベルエンコーダーのパス
    output_path="./do/data/output"  # 出力先ディレクトリ
):
    # モデルとラベルエンコーダーを読み込み
    model = load_model(model_path)
    with open(label_encoder_path, "r") as f:
        label_encoder_classes = json.load(f)
    
    # X と y が numpy 配列であることを確認
    X = np.array(X)
    y = np.array(y)

    # デバッグ情報を表示
    print(f"Shape of y before processing: {y.shape}")
    print(f"Sample of y before encoding: {y[:5]}")

    # ラベルを整数にエンコード
    y_encoded = label_encoder.transform(labels)  # labels を数値に変換
    print(f"Sample of y after encoding: {y_encoded[:5]}")

    # ワンホットエンコーディング
    num_classes = len(label_encoder_classes)
    y_one_hot = to_categorical(y_encoded, num_classes=num_classes)
    print(f"Shape of y after one-hot encoding: {y_one_hot.shape}")

    # 予測値の生成
    predictions = model.predict(X)
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(y_one_hot, axis=1)

    # ラベルをデコード
    true_label_names = [label_encoder_classes[label] for label in true_labels]
    predicted_label_names = [label_encoder_classes[label] for label in predicted_labels]

    # 分類レポートを計算
    class_report = classification_report(true_label_names, predicted_label_names, output_dict=True)

    # 混同行列の作成
    cm = confusion_matrix(true_label_names, predicted_label_names, labels=label_encoder_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder_classes)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig(f"{output_path}/confusion_matrix.png")
    plt.close()

    # ラベルごとの精度を保存
    label_accuracy = {
        label: {
            "correct": cm[i, i],
            "total": sum(cm[i]),
            "accuracy": cm[i, i] / sum(cm[i]) if sum(cm[i]) > 0 else 0
        }
        for i, label in enumerate(label_encoder_classes)
    }
    
    # ラベルごとの精度を視覚化
    accuracies = [value["accuracy"] for value in label_accuracy.values()]
    plt.bar(label_encoder_classes, accuracies)
    plt.xticks(rotation=90)
    plt.title("Label-wise Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Labels")
    plt.tight_layout()
    plt.savefig(f"{output_path}/label_accuracy.png")
    plt.close()

    # TOP3予測結果を保存
    top3_predictions = []
    for i, probs in enumerate(predictions):
        top3_indices = np.argsort(probs)[::-1][:3]
        top3_labels = [label_encoder_classes[idx] for idx in top3_indices]
        top3_probs = [probs[idx] for idx in top3_indices]
        top3_predictions.append({
            "true_label": true_label_names[i],
            "top3_predictions": [
                {"label": label, "probability": float(prob)}
                for label, prob in zip(top3_labels, top3_probs)
            ]
        })
    
    # 結果をJSONファイルに保存
    results = {
        "mean_accuracy": np.mean(accuracies),
        "std_accuracy": np.std(accuracies),
        "label_accuracy": label_accuracy,
        "top3_predictions": top3_predictions
    }

    # JSONファイルに保存する際に int64 を int に変換
    def convert_int64(obj):
        if isinstance(obj, np.int64):
            return int(obj)  # np.int64 を int に変換
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

    with open(f"{output_path}/results.json", "w") as f:
        json.dump(results, f, indent=4, default=convert_int64)
    
    print(f"Evaluation completed. Results saved to {output_path}/results.json")

# 使用例
evaluate_model_with_visualization(
    X=angles,
    y=labels,
    subjects=subjects,
    model_path="./do/data/output/trained_model.h5",  # 学習済みモデルのパス
    label_encoder_path="./do/data/output/label_encoder.json",  # ラベルエンコーダーのパス
    output_path="./do/data/output"  # 出力先ディレクトリ
)
