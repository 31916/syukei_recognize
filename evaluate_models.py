"""学習時に保存した標本・検証インデックスを使って、各 fold の予測を評価する。"""

import argparse
from pathlib import Path

import numpy as np

from training_utils import DEFAULT_OUTPUT_DIR, prepare_data, read_json, write_json


def build_prediction_records(probabilities, true_labels, classes):
    """Softmax 確率を上位10件まで保存する（logit ではない）。"""
    probabilities = np.asarray(probabilities)
    classes = np.asarray(classes)
    if probabilities.shape != (len(true_labels), len(classes)):
        raise ValueError("Model output shape does not match the saved labels.")
    records = []
    for true_label, row in zip(true_labels, probabilities):
        indices = np.argsort(row)[::-1][:10]
        records.append({
            "true_label": str(true_label),
            "top_10": [
                {"label": str(classes[index]), "probability": float(row[index])}
                for index in indices
            ],
        })
    return records


def accuracy_report(records, classes):
    """クラス別正解率と、その単純平均・母分散を計算する。"""
    counts = {str(label): {"correct": 0, "total": 0} for label in classes}
    for entry in records:
        stats = counts[entry["true_label"]]
        stats["total"] += 1
        stats["correct"] += int(entry["top_10"][0]["label"] == entry["true_label"])
    for stats in counts.values():
        stats["accuracy"] = stats["correct"] / stats["total"] if stats["total"] else 0.0
    accuracies = [stats["accuracy"] for stats in counts.values() if stats["total"]]
    if not accuracies:
        raise ValueError("No predictions to evaluate.")
    return {
        "mean_accuracy": float(np.mean(accuracies)),
        "variance_accuracy": float(np.var(accuracies)),
        "accuracy_per_label": counts,
    }


def evaluate(model_dir, output_dir):
    from tensorflow.keras.models import load_model

    selected_path = model_dir / "selected_data.json"
    if not selected_path.is_file():
        raise FileNotFoundError(f"Training sample snapshot is required: {selected_path}")
    # 元の特徴量ファイルが変更されても、学習時の行番号と標本を保つ。
    features, labels, _ = prepare_data(None, selected_data_path=selected_path)
    classes = read_json(model_dir / "label_encoder.json")
    fold_info = read_json(model_dir / "fold_info.json")
    all_indices = [index for _, indices in fold_info for index in indices]
    if (
        any(type(index) is not int for index in all_indices)
        or sorted(all_indices) != list(range(len(features)))
    ):
        raise ValueError("Fold indices must cover every saved sample exactly once.")
    if not set(labels).issubset(classes):
        raise ValueError("Saved samples contain labels not present in label_encoder.json.")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Use an empty evaluation directory: {output_dir}")

    for fold, (model_name, indices) in enumerate(fold_info, start=1):
        print(f"Evaluating fold {fold}/{len(fold_info)}")
        model = load_model(model_dir / model_name, compile=False)
        probabilities = model.predict(features[indices], verbose=0)
        records = build_prediction_records(probabilities, labels[indices], classes)
        fold_dir = output_dir / f"fold_{fold}"
        write_json(fold_dir / "top_10_predictions.json", records)
        write_json(fold_dir / "results.json", accuracy_report(records, classes))
    print(f"Evaluation completed: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate each fold using the saved training sample snapshot.")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "model")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "evaluation")
    args = parser.parse_args()
    evaluate(args.model_dir, args.output_dir)


if __name__ == "__main__":
    main()
