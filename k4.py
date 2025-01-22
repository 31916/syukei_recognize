import os
import json
from collections import defaultdict

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def normalize_labels(label, label_mapping):
    """Normalize labels using the mapping from label_encoder.json to normalized_label_encoder.json."""
    for normalized_label in label_mapping:
        if label.startswith(normalized_label):
            return normalized_label
    return label

def process_fold(fold_dir, label_mapping):
    """Process a single fold's evaluation data."""
    # Load data
    top_10_predictions = load_json(os.path.join(fold_dir, 'top_10_predictions.json'))
    results = load_json(os.path.join(fold_dir, 'results.json'))

    # Normalize results
    normalized_results = {
        "mean_accuracy": 0,
        "variance_accuracy": 0,
        "label_accuracy": defaultdict(lambda: {"correct": 0, "total": 0, "accuracy": 0.0})
    }

    for label, stats in results["label_accuracy"].items():
        normalized_label = normalize_labels(label, label_mapping)
        normalized_results["label_accuracy"][normalized_label]["correct"] += stats["correct"]
        normalized_results["label_accuracy"][normalized_label]["total"] += stats["total"]

    for label, stats in normalized_results["label_accuracy"].items():
        stats["accuracy"] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0

    normalized_results["mean_accuracy"] = sum(
        stats["accuracy"] for stats in normalized_results["label_accuracy"].values()
    ) / len(normalized_results["label_accuracy"])

    # Normalize top-10 predictions
    normalized_top_10 = []
    for prediction in top_10_predictions:
        true_label = normalize_labels(prediction["true_label"], label_mapping)
        normalized_top_10_labels = []

        for label_data in prediction["top_10_labels"]:
            label = normalize_labels(label_data["label"], label_mapping)
            score = label_data["score"]

            # Aggregate scores for identical normalized labels
            found = False
            for entry in normalized_top_10_labels:
                if entry["label"] == label:
                    entry["score"] += score
                    found = True
                    break

            if not found:
                normalized_top_10_labels.append({"label": label, "score": score})

        # Normalize scores to percentage
        total_score = sum(item["score"] for item in normalized_top_10_labels)
        for item in normalized_top_10_labels:
            item["score"] = (item["score"] / total_score) * 100 if total_score > 0 else 0

        normalized_top_10.append({
            "true_label": true_label,
            "top_10_labels": sorted(normalized_top_10_labels, key=lambda x: x["score"], reverse=True)
        })

    return normalized_results, normalized_top_10

def integrate_folds(valuation_dir, output_dir, label_mapping):
    """Integrate all folds' evaluation data."""
    os.makedirs(output_dir, exist_ok=True)

    all_results = []
    all_top_10 = []

    for fold_num in range(1, 6):
        fold_dir = os.path.join(valuation_dir, f'fold_{fold_num}')
        fold_results, fold_top_10 = process_fold(fold_dir, label_mapping)

        all_results.append(fold_results)
        all_top_10.extend(fold_top_10)

        # Save fold results
        save_json(fold_results, os.path.join(fold_dir, 'rl_result.json'))
        save_json(fold_top_10, os.path.join(fold_dir, 'rl_top10prediction.json'))

    # Aggregate results across all folds
    integrated_results = {
        "mean_accuracy": sum(result["mean_accuracy"] for result in all_results) / len(all_results),
        "label_accuracy": defaultdict(lambda: {"correct": 0, "total": 0, "accuracy": 0.0})
    }

    for result in all_results:
        for label, stats in result["label_accuracy"].items():
            integrated_results["label_accuracy"][label]["correct"] += stats["correct"]
            integrated_results["label_accuracy"][label]["total"] += stats["total"]

    for label, stats in integrated_results["label_accuracy"].items():
        stats["accuracy"] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0

    # Save integrated results
    save_json(integrated_results, os.path.join(output_dir, 'rl_final_result.json'))
    save_json(all_top_10, os.path.join(output_dir, 'rl_final_top10prediction.json'))

if __name__ == '__main__':
    valuation_dir = './do/data/output/valuation'
    output_dir = './do/data/output/valuation'

    # Load label mappings
    original_labels = load_json('./do/data/output/model/label_encoder.json')
    normalized_labels = load_json('./do/data/output/model/normalized_label_encoder.json')

    # 修正されたラベルマッピング生成部分
    label_mapping = {}

    for original_label in original_labels:
        # Remove the 'l' or 'r' suffix from the label to match normalized labels
        normalized_label = original_label[:-1]  # Exclude the last character ('l' or 'r')
        if normalized_label in normalized_labels:
            label_mapping[original_label] = normalized_label
        else:
            print(f"Warning: Normalized label not found for {original_label}")


    # Process and integrate
    integrate_folds(valuation_dir, output_dir, label_mapping)