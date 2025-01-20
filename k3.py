import json
import os
from collections import defaultdict
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 統合された結果を保存するための関数
def integrate_results(valuation_dir, output_dir):
    all_top_10_predictions = []
    all_results = []

    # 5foldの結果を統合
    for fold_num in range(1, 6):
        fold_valuation_path = f"{valuation_dir}/fold_{fold_num}"
        
        # top_10_predictions.json と results.json を読み込む
        with open(f"{fold_valuation_path}/top_10_predictions.json", 'r', encoding='utf-8') as f:
            top_10_predictions = json.load(f)
        
        with open(f"{fold_valuation_path}/results.json", 'r', encoding='utf-8') as f:
            results = json.load(f)

        # top_10_predictions と results を統合
        all_top_10_predictions.extend(top_10_predictions)
        all_results.append(results)

    # 統合された結果をファイルに保存
    with open(f"{output_dir}/final_top_10_predictions.json", 'w', encoding='utf-8') as f:
        json.dump(all_top_10_predictions, f, indent=4)

    # 結果の統計情報を統合
    final_results = {
        "mean_accuracy": sum([result["mean_accuracy"] for result in all_results]) / len(all_results),
        "variance_accuracy": sum([result["variance_accuracy"] for result in all_results]) / len(all_results),
        "label_accuracy": {label: {"correct": 0, "total": 0, "accuracy": 0.0} for label in all_results[0]["label_accuracy"]},
        "top_10_mean_variance": {}
    }

    # 各ラベルごとの統計を計算
    for result in all_results:
        for label, label_data in result["label_accuracy"].items():
            final_results["label_accuracy"][label]["correct"] += label_data["correct"]
            final_results["label_accuracy"][label]["total"] += label_data["total"]
            final_results["label_accuracy"][label]["accuracy"] += label_data["accuracy"]

    # 平均精度、分散精度を再計算
    num_labels = len(final_results["label_accuracy"])
    for label in final_results["label_accuracy"]:
        final_results["label_accuracy"][label]["accuracy"] /= len(all_results)
    
    # top_10_mean_variance を統合
    all_top_10_mean_variance = {}
    for result in all_results:
        for label, stats in result["top_10_mean_variance"].items():
            if label not in all_top_10_mean_variance:
                all_top_10_mean_variance[label] = {"mean_score": 0.0, "variance_score": 0.0, "count": 0}
            all_top_10_mean_variance[label]["mean_score"] += stats["mean_score"]
            all_top_10_mean_variance[label]["variance_score"] += stats["variance_score"]
            all_top_10_mean_variance[label]["count"] += stats["count"]
    
    for label, stats in all_top_10_mean_variance.items():
        stats["mean_score"] /= len(all_results)
        stats["variance_score"] /= len(all_results)

    final_results["top_10_mean_variance"] = all_top_10_mean_variance

    # 統合された結果を保存
    with open(f"{output_dir}/final_results.json", 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=4)

# 統合された結果を保存するための関数
def integrate_results(valuation_dir, output_dir):
    all_top_10_predictions = []
    all_results = []

    # 5foldの結果を統合
    for fold_num in range(1, 6):
        fold_valuation_path = f"{valuation_dir}/fold_{fold_num}"
        
        # top_10_predictions.json と results.json を読み込む
        with open(f"{fold_valuation_path}/top_10_predictions.json", 'r', encoding='utf-8') as f:
            top_10_predictions = json.load(f)
        
        with open(f"{fold_valuation_path}/results.json", 'r', encoding='utf-8') as f:
            results = json.load(f)

        # top_10_predictions と results を統合
        all_top_10_predictions.extend(top_10_predictions)
        all_results.append(results)

    # 統合された結果をファイルに保存
    with open(f"{output_dir}/final_top_10_predictions.json", 'w', encoding='utf-8') as f:
        json.dump(all_top_10_predictions, f, indent=4)

    # 結果の統計情報を統合
    final_results = {
        "mean_accuracy": sum([result["mean_accuracy"] for result in all_results]) / len(all_results),
        "variance_accuracy": sum([result["variance_accuracy"] for result in all_results]) / len(all_results),
        "label_accuracy": {label: {"correct": 0, "total": 0, "accuracy": 0.0} for label in all_results[0]["label_accuracy"]},
        "top_10_mean_variance": {}
    }

    # 各ラベルごとの統計を計算
    for result in all_results:
        for label, label_data in result["label_accuracy"].items():
            final_results["label_accuracy"][label]["correct"] += label_data["correct"]
            final_results["label_accuracy"][label]["total"] += label_data["total"]
            final_results["label_accuracy"][label]["accuracy"] += label_data["accuracy"]

    # 平均精度、分散精度を再計算
    num_labels = len(final_results["label_accuracy"])
    for label in final_results["label_accuracy"]:
        final_results["label_accuracy"][label]["accuracy"] /= len(all_results)
    
    # top_10_mean_variance を統合
    all_top_10_mean_variance = {}
    for result in all_results:
        for label, stats in result["top_10_mean_variance"].items():
            if label not in all_top_10_mean_variance:
                all_top_10_mean_variance[label] = {"mean_score": 0.0, "variance_score": 0.0, "count": 0}
            all_top_10_mean_variance[label]["mean_score"] += stats["mean_score"]
            all_top_10_mean_variance[label]["variance_score"] += stats["variance_score"]
            all_top_10_mean_variance[label]["count"] += stats["count"]
    
    for label, stats in all_top_10_mean_variance.items():
        stats["mean_score"] /= len(all_results)
        stats["variance_score"] /= len(all_results)

    final_results["top_10_mean_variance"] = all_top_10_mean_variance

    # 統合された結果を保存
    with open(f"{output_dir}/final_results.json", 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=4)

# 使用例
valuation_dir = './do/data/output/valuation'  # 評価結果の保存場所
output_dir = './do/data/output/merged_results'  # 統合された結果の保存場所

os.makedirs(output_dir, exist_ok=True)
integrate_results(valuation_dir, output_dir)


def integrate_top_10_predictions(valuation_dir):
    all_top_10_predictions = defaultdict(lambda: defaultdict(float))  # labelごとにスコアを加算する

    # 5foldの結果を統合
    for fold_num in range(1, 6):
        fold_valuation_path = f"{valuation_dir}/fold_{fold_num}"
        
        # top_10_predictions.json を読み込む
        with open(f"{fold_valuation_path}/top_10_predictions.json", 'r', encoding='utf-8') as f:
            top_10_predictions = json.load(f)

        # foldごとのtop_10_predictionsを統合
        for prediction in top_10_predictions:
            true_label = prediction["true_label"]
            top_10_labels = prediction["top_10_labels"]

            # labelごとにスコアを加算
            for label_data in top_10_labels:
                label = label_data["label"]
                score = label_data["score"]
                all_top_10_predictions[true_label][label] += score

    # 統合された結果を整理して1つのリストにまとめる
    final_top_10_predictions = []
    for true_label, label_scores in all_top_10_predictions.items():
        # 各true_labelについて、スコアを合算したラベルを取得
        sorted_labels = sorted(label_scores.items(), key=lambda x: x[1], reverse=True)
        
        # 合計スコアを計算
        total_score = sum([score for label, score in sorted_labels])
        
        # スコアを100%に正規化
        normalized_labels = [{"label": label, "score": min(score / total_score * 100, 100)} for label, score in sorted_labels]

        # 上位10個を選出（スコアが高い順）
        final_top_10_predictions.append({
            "true_label": true_label,
            "top_10_labels": normalized_labels[:10]  # 上位10件
        })

    # 統合された結果を保存（評価結果の保存場所と同じ場所）
    os.makedirs(valuation_dir, exist_ok=True)
    with open(f"{valuation_dir}/final_top_10_predictions.json", 'w', encoding='utf-8') as f:
        json.dump(final_top_10_predictions, f, indent=4)

    print(f"統合されたtop_10_predictionsが{valuation_dir}/final_top_10_predictions.jsonに保存されました。")



# 使用例
valuation_dir = './do/data/output/valuation'  # 評価結果の保存場所
integrate_results(valuation_dir, valuation_dir)
integrate_top_10_predictions(valuation_dir)
