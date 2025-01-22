import json
from collections import defaultdict
import numpy as np

# 各FOLDのファイルパス
top_10_files = [
    './do/data/output/valuation/fold_1/top_10_predictions.json',
    './do/data/output/valuation/fold_2/top_10_predictions.json',
    './do/data/output/valuation/fold_3/top_10_predictions.json',
    './do/data/output/valuation/fold_4/top_10_predictions.json',
    './do/data/output/valuation/fold_5/top_10_predictions.json'
]

results_files = [
    './do/data/output/valuation/fold_1/results.json',
    './do/data/output/valuation/fold_2/results.json',
    './do/data/output/valuation/fold_3/results.json',
    './do/data/output/valuation/fold_4/results.json',
    './do/data/output/valuation/fold_5/results.json'
]

# 左右を無視して数字部分を抽出する関数
def remove_rl(label):
    return label[:-1] if label[-1] in ['l', 'r'] else label

# 集約結果を格納する辞書
top1_correct = defaultdict(lambda: {"correct": 0, "total": 0})
top5_predictions = defaultdict(lambda: defaultdict(float))
aggregated_results = defaultdict(lambda: {"correct": 0, "total": 0})

# TOP1評価とTOP5の集計
for top_10_file in top_10_files:
    with open(top_10_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
        for entry in data:
            true_label = remove_rl(entry['true_label'])
            top_predictions = entry['top_10'][:5]  # TOP5を取得

            # TOP1が正解か判定
            if remove_rl(top_predictions[0]['label']) == true_label:
                top1_correct[true_label]['correct'] += 1
            top1_correct[true_label]['total'] += 1

            # TOP5の確率を集計
            for prediction in top_predictions:
                predicted_label = remove_rl(prediction['label'])
                top5_predictions[true_label][predicted_label] += prediction['probability']

# results.jsonを集約
total_mean_accuracy = 0
mean_accuracies = []  # 追加: mean_accuracy を収集するリスト
for results_file in results_files:
    with open(results_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
        mean_accuracies.append(data['mean_accuracy'])  # 追加: mean_accuracy を収集
        total_mean_accuracy += data['mean_accuracy']
        for label, stats in data['accuracy_per_label'].items():
            label_no_rl = remove_rl(label)
            aggregated_results[label_no_rl]['correct'] += stats['correct']
            aggregated_results[label_no_rl]['total'] += stats['total']

# 平均精度を計算
total_mean_accuracy /= len(results_files)
total_variance = np.var(mean_accuracies)  # 追加: 分散を計算

accuracy_per_label = {
    label: {
        'correct': stats['correct'],
        'total': stats['total'],
        'accuracy': stats['correct'] / stats['total'] if stats['total'] > 0 else 0
    }
    for label, stats in aggregated_results.items()
}

# TOP5結果の整形
consolidated_top5 = []
for true_label, predictions in top5_predictions.items():
    sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
    consolidated_top5.append({
        'true_label': true_label,
        'top_5': [
            {'label': label, 'probability': prob / len(top_10_files)}  # 平均化
            for label, prob in sorted_predictions[:5]
        ]
    })

# TOP1全体のmean_accuracyを計算
total_correct = sum(stats['correct'] for stats in top1_correct.values())
total_count = sum(stats['total'] for stats in top1_correct.values())
top1_mean_accuracy = total_correct / total_count if total_count > 0 else 0

# TOP1結果の整形
top1_accuracy = {
    label: {
        'correct': stats['correct'],
        'total': stats['total'],
        'accuracy': stats['correct'] / stats['total'] if stats['total'] > 0 else 0
    }
    for label, stats in top1_correct.items()
}

# 結果を保存
output_top5 = './do/data/output/valuation/top5_predictions.json'
output_top1 = './do/data/output/valuation/top1_results.json'
output_results = './do/data/output/valuation/consolidated_results.json'

with open(output_top5, 'w', encoding='utf-8') as f:
    json.dump(consolidated_top5, f, indent=4, ensure_ascii=False)

with open(output_top1, 'w', encoding='utf-8') as f:
    json.dump({
        'mean_accuracy': top1_mean_accuracy,
        'accuracy_per_label': top1_accuracy
    }, f, indent=4, ensure_ascii=False)

with open(output_results, 'w', encoding='utf-8') as f:
    json.dump({
        'mean_accuracy': total_mean_accuracy,
        'variance': total_variance,  # 追加: 分散を記録
        'accuracy_per_label': accuracy_per_label
    }, f, indent=4, ensure_ascii=False)

print(f"TOP5予測結果を '{output_top5}' に保存しました。")
print(f"TOP1予測結果を '{output_top1}' に保存しました。")
print(f"全体の結果を '{output_results}' に保存しました。")
print(f"全体の分散: {total_variance}")
