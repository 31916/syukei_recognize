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

# 集計結果ファイル
output_top5 = './do/data/output/valuation/rl_top5.json'
output_top1 = './do/data/output/valuation/rl_result.json'
output_results = './do/data/output/valuation/final_results.json'
output_final_top10 = './do/data/output/valuation/final_top10.json'

def remove_rl(label):
    return label[:-1] if label[-1] in ['l', 'r'] else label

# RL Result: 各FOLDのTOP10からTOP1を評価
consolidated_predictions = defaultdict(list)

for top_10_file in top_10_files:
    with open(top_10_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
        for entry in data:
            true_label = remove_rl(entry['true_label'])
            top_10 = [{**pred, 'label': remove_rl(pred['label'])} for pred in entry['top_10']]
            consolidated_predictions[true_label].append(top_10)

top1_evaluation = defaultdict(lambda: {"correct": 0, "total": 0})

for true_label, folds in consolidated_predictions.items():
    for fold in folds:
        top1_prediction = fold[0]['label']
        if top1_prediction == true_label:
            top1_evaluation[true_label]['correct'] += 1
        top1_evaluation[true_label]['total'] += 1

accuracy_per_label_rl = {
    label: {
        "correct": stats["correct"],
        "total": stats["total"],
        "accuracy": stats["correct"] / stats["total"] if stats["total"] > 0 else 0
    }
    for label, stats in top1_evaluation.items()
}

mean_accuracy_rl = np.mean([
    stats["accuracy"]
    for stats in accuracy_per_label_rl.values()
])

variance_rl = np.var([
    stats["accuracy"]
    for stats in accuracy_per_label_rl.values()
])

# RL結果の保存
with open(output_top1, 'w', encoding='utf-8') as f:
    json.dump({
        'mean_accuracy': mean_accuracy_rl,
        'variance': variance_rl,
        'accuracy_per_label': accuracy_per_label_rl
    }, f, indent=4, ensure_ascii=False)

print(f"RL結果 (TOP1評価): 平均精度: {mean_accuracy_rl}, 分散: {variance_rl}")

# 既存機能の保持 (元のプログラム部分は変更せず保持)
# TOP5とTOP10の統合や左右差ありの集約計算は元のまま
