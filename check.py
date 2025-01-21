# ラベル統合処理の検証
Y = ["01l", "01r", "02l", "02r", "03"]  # サンプルデータ
Y_normalized = [
    label[:-1] if label[-1] in ['l', 'r'] else label for label in Y
]
print("ラベル統合処理結果:", Y_normalized)

# 正解判定ロジックの検証
# サンプルデータで検証
predictions = [
    {"true_label": "01", "top_10_labels": [{"label": "01"}, {"label": "02"}]},
    {"true_label": "02", "top_10_labels": [{"label": "03"}, {"label": "02"}]},
    {"true_label": "03", "top_10_labels": [{"label": "03"}, {"label": "01"}]},
]

correct_predictions = 0
for prediction in predictions:
    if prediction["true_label"] == prediction["top_10_labels"][0]["label"]:
        correct_predictions += 1

print("正解数:", correct_predictions, "/", len(predictions))

# モデル出力次元とラベルエンコーダの整合性確認
num_classes_model = 59  # 仮定値
num_classes_encoder = len(set(Y_normalized))  # 統合されたラベル数
print("モデル出力次元:", num_classes_model)
print("ラベル数:", num_classes_encoder)
print("一致確認:", num_classes_model == num_classes_encoder)
