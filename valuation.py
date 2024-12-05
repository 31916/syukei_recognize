import os
import json
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from scipy import stats
import learn as l

def convert_to_serializable(obj):
    if isinstance(obj, (np.ndarray, list)):  # NumPy配列やリストをリストに変換
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):  # NumPyの浮動小数点数をPythonのfloatに変換
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):  # NumPyの整数をPythonのintに変換
        return int(obj)
    elif obj is None:  # Noneをそのまま返す
        return None
    raise TypeError(f"Type {type(obj)} not serializable")  # 未対応の型の場合はエラー

# データ前処理: NaNやInfを除去または置き換え
def clean_data(array):
    return np.where(np.isnan(array) | np.isinf(array), 0, array)

# 信頼区間の計算
def calculate_confidence_interval(data, confidence=0.95):
    if len(data) < 2 or np.any(np.isnan(data)) or np.any(np.isinf(data)):
        return [None, None]
    std_dev = np.std(data)
    if std_dev == 0:
        return [None, None]

    mean = np.mean(data)
    interval = stats.t.interval(confidence, len(data)-1, loc=mean, scale=stats.sem(data))
    return [interval[0], interval[1]]

def evaluate_model(X, Y, groups, model_path, label_encoder_path):
    with open(label_encoder_path, 'r', encoding='utf-8') as f:
        classes = json.load(f)

    group_kfold = GroupKFold(n_splits=5)
    model = load_model(model_path)

    Y_encoded = np.array([classes.index(label) for label in Y])
    Y_one_hot = to_categorical(Y_encoded, num_classes=len(classes))

    accuracies = []
    for train_index, val_index in group_kfold.split(X, Y_one_hot, groups):
        X_train_k, X_val_k = clean_data(X[train_index]), clean_data(X[val_index])
        Y_train_k, Y_val_k = clean_data(Y_one_hot[train_index]), clean_data(Y_one_hot[val_index])

        Y_val_pred = model.predict(X_val_k, verbose=0)
        val_accuracy = accuracy_score(np.argmax(Y_val_k, axis=1), np.argmax(Y_val_pred, axis=1))
        accuracies.append(val_accuracy)

    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    confidence_interval = calculate_confidence_interval(accuracies)

    results = {
        'mean_accuracy': mean_accuracy,
        'std_accuracy': std_accuracy,
        'confidence_interval': confidence_interval,
        'fold_accuracies': accuracies
    }

    output_dir = './do/data/output'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "results.json")

    with open(output_path, "w", encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False, default=convert_to_serializable)

if __name__ == '__main__':
    data_dir = r'./do/data/output/hand_info'
    model_path = r'./do/data/output/trained_model.h5'
    label_encoder_path = r'./do/data/output/label_encoder.json'

    angles, labels, subjects = l.prepare_data(data_dir)
    evaluate_model(np.array(angles), np.array(labels), np.array(subjects), model_path, label_encoder_path)
