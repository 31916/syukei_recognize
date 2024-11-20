import json
import numpy as np
import os
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.utils import to_categorical
from scipy import stats
from sklearn.metrics import accuracy_score

# モデルの構築
def create_model(input_shape, num_classes):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model

# データに無効な値が含まれているか確認
def check_invalid_values(data, name="Data"):
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        print(f"Warning: {name} contains NaN or Inf values.")

# 信頼区間の計算（無効な値の場合はスキップ）
def calculate_confidence_interval(data, confidence=0.95):
    if len(data) < 2 or np.any(np.isnan(data)) or np.any(np.isinf(data)):
        print("Insufficient data or invalid values, skipping confidence interval calculation.")
        return [None, None]  # 信頼区間を計算できない場合はNoneを返す

    # 標準誤差が0の場合は信頼区間を計算しない
    std_dev = np.std(data)
    if std_dev == 0:
        print("Standard deviation is 0, skipping confidence interval calculation.")
        return [None, None]

    mean = np.mean(data)
    interval = stats.t.interval(confidence, len(data)-1, loc=mean, scale=stats.sem(data))
    return [interval[0], interval[1]]

def main():
    # JSONファイルが保存されているディレクトリのパス
    data_dir = r'./do/data/output/angle'
    
    # データの準備
    angles = []
    labels = []
    subjects = []

    # ディレクトリ内のすべてのJSONファイルを処理
    for file_name in os.listdir(data_dir):
        if file_name.endswith('.json'):
            # ファイル名から手形番号と被験者IDを抽出
            hand_shape_label = file_name.split('_')[1].split('.')[0]  # 例: '01' を抽出
            subject_id = file_name.split('_')[0]  # 例: 't1' を抽出
            
            with open(os.path.join(data_dir, file_name), 'r', encoding='utf-8') as f:
                data = json.load(f)

            # ランダムに20個のデータをサンプリング
            sampled_data = np.random.choice(data, 20, replace=False) if len(data) >= 20 else data

            # 右手と左手の角度を取得
            for entry in sampled_data:
                angles.append(entry['right_hand_angle'])  # 右手のデータ
                labels.append(f"{hand_shape_label}_right")  # 右手ラベル
                subjects.append(subject_id)

                angles.append(entry['left_hand_angle'])  # 左手のデータ
                labels.append(f"{hand_shape_label}_left")  # 左手ラベル
                subjects.append(subject_id)

    # numpy配列に変換
    X = np.array(angles)
    Y = np.array(labels)
    groups = np.array(subjects)  # 被験者IDをグループとして使用

    # ラベルエンコーディング
    label_encoder = LabelEncoder()
    Y_encoded = label_encoder.fit_transform(Y)
    Y_one_hot = to_categorical(Y_encoded)

    # GroupKFoldを使用して被験者ごとに分割
    group_kfold = GroupKFold(n_splits=5)

    # 結果を保存するための辞書
    results = {'right_hand': {}, 'left_hand': {}, 'mean_accuracy': None, 'std_accuracy': None, 'confidence_interval': None, 'fold_accuracies': []}

    print('Training model with combined right and left hand labels...')
    accuracies = []
    label_accuracies = {'right': {}, 'left': {}}

    # 各foldでの処理
    for fold, (train_index, val_index) in enumerate(group_kfold.split(X, Y_one_hot, groups), 1):
        print(f"  Fold {fold}: Training and evaluating the model...")

        X_train_k, X_val_k = X[train_index], X[val_index]
        Y_train_k, Y_val_k = Y_one_hot[train_index], Y_one_hot[val_index]

        # 訓練データとバリデーションデータで無効な値をチェック
        check_invalid_values(X_train_k, name="X_train_k")
        check_invalid_values(X_val_k, name="X_val_k")
        check_invalid_values(Y_train_k, name="Y_train_k")
        check_invalid_values(Y_val_k, name="Y_val_k")

        # モデルの構築と訓練
        model = create_model(input_shape=(X.shape[1],), num_classes=Y_one_hot.shape[1])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(X_train_k, Y_train_k, epochs=100, batch_size=16, verbose=0)

        # バリデーションセットでの評価
        Y_val_pred = model.predict(X_val_k, verbose=0)
        val_accuracy = accuracy_score(np.argmax(Y_val_k, axis=1), np.argmax(Y_val_pred, axis=1))
        print(f"  Fold {fold}: Validation accuracy: {val_accuracy * 100:.2f}%")
        accuracies.append(val_accuracy)

        # ラベル別の認識率の計算
        for label in ['right', 'left']:
            label_indices = [i for i, y in enumerate(Y[val_index]) if label in y]
            label_true = np.array([Y_val_k[i] for i in label_indices])
            label_pred = np.array([Y_val_pred[i] for i in label_indices])

            label_acc = accuracy_score(np.argmax(label_true, axis=1), np.argmax(label_pred, axis=1))

            # 被験者ごとの精度を記録
            for i, subject in enumerate(np.array(subjects)[val_index][label_indices]):
                hand_label = f"{subject}_{label}"
                if hand_label not in label_accuracies[label]:
                    label_accuracies[label][hand_label] = []
                label_accuracies[label][hand_label].append(label_acc)

    # 結果を保存
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    confidence_interval = calculate_confidence_interval(accuracies)

    results['mean_accuracy'] = mean_accuracy
    results['std_accuracy'] = std_accuracy
    results['confidence_interval'] = confidence_interval
    results['fold_accuracies'] = accuracies

    # ラベル別の精度の平均と分散、信頼区間の計算
    for label in label_accuracies:
        for hand_label in label_accuracies[label]:
            accuracies_for_label = label_accuracies[label][hand_label]
            mean_label_accuracy = np.mean(accuracies_for_label)
            std_label_accuracy = np.std(accuracies_for_label)
            confidence_interval_label = calculate_confidence_interval(accuracies_for_label)
            if label not in results:
                results[label] = {}
            results[label][hand_label] = {
                'mean_accuracy': mean_label_accuracy,
                'std_accuracy': std_label_accuracy,
                'confidence_interval': confidence_interval_label
            }

    # 結果をJSON形式で保存
    output_dir = './do/data/output'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "combined_hand_results.json")

    with open(output_path, "w", encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

if __name__ == '__main__':
    main()
