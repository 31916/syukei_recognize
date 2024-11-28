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
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model

# データに無効な値が含まれているか確認
def check_invalid_values(data, name="Data"):
    try:
        if np.issubdtype(data.dtype, np.number):  # 数値型の場合のみチェック
            if np.any(np.isnan(data)) or np.any(np.isinf(data)):
                print(f"Warning: {name} contains NaN or Inf values.")
        else:
            print(f"Info: {name} is not a numeric array, skipping NaN/Inf check.")
    except AttributeError:
        print(f"Error: {name} is not a NumPy array. Skipping validation.")


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

# データ前処理: NaNやInfを除去または置き換え
def clean_data(array):
    # NaNやInfを0に置き換える
    return np.where(np.isnan(array) | np.isinf(array), 0, array)

# 平均値を計算する前に、空でないことを確認
def safe_mean(array):
    if len(array) > 0:
        return np.mean(array)
    else:
        return 0  # 配列が空の場合は0を返す

# 除算を安全に行う
def safe_divide(numerator, denominator):
    safe_denominator = np.where(denominator == 0, 1, denominator)  # ゼロの分母を1に変更
    return numerator / safe_denominator

# メイン処理
def main():
    # JSONファイルが保存されているディレクトリのパス
    data_dir = r'./do/data/output/hand_info'
    
    # データの準備
    angles = []
    labels = []
    subjects = []

    # ディレクトリ内のすべてのJSONファイルを処理
    for file_name in os.listdir(data_dir):
        if file_name.endswith('.json'):
            hand_shape_label = file_name.split('_')[1].split('.')[0]
            subject_id = file_name.split('_')[0]
            
            with open(os.path.join(data_dir, file_name), 'r', encoding='utf-8') as f:
                data = json.load(f)

            if not data or any('angles' not in entry or 'palm_orientation' not in entry for entry in data):
                print(f"Invalid or empty data in file: {file_name}")
                continue

            sampled_data = np.random.choice(data, 20, replace=False) if len(data) >= 20 else data

            for entry in sampled_data:
                right_hand_info = entry.get('angles')
                right_hand_orientation = entry.get('palm_orientation')

                if right_hand_info is None or np.any(np.isnan(right_hand_info)):
                    print(f"Invalid angles in file: {file_name}")
                    continue

                angles.append(right_hand_info)
                labels.append(f"{hand_shape_label}_right_{right_hand_orientation}")
                subjects.append(subject_id)

                left_hand_info = entry.get('angles')
                left_hand_orientation = entry.get('palm_orientation')

                if left_hand_info is None or np.any(np.isnan(left_hand_info)):
                    print(f"Invalid angles in file: {file_name}")
                    continue

                angles.append(left_hand_info)
                labels.append(f"{hand_shape_label}_left_{left_hand_orientation}")
                subjects.append(subject_id)

    # デバッグ: データ型の確認
    print(f"Type of Y: {type(labels)}, Example: {labels[:5]}")

    # numpy配列に変換
    X = np.array(angles)
    Y = np.array(labels)
    groups = np.array(subjects)

    check_invalid_values(X, name="X")
    check_invalid_values(Y, name="Y")  # 修正済みの関数を使用

    # ラベルエンコーディング
    label_encoder = LabelEncoder()
    Y_encoded = label_encoder.fit_transform(Y)
    Y_one_hot = to_categorical(Y_encoded)

    group_kfold = GroupKFold(n_splits=5)
    # 以下は元のコードの処理に続く...


    results = {
        'mean_accuracy': None,
        'std_accuracy': None,
        'confidence_interval': None,
        'fold_accuracies': [],
        'hand_shape_accuracies': {}
    }

    print('Training model with combined right and left hand labels...')
    accuracies = []

    for fold, (train_index, val_index) in enumerate(group_kfold.split(X, Y_one_hot, groups), 1):
        print(f"  Fold {fold}: Training and evaluating the model...")

        X_train_k, X_val_k = clean_data(X[train_index]), clean_data(X[val_index])
        Y_train_k, Y_val_k = clean_data(Y_one_hot[train_index]), clean_data(Y_one_hot[val_index])

        check_invalid_values(X_train_k, name="X_train_k")
        check_invalid_values(X_val_k, name="X_val_k")
        check_invalid_values(Y_train_k, name="Y_train_k")
        check_invalid_values(Y_val_k, name="Y_val_k")

        model = create_model(input_shape=(X.shape[1],), num_classes=Y_one_hot.shape[1])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(X_train_k, Y_train_k, epochs=100, batch_size=16, verbose=0)

        Y_val_pred = model.predict(X_val_k, verbose=0)
        val_accuracy = accuracy_score(np.argmax(Y_val_k, axis=1), np.argmax(Y_val_pred, axis=1))
        print(f"  Fold {fold}: Validation accuracy: {val_accuracy * 100:.2f}%")
        accuracies.append(val_accuracy)

    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    confidence_interval = calculate_confidence_interval(accuracies)

    results['mean_accuracy'] = mean_accuracy
    results['std_accuracy'] = std_accuracy
    results['confidence_interval'] = confidence_interval
    results['fold_accuracies'] = accuracies

    output_dir = './do/data/output'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "results.json")

    with open(output_path, "w", encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False, default=convert_to_serializable)

if __name__ == '__main__':
    main()
