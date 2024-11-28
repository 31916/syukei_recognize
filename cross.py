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

# データ前処理: NaNやInfを除去または置き換え
def clean_data(array):
    # NaNやInfを0に置き換える
    return np.where(np.isnan(array) | np.isinf(array), 0, array)

# 平均値を計算する前に、空でないことを確認
def safe_mean(array):
    if np.any(array):  # 配列が空でない場合
        return np.mean(array)
    else:
        return 0  # 配列が空の場合は0を返す（適切な処理に変更可能）

# 除算を安全に行う
def safe_divide(numerator, denominator):
    # ゼロ除算を防ぐために、分母がゼロでないことを確認
    safe_denominator = np.where(denominator == 0, 1, denominator)  # ゼロの分母を1に変更
    return numerator / safe_denominator  # 安全に除算

# 例: トレーニングループ内での精度の計算
def train_model(model, train_data, val_data):
    # トレーニングと検証データの前処理
    train_data = clean_data(train_data)
    val_data = clean_data(val_data)


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
            # ファイル名から手形番号と被験者IDを抽出
            hand_shape_label = file_name.split('_')[1].split('.')[0]  # 例: '01' を抽出
            subject_id = file_name.split('_')[0]  # 例: 't1' を抽出
            
            with open(os.path.join(data_dir, file_name), 'r', encoding='utf-8') as f:
                data = json.load(f)

            # ランダムに20個のデータをサンプリング
            sampled_data = np.random.choice(data, 20, replace=False) if len(data) >= 20 else data

            # 右手と左手の角度を取得
            for entry in sampled_data:
                # 右手の情報を取得
                right_hand_info = entry['angles']
                right_hand_orientation = entry['palm_orientation']
                angles.append(right_hand_info)  # 右手のデータ
                labels.append(f"{hand_shape_label}_right_{right_hand_orientation}")  # 右手ラベル
                subjects.append(subject_id)

                # 左手の情報を取得
                left_hand_info = entry['angles']
                left_hand_orientation = entry['palm_orientation']
                angles.append(left_hand_info)  # 左手のデータ
                labels.append(f"{hand_shape_label}_left_{left_hand_orientation}")  # 左手ラベル
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
    results = {
        'mean_accuracy': None,
        'std_accuracy': None,
        'confidence_interval': None,
        'fold_accuracies': [],
        'hand_shape_accuracies': {}
    }

    print('Training model with combined right and left hand labels...')
    accuracies = []
    
    # 手形ごとの認識精度を格納する辞書
    hand_shape_accuracies = {}
    for i in range(1, 65):
        hand_shape_accuracies[f"{str(i).zfill(2)}r"] = []  # 右手
        hand_shape_accuracies[f"{str(i).zfill(2)}l"] = []  # 左手

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

        # 手形ごとの認識精度を格納
        for i in range(1, 65):
            label_r = f"{str(i).zfill(2)}r"
            label_l = f"{str(i).zfill(2)}l"
            # 右手
            hand_shape_accuracies[label_r].append(np.mean(Y_val_pred[Y_val_k == label_r]))
            # 左手
            hand_shape_accuracies[label_l].append(np.mean(Y_val_pred[Y_val_k == label_l]))

    # 結果を保存
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    confidence_interval = calculate_confidence_interval(accuracies)

    results['mean_accuracy'] = mean_accuracy
    results['std_accuracy'] = std_accuracy
    results['confidence_interval'] = confidence_interval
    results['fold_accuracies'] = accuracies
    results['hand_shape_accuracies'] = hand_shape_accuracies

    # 結果をJSON形式で保存
    output_dir = './do/data/output'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "results.json")

    with open(output_path, "w", encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

if __name__ == '__main__':
    main()
