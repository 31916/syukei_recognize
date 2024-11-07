import json
import numpy as np
import os
from sklearn.model_selection import KFold, GroupKFold
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.utils import to_categorical
from scipy import stats

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

def main():
    # JSONファイルが保存されているディレクトリのパス
    data_dir = r'./do/data/output/angle'
    
    # データの準備
    right_hand_angles = []
    left_hand_angles = []
    labels = []  # ファイル名から抽出した手形番号をラベルとして格納
    subjects = []  # 被験者IDを格納するリスト

    # ディレクトリ内のすべてのJSONファイルを処理
    for file_name in os.listdir(data_dir):
        if file_name.endswith('.json'):
            # ファイル名から手形番号と被験者IDを抽出
            hand_shape_label = file_name.split('_')[1].split('.')[0]  # 例: '01' を抽出
            subject_id = file_name.split('_')[0]  # 例: 't1' を抽出
            
            with open(os.path.join(data_dir, file_name), 'r', encoding='utf-8') as f:
                data = json.load(f)

            # データが10個以上ある場合、ランダムに10個選ぶ
            sampled_data = np.random.choice(data, size=10, replace=False).tolist() if len(data) >= 10 else data

            # 右手と左手の角度と被験者IDを取得
            for entry in sampled_data:
                right_hand_angles.append(entry['right_hand_angle'])
                left_hand_angles.append(entry['left_hand_angle'])
                labels.append(hand_shape_label)
                subjects.append(subject_id)  # 被験者IDをリストに追加

    # numpy配列に変換
    X_right = np.array(right_hand_angles)
    X_left = np.array(left_hand_angles)
    Y = np.array(labels)
    groups = np.array(subjects)  # 被験者IDをグループとして使用

    # ラベルエンコーディング（手形番号を数値に変換）
    label_encoder = LabelEncoder()
    Y_encoded = label_encoder.fit_transform(Y)
    Y_one_hot = to_categorical(Y_encoded)

    # GroupKFoldを使用して被験者ごとに分割
    group_kfold = GroupKFold(n_splits=5)

    # 結果を保存するための辞書
    right_hand_results = {'accuracies': [], 'label_accuracy': {}}
    left_hand_results = {'accuracies': [], 'label_accuracy': {}}

    # 右手と左手で別々にモデルを作成・検証
    for hand, X, results in zip(['right_hand', 'left_hand'], [X_right, X_left], [right_hand_results, left_hand_results]):
        print(f'Processing {hand} angles...')  # 右手または左手を処理していることを表示
        accuracies = []

        for fold, (train_index, val_index) in enumerate(group_kfold.split(X, Y_one_hot, groups), 1):
            print(f'  Fold {fold}: Training and evaluating...')  # 現在のfoldの推定処理を表示
            
            X_train_k, X_val_k = X[train_index], X[val_index]
            Y_train_k, Y_val_k = Y_one_hot[train_index], Y_one_hot[val_index]

            # モデルの構築と訓練
            model = create_model(input_shape=(X.shape[1],), num_classes=Y_one_hot.shape[1])
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            model.fit(X_train_k, Y_train_k, epochs=100, batch_size=16, verbose=0)

            # バリデーションセットでの評価
            score = model.evaluate(X_val_k, Y_val_k, verbose=0)[1]
            print(f'*** {hand}  {fold}: {score*100}% ***')
            accuracies.append(score)

            # 各ラベルの精度を計算
            Y_val_encoded = np.argmax(Y_val_k, axis=1)
            predictions = np.argmax(model.predict(X_val_k), axis=1)

            for label in label_encoder.classes_:
                label_indices = np.where(Y_val_encoded == label_encoder.transform([label])[0])[0]
                if label not in results['label_accuracy']:
                    results['label_accuracy'][label] = []
                results['label_accuracy'][label].append(np.mean(predictions[label_indices] == label_encoder.transform([label])[0]))

        # 平均精度と標準偏差の計算
        mean_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)
        confidence_interval = stats.t.interval(0.95, len(accuracies)-1, loc=mean_accuracy, scale=stats.sem(accuracies))

        # 結果を保存
        results['accuracies'].append({
            'mean_accuracy': mean_accuracy,
            'std_accuracy': std_accuracy,
            'confidence_interval': [confidence_interval[0], confidence_interval[1]]
        })

    # 結果をJSON形式で保存
    output_dir = './do/data/output'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "right_hand_accuracy.json")

    with open(output_path, "w", encoding='utf-8') as f:
        json.dump(right_hand_results, f, indent=4, ensure_ascii=False)

    output_path_left = os.path.join(output_dir, "left_hand_accuracy.json")

    with open(output_path_left, "w", encoding='utf-8') as f:
        json.dump(left_hand_results, f, indent=4, ensure_ascii=False)

if __name__ == '__main__':
    main()
