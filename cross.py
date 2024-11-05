import json
import numpy as np
import os
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.utils import to_categorical
from scipy import stats

# モデルの構築
def create_model(input_shape, num_classes):
    model = Sequential()
    model.add(Input(shape=input_shape))  # Inputレイヤーを追加
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))             # ドロップアウト層を追加
    model.add(Dense(num_classes, activation='softmax'))
    return model

def main():
    # JSONファイルが保存されているディレクトリのパス
    data_dir = r'./do/data/output/angle'
    
    # データの準備
    right_hand_angles = []
    left_hand_angles = []
    labels = []  # ファイル名から抽出した手形番号をラベルとして格納

    # ディレクトリ内のすべてのJSONファイルを処理
    for file_name in os.listdir(data_dir):
        if file_name.endswith('.json'):
            # ファイル名から手形番号（01, 02, ..., 64）をラベルとして抽出
            hand_shape_label = file_name.split('_')[1].split('.')[0]  # 例: '01' を抽出
            with open(os.path.join(data_dir, file_name), 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 右手と左手の角度を取得
            for entry in data:
                right_hand_angles.append(entry['right_hand_angle'])
                left_hand_angles.append(entry['left_hand_angle'])
                labels.append(hand_shape_label)  # 手形番号をラベルとして追加

    # numpy配列に変換
    X_right = np.array(right_hand_angles)
    X_left = np.array(left_hand_angles)
    Y = np.array(labels)

    # ラベルエンコーディング（手形番号を数値に変換）
    label_encoder = LabelEncoder()
    Y_encoded = label_encoder.fit_transform(Y)  # 手形番号をエンコード
    Y_one_hot = to_categorical(Y_encoded)  # ワンホットエンコーディング

    # k分割交差検証の設定
    kfold = KFold(n_splits=5, shuffle=True)

    # 結果を保存するための辞書
    results = {'right_hand': [], 'left_hand': []}
    label_accuracy = {label: [] for label in label_encoder.classes_}  # 各ラベルごとの精度を保存する辞書

    # 右手と左手で別々にモデルを作成・検証
    for hand, X in zip(['right_hand', 'left_hand'], [X_right, X_left]):
        print(f'Processing {hand} angles...')
        accuracies = []

        for train_index, val_index in kfold.split(X):
            X_train_k, X_val_k = X[train_index], X[val_index]
            Y_train_k, Y_val_k = Y_one_hot[train_index], Y_one_hot[val_index]

            # モデルの構築と訓練
            model = create_model(input_shape=(X.shape[1],), num_classes=Y_one_hot.shape[1])  # input_shapeを修正
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            model.fit(X_train_k, Y_train_k, epochs=100, batch_size=16, verbose=0)

            # バリデーションセットでの評価
            score = model.evaluate(X_val_k, Y_val_k, verbose=0)[1]
            print(f'Validation accuracy for {hand}: {score*100}%')
            accuracies.append(score)

            # 各ラベルの精度を計算
            Y_val_encoded = np.argmax(Y_val_k, axis=1)
            predictions = np.argmax(model.predict(X_val_k), axis=1)

            for label in label_encoder.classes_:
                label_indices = np.where(Y_val_encoded == label_encoder.transform([label])[0])[0]
                label_accuracy[label].append(np.mean(predictions[label_indices] == label_encoder.transform([label])[0]))

        # 平均精度と標準偏差の計算
        mean_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)

        # 95%信頼区間
        confidence_interval = stats.t.interval(0.95, len(accuracies)-1, loc=mean_accuracy, scale=stats.sem(accuracies))

        # 結果を保存
        results[hand].append({
            'mean_accuracy': mean_accuracy,
            'std_accuracy': std_accuracy,
            'confidence_interval': [confidence_interval[0], confidence_interval[1]]
        })

    # ラベルごとの精度を追加
    results['label_accuracy'] = {label: np.mean(accuracies) for label, accuracies in label_accuracy.items()}

    # 結果をJSON形式で保存（outputディレクトリ内に保存）
    output_dir = './do/data/output'  # outputディレクトリを指定
    os.makedirs(output_dir, exist_ok=True)  # ディレクトリが存在しない場合は作成
    output_path = os.path.join(output_dir, "crossValidationResults.json")

    with open(output_path, "w", encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

if __name__ == '__main__':
    main()
