import os
import json
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.utils import to_categorical

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

# データ前処理: NaNやInfを除去または置き換え
def clean_data(array):
    return np.where(np.isnan(array) | np.isinf(array), 0, array)
def prepare_data(data_dir):
    angles, labels, subjects = [], [], []

    for file_name in os.listdir(data_dir):
        if file_name.endswith('.json'):
            hand_shape_label = file_name.split('_')[1].split('.')[0]
            subject_id = file_name.split('_')[0]
            
            with open(os.path.join(data_dir, file_name), 'r', encoding='utf-8') as f:
                data = json.load(f)

            # データが正しく含まれているか確認
            if not data or any('angles' not in entry or 'palm_orientation' not in entry or 'rl' not in entry for entry in data):
                continue

            right_hand_data = [entry for entry in data if entry.get('rl') == 1]
            left_hand_data = [entry for entry in data if entry.get('rl') == 0]

            if right_hand_data:
                sampled_right_hand_data = np.random.choice(right_hand_data, 5, replace=False) if len(right_hand_data) >= 5 else right_hand_data
                for entry in sampled_right_hand_data:
                    right_hand_info = entry.get('angles')

                    if right_hand_info is None or np.any(np.isnan(right_hand_info)):
                        continue

                    angles.append(right_hand_info)
                    labels.append(f"{hand_shape_label}")  # "_right" を削除
                    subjects.append(subject_id)

            if left_hand_data:
                sampled_left_hand_data = np.random.choice(left_hand_data, 5, replace=False) if len(left_hand_data) >= 5 else left_hand_data
                for entry in sampled_left_hand_data:
                    left_hand_info = entry.get('angles')

                    if left_hand_info is None or np.any(np.isnan(left_hand_info)):
                        continue

                    angles.append(left_hand_info)
                    labels.append(f"{hand_shape_label}")  # "_left" を削除
                    subjects.append(subject_id)

    # データをNumPyの配列に変換
    X = np.array(angles)
    Y = np.array(labels)
    groups = np.array(subjects)

    # Xの形状を確認し、必要に応じて整形
    if X.ndim == 1:
        X = X.reshape(-1, 1)  # 角度データが1次元の場合、2次元に変換する

    return X, Y, groups

def train_model(X, Y, output_dir):
    label_encoder = LabelEncoder()
    # 修正: 簡潔なラベルに基づいてエンコード
    Y_encoded = label_encoder.fit_transform(Y)
    Y_one_hot = to_categorical(Y_encoded)

    # モデルの作成
    model = create_model(input_shape=(X.shape[1],), num_classes=Y_one_hot.shape[1])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # 学習の実行
    model.fit(clean_data(X), clean_data(Y_one_hot), epochs=100, batch_size=16, verbose=1)

    # モデルとラベルエンコーダーの保存
    os.makedirs(output_dir, exist_ok=True)
    model.save(os.path.join(output_dir, 'trained_model.h5'))
    with open(os.path.join(output_dir, 'label_encoder.json'), 'w', encoding='utf-8') as f:
        json.dump(label_encoder.classes_.tolist(), f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    data_dir = r'./do/data/output/hand_info'
    output_dir = r'./do/data/output/model'

    # データの準備
    X, Y, groups = prepare_data(data_dir)
    # モデルの学習
    train_model(X, Y, output_dir)