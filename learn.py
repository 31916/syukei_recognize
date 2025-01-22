import os
import json
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupKFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
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


# データ準備
def prepare_data(data_dir, random_seed=42, save_selected=False, selected_data_path=None):
    """
    データ準備と固定化されたランダム選択を行う。
    必要に応じて選択データを保存または再利用。

    Args:
        data_dir (str): データディレクトリのパス。
        random_seed (int): ランダムシード値（デフォルト: 42）。
        save_selected (bool): 選択したデータを保存するかどうか。
        selected_data_path (str): 保存または読み込み用のパス。

    Returns:
        X (np.array): 特徴量。
        Y (np.array): ラベル。
        groups (np.array): グループ（被験者 ID）。
    """
    np.random.seed(random_seed)  # シードを固定してランダム性を再現可能に
    angles, labels, subjects = [], [], []
    excluded_labels = {"09", "17", "18", "28", "64"}  # 除外するラベル

    selected_data = {}

    if selected_data_path and os.path.exists(selected_data_path):
        # 保存されたデータをロード
        with open(selected_data_path, 'r', encoding='utf-8') as f:
            selected_data = json.load(f)
    else:
        # データの選択と準備
        for file_name in os.listdir(data_dir):
            if file_name.endswith('.json'):
                hand_shape_label = file_name.split('_')[1].split('.')[0]
                subject_id = file_name.split('_')[0]

                # 除外ラベルをスキップ
                if hand_shape_label[:2] in excluded_labels:
                    continue

                with open(os.path.join(data_dir, file_name), 'r', encoding='utf-8') as f:
                    data = json.load(f)

                if not data or any('angles' not in entry or 'rl' not in entry for entry in data):
                    continue

                # 右手と左手のデータを分けて処理
                right_hand_data = [entry for entry in data if entry.get('rl') == 1]
                left_hand_data = [entry for entry in data if entry.get('rl') == 0]

                selected_data[file_name] = {
                    "right": np.random.choice(right_hand_data, 5, replace=False).tolist()
                    if len(right_hand_data) >= 5 else right_hand_data,
                    "left": np.random.choice(left_hand_data, 5, replace=False).tolist()
                    if len(left_hand_data) >= 5 else left_hand_data,
                }

        # 選択データを保存
        if save_selected and selected_data_path:
            with open(selected_data_path, 'w', encoding='utf-8') as f:
                json.dump(selected_data, f, indent=4)

    # データの整形
    for file_name, data in selected_data.items():
        for entry in data["right"] + data["left"]:
            if entry.get('angles') is not None and not np.any(np.isnan(entry.get('angles'))):
                angles.append(entry['angles'])
                labels.append(file_name.split('_')[1].split('.')[0])  # ラベル
                subjects.append(file_name.split('_')[0])  # 被験者 ID

    X = np.array(angles)
    Y = np.array(labels)
    groups = np.array(subjects)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    return X, Y, groups


# モデルのトレーニングと保存
def train_and_save_fold_models(X, Y, groups, output_dir):
    label_encoder = LabelEncoder()
    Y_encoded = label_encoder.fit_transform(Y)
    Y_one_hot = to_categorical(Y_encoded)

    # FOLDごとの分割
    kf = GroupKFold(n_splits=5)
    for fold, (train_index, test_index) in enumerate(kf.split(X, Y, groups=groups)):
        print(f"Training fold {fold + 1}")

        # データの分割
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y_one_hot[train_index], Y_one_hot[test_index]

        # モデル作成
        model = create_model(input_shape=(X.shape[1],), num_classes=Y_one_hot.shape[1])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # EarlyStoppingとModelCheckpointの設定
        fold_model_path = os.path.join(output_dir, f'model_fold_{fold + 1}.h5')
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model_checkpoint = ModelCheckpoint(
            fold_model_path, monitor='val_loss', save_best_only=True, verbose=1
        )

        # モデルのトレーニング
        model.fit(
            clean_data(X_train), clean_data(Y_train),
            validation_data=(clean_data(X_test), clean_data(Y_test)),
            epochs=100, batch_size=16, verbose=1,
            callbacks=[early_stopping, model_checkpoint]
        )

    # ラベルエンコーダーを保存
    with open(os.path.join(output_dir, 'label_encoder.json'), 'w', encoding='utf-8') as f:
        json.dump(label_encoder.classes_.tolist(), f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    data_dir = r'./do/data/output/hand_info'
    output_dir = r'./do/data/output/model'

    # データの準備
    X, Y, groups = prepare_data(data_dir)

    # 各FOLDのモデルをトレーニングおよび保存
    train_and_save_fold_models(X, Y, groups, output_dir)
