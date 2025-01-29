import os
import json
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

def prepare_data(data_dir, random_seed=42, save_selected=False, selected_data_path=None):
    np.random.seed(random_seed)
    angles, labels, subjects = [], [], []
    excluded_labels = {"09", "17", "18", "28", "64"}  # 除外するラベル

    selected_data = {}
    if selected_data_path and os.path.exists(selected_data_path):
        with open(selected_data_path, 'r', encoding='utf-8') as f:
            selected_data = json.load(f)
    else:
        for file_name in os.listdir(data_dir):
            if file_name.endswith('.json'):
                hand_shape_label = file_name.split('_')[1].split('.')[0]
                if hand_shape_label[:2] in excluded_labels:
                    continue

                with open(os.path.join(data_dir, file_name), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if not data or any('angles' not in entry or 'rl' not in entry for entry in data):
                    continue

                right_hand_data = [entry for entry in data if entry.get('rl') == 1]
                left_hand_data = [entry for entry in data if entry.get('rl') == 0]

                selected_data[file_name] = {
                    "right": np.random.choice(right_hand_data, 5, replace=False).tolist()
                    if len(right_hand_data) >= 5 else right_hand_data,
                    "left": np.random.choice(left_hand_data, 5, replace=False).tolist()
                    if len(left_hand_data) >= 5 else left_hand_data,
                }

        if save_selected and selected_data_path:
            with open(selected_data_path, 'w', encoding='utf-8') as f:
                json.dump(selected_data, f, indent=4)

    for file_name, data in selected_data.items():
        for entry in data["right"] + data["left"]:
            if entry.get('angles') is not None and not np.any(np.isnan(entry.get('angles'))):
                angles.append(entry['angles'])
                labels.append(file_name.split('_')[1].split('.')[0])
                subjects.append(file_name.split('_')[0])

    X = np.array(angles)
    Y = np.array(labels)
    groups = np.array(subjects)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    return X, Y, groups

def clean_data(X, Y):
    valid_indices = ~np.any(np.isnan(X), axis=1)
    X_cleaned = X[valid_indices]
    Y_cleaned = Y[valid_indices]
    return X_cleaned, Y_cleaned

def create_model(input_shape, num_classes):
    model = Sequential()

    # 入力層
    model.add(Dense(128, input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # 隠れ層1
    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))  # 過学習を防ぐ

    # 隠れ層2
    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # 隠れ層3（追加）
    model.add(Dense(128))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.3))

    # 出力層
    model.add(Dense(num_classes, activation='softmax'))

    # コンパイル
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

def train_and_save_fold_models(X, Y, groups, num_folds, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    unique_labels = np.unique(Y)
    num_classes = len(unique_labels)
    input_shape = (X.shape[1],)

    for fold in range(1, num_folds + 1):
        print(f"Starting fold {fold}")
        X_train, X_val, Y_train, Y_val = train_test_split(
            X, Y, test_size=0.2, random_state=42 + fold, stratify=groups
        )

        model = create_model(input_shape, num_classes)

        # 最良モデル保存用
        checkpoint_best = ModelCheckpoint(
            os.path.join(output_dir, f"best_model_fold_{fold}.h5"),
            save_best_only=True, monitor='val_loss', mode='min'
        )

        # 早期停止
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        model.fit(
            X_train, Y_train,
            epochs=150,
            batch_size=32,
            validation_data=(X_val, Y_val),
            callbacks=[checkpoint_best, early_stopping]
        )

        print(f"Fold {fold} completed. Best model saved to best_model_fold_{fold}.h5.")
