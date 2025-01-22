### 交差検証を行うプログラム

import os
import json
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from learn import prepare_data, create_model, clean_data

# データの準備
data_dir = './do/data/output/hand_info'
output_dir = './do/data/output/model'

# データを準備
X, Y, groups = prepare_data(
    data_dir='./do/data/output/hand_info',
    random_seed=42,
    save_selected=True,
    selected_data_path='./do/data/output/model/selected_data.json'
)

# ラベルエンコーディング
label_encoder = LabelEncoder()
Y_encoded = label_encoder.fit_transform(Y)
Y_one_hot = to_categorical(Y_encoded)

# 保存ディレクトリ作成
os.makedirs(output_dir, exist_ok=True)
with open(os.path.join(output_dir, 'label_encoder.json'), 'w', encoding='utf-8') as f:
    json.dump(label_encoder.classes_.tolist(), f, ensure_ascii=False, indent=4)

# 5分割交差検証
kf = GroupKFold(n_splits=5)
fold_model_paths = []

for fold, (train_index, test_index) in enumerate(kf.split(X, Y, groups=groups)):
    print(f"Starting fold {fold + 1}")

    # データの分割
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y_one_hot[train_index], Y_one_hot[test_index]

    # モデル作成とトレーニング
    model = create_model(input_shape=(X_train.shape[1],), num_classes=Y_one_hot.shape[1])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(
        clean_data(X_train), clean_data(Y_train), epochs=100, batch_size=16, verbose=1
    )

    # モデル保存
    fold_model_path = os.path.join(output_dir, f'model_fold_{fold + 1}.h5')
    model.save(fold_model_path)

    # 修正: NumPy 配列をリスト形式に変換して保存
    fold_model_paths.append((fold_model_path, test_index.tolist()))

print("被験者ごとの5分割交差検証が完了しました。")

# モデルパスとテストデータインデックスを保存
with open(os.path.join(output_dir, 'fold_info.json'), 'w', encoding='utf-8') as f:
    json.dump(fold_model_paths, f, ensure_ascii=False, indent=4)


