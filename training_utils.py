"""学習・評価に共通するデータ読み込みと分類モデル。"""

import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "output"
EXCLUDED_LABELS = {"09", "17", "18", "28", "64"}
NUM_ANGLES = 20
SAMPLES_PER_HAND = 5


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)


def parse_filename(filename):
    """subject_01r.json から被験者 ID と左右を含むクラス名を得る。"""
    parts = Path(filename).stem.split("_")
    if len(parts) != 2 or not all(parts):
        raise ValueError(f"Expected <subject>_<label>.json, got: {filename}")
    subject, label = parts
    return subject, label


def has_valid_angles(entry):
    angles = np.asarray(entry.get("angles"), dtype=float)
    return angles.shape == (NUM_ANGLES,) and np.isfinite(angles).all()


def prepare_data(data_dir, random_seed=42, save_selected=False, selected_data_path=None):
    """各ファイルの左右それぞれから最大5件を選び、20角度・ラベル・被験者を返す。

    保存済みの selected_data があれば、その挿入順と標本をそのまま使う。
    fold の行番号はこの順序に依存するため、評価時には再抽出しない。
    """
    selected_path = Path(selected_data_path) if selected_data_path else None
    if selected_path and selected_path.exists():
        selected_data = read_json(selected_path)
    else:
        data_dir = Path(data_dir)
        if not data_dir.is_dir():
            raise FileNotFoundError(f"Hand feature directory not found: {data_dir}")
        # 旧実装と同じ乱数方式を使い、ファイル列挙順だけを固定する。
        rng = np.random.RandomState(random_seed)
        selected_data = {}
        for path in sorted(data_dir.glob("*.json")):
            _, label = parse_filename(path.name)
            if label[:2] in EXCLUDED_LABELS:
                continue
            entries = read_json(path)
            if not entries or any("angles" not in item or "rl" not in item for item in entries):
                continue
            hands = {}
            for name, side in (("right", 1), ("left", 0)):
                candidates = [item for item in entries if item["rl"] == side]
                if len(candidates) >= SAMPLES_PER_HAND:
                    indices = rng.choice(len(candidates), SAMPLES_PER_HAND, replace=False)
                    candidates = [candidates[index] for index in indices]
                hands[name] = candidates
            selected_data[path.name] = hands

        # NaN を含む過去の抽出結果は学習対象にせず、JSON にも書き戻さない。
        for hands in selected_data.values():
            for side in ("right", "left"):
                hands[side] = [entry for entry in hands[side] if has_valid_angles(entry)]
        if save_selected and selected_path:
            write_json(selected_path, selected_data)

    angles, labels, subjects = [], [], []
    for filename, hands in selected_data.items():
        subject, label = parse_filename(filename)
        for entry in hands["right"] + hands["left"]:
            if has_valid_angles(entry):
                angles.append(entry["angles"])
                labels.append(label)
                subjects.append(subject)
    if not angles:
        raise ValueError("No valid 20-angle samples were found. Check the input JSON files.")
    return np.asarray(angles, dtype=float), np.asarray(labels), np.asarray(subjects)


def create_model(input_shape, num_classes):
    """研究時の全結合ネットワーク（128 → 256 → 256 → 128）を構築する。"""
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Activation, BatchNormalization, Dense, Dropout, Input

    model = Sequential([Input(shape=input_shape)])
    for units, dropout in ((128, 0), (256, 0.5), (256, 0.5), (128, 0.3)):
        model.add(Dense(units))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        if dropout:
            model.add(Dropout(dropout))
    model.add(Dense(num_classes, activation="softmax"))
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model
