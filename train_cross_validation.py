"""被験者をまたがない GroupKFold で学習し、評価用の標本と分割を保存する。"""

import argparse
from pathlib import Path

import numpy as np

from training_utils import DEFAULT_OUTPUT_DIR, create_model, prepare_data, write_json


def train(data_dir, output_dir, folds=5, epochs=100, batch_size=16, seed=42):
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import LabelEncoder
    from tensorflow.keras import backend, utils
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

    # 実験を混在させない。再実行には別の出力先を指定する。
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Use an empty model directory for a new run: {output_dir}")
    features, labels, subjects = prepare_data(data_dir, random_seed=seed)
    if not 2 <= folds <= len(np.unique(subjects)):
        raise ValueError("--folds must be between 2 and the number of distinct subjects.")
    if epochs < 1 or batch_size < 1:
        raise ValueError("--epochs and --batch-size must be positive.")

    output_dir.mkdir(parents=True, exist_ok=True)
    # 選んだ標本を保存し、以後の評価ではこのファイルだけを読む。
    features, labels, subjects = prepare_data(
        data_dir, random_seed=seed, save_selected=True,
        selected_data_path=output_dir / "selected_data.json",
    )
    encoder = LabelEncoder()
    encoded = encoder.fit_transform(labels)
    targets = utils.to_categorical(encoded, num_classes=len(encoder.classes_))
    write_json(output_dir / "label_encoder.json", encoder.classes_.tolist())
    write_json(output_dir / "run_config.json", {
        "folds": folds, "epochs": epochs, "batch_size": batch_size, "seed": seed,
        "samples": len(features), "subjects": len(np.unique(subjects)),
        "classes": len(encoder.classes_),
    })

    fold_info, best_models = [], {}
    splitter = GroupKFold(n_splits=folds)
    for fold, (train_indices, validation_indices) in enumerate(
        splitter.split(features, labels, groups=subjects), start=1
    ):
        print(f"Training fold {fold}/{folds}")
        backend.clear_session()
        utils.set_random_seed(seed + fold)
        model = create_model((features.shape[1],), len(encoder.classes_))
        model_name = f"best_model_fold_{fold}.keras"
        callbacks = [
            ModelCheckpoint(output_dir / model_name, save_best_only=True, monitor="val_loss"),
            EarlyStopping(monitor="val_loss", patience=30, restore_best_weights=True),
        ]
        history = model.fit(
            features[train_indices], targets[train_indices],
            validation_data=(features[validation_indices], targets[validation_indices]),
            epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=1,
        )
        # モデル名は出力ディレクトリ基準。フォルダを移動しても評価できる。
        fold_info.append([model_name, validation_indices.tolist()])
        best_models[f"fold_{fold}"] = {
            "best_val_loss": float(min(history.history["val_loss"])),
            "best_model_path": model_name,
        }
    # 全 fold が終わった実験だけを評価対象にする。
    write_json(output_dir / "fold_info.json", fold_info)
    write_json(output_dir / "best_models_info.json", best_models)
    print(f"Training completed: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train subject-grouped cross-validation models.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "hand_info")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "model")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args.data_dir, args.output_dir, args.folds, args.epochs, args.batch_size, args.seed)


if __name__ == "__main__":
    main()
