"""人工データで公開コマンドをつなぐ。実データの認識精度は検証しない。"""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from training_utils import PROJECT_ROOT, prepare_data, read_json, write_json


class PipelineTests(unittest.TestCase):
    def run_script(self, script, cwd, *arguments):
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / script), *map(str, arguments)],
            cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace",
            env={**os.environ, "PYTHONIOENCODING": "utf-8", "TF_CPP_MIN_LOG_LEVEL": "2"},
            timeout=180,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def test_train_evaluate_and_summarize_from_another_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, model_dir = root / "features", root / "models"
            rng = np.random.RandomState(0)
            for subject in ("subject01", "subject02"):
                for label in ("01r", "02l"):
                    entries = [
                        {"frame": index, "angles": rng.uniform(10, 170, 20).tolist(),
                         "yaw": 0, "rl": int(label.endswith("r"))}
                        for index in range(6)
                    ]
                    write_json(source / f"{subject}_{label}.json", entries)
            self.run_script(
                "train_cross_validation.py", root,
                "--data-dir", source, "--output-dir", model_dir,
                "--folds", 2, "--epochs", 1, "--batch-size", 4,
            )
            features, labels, subjects = prepare_data(
                None, selected_data_path=model_dir / "selected_data.json",
            )
            self.assertEqual(len(features), 20)
            folds = read_json(model_dir / "fold_info.json")
            for _, indices in folds:
                train_indices = sorted(set(range(len(features))) - set(indices))
                self.assertFalse(set(subjects[indices]) & set(subjects[train_indices]))

            # 元データがなく、モデルフォルダを移動しても正しい標本を評価する。
            source.rename(root / "unused-source")
            moved_models = root / "moved-models"
            model_dir.rename(moved_models)
            evaluation_dir = root / "evaluation"
            self.run_script(
                "evaluate_models.py", root,
                "--model-dir", moved_models, "--output-dir", evaluation_dir,
            )
            for fold, (_, indices) in enumerate(folds, start=1):
                records = read_json(evaluation_dir / f"fold_{fold}" / "top_10_predictions.json")
                self.assertEqual([entry["true_label"] for entry in records], labels[indices].tolist())
                for entry in records:
                    self.assertAlmostEqual(sum(item["probability"] for item in entry["top_10"]), 1, places=5)
            self.run_script("summarize_results.py", root, "--evaluation-dir", evaluation_dir)
            report = read_json(evaluation_dir / "rl_result.json")
            self.assertEqual(sum(item["total"] for item in report["accuracy_per_label"].values()), 20)

    def test_video_without_hands_writes_empty_feature_files(self):
        import cv2

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video_dir = root / "videos"
            video_dir.mkdir()
            writer = cv2.VideoWriter(
                str(video_dir / "subject01_01.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                10, (64, 64),
            )
            self.assertTrue(writer.isOpened())
            for _ in range(3):
                writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
            writer.release()
            output = root / "output"
            self.run_script("extract_hand_features.py", root, "--video-dir", video_dir, "--output-dir", output)
            for name in ("subject01_01r.json", "subject01_01l.json"):
                self.assertEqual(read_json(output / "hand_info" / name), [])
            self.assertEqual(read_json(output / "pos" / "subject01_01.json"), [])


if __name__ == "__main__":
    unittest.main()
