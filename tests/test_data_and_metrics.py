"""データの順序・数値処理・評価指標の回帰テスト。"""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluate_models import accuracy_report, build_prediction_records
from extract_hand_features import calculate_joint_angles, maximum_movement
from summarize_results import summarize
from training_utils import parse_filename, prepare_data, write_json


def sample(value, side=1):
    return {"frame": value, "angles": [float(value)] * 20, "yaw": 0, "rl": side}


class DataAndMetricsTests(unittest.TestCase):
    def test_saved_samples_survive_source_changes_and_seed_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "input"
            source.mkdir()
            write_json(source / "b_01r.json", [sample(index) for index in range(8)])
            write_json(source / "a_02l.json", [sample(index, 0) for index in range(8)])
            write_json(source / "a_09r.json", [sample(1)])  # 除外番号
            snapshot = root / "model" / "selected_data.json"
            original = prepare_data(source, save_selected=True, selected_data_path=snapshot)
            self.assertEqual(original[0].shape, (10, 20))
            self.assertEqual(original[1].tolist(), ["02l"] * 5 + ["01r"] * 5)
            write_json(source / "a_02l.json", [sample(999, 0)])
            restored = prepare_data(None, random_seed=999, selected_data_path=snapshot)
            for before, after in zip(original, restored):
                np.testing.assert_array_equal(before, after)

    def test_invalid_angles_do_not_shift_labels_or_subjects(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            write_json(source / "a_01r.json", [sample(1), {"angles": None, "rl": 1}])
            write_json(source / "b_02l.json", [sample(2, 0), {"angles": [1, 2], "rl": 0}])
            features, labels, subjects = prepare_data(source)
            np.testing.assert_array_equal(features[:, 0], [1, 2])
            self.assertEqual(labels.tolist(), ["01r", "02l"])
            self.assertEqual(subjects.tolist(), ["a", "b"])

    def test_empty_input_and_ambiguous_filename_fail_clearly(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "No valid"):
                prepare_data(Path(temporary))
        with self.assertRaisesRegex(ValueError, "Expected"):
            parse_filename("subject_with_underscore_01r.json")

    def test_angles_are_invariant_to_translation_and_uniform_scale(self):
        positions = np.random.RandomState(7).uniform(0, 100, size=(21, 2))
        angles = calculate_joint_angles(positions)
        self.assertEqual(len(angles), 20)
        np.testing.assert_allclose(angles, calculate_joint_angles(positions * 3 + 50))
        self.assertTrue(np.isnan(calculate_joint_angles(np.zeros((21, 2)))).all())
        self.assertEqual(maximum_movement(positions, np.full((21, 2), np.nan)), 0)

    def test_macro_accuracy_and_left_right_aggregation(self):
        classes = ["01r", "01l", "02r", "03r"]
        probabilities = [[0.1, 0.8, 0.1, 0], [0.9, 0, 0.1, 0], [0.8, 0, 0.2, 0]]
        records = build_prediction_records(probabilities, ["01r", "01r", "02r"], classes)
        self.assertEqual(len(records[0]["top_10"]), 4)
        report = accuracy_report(records, classes)
        self.assertEqual(report["mean_accuracy"], 0.25)  # (1/2 + 0/1) / 2
        self.assertEqual(report["variance_accuracy"], 0.0625)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_json(root / "fold_1" / "top_10_predictions.json", records)
            merged = summarize(root)
            self.assertEqual(merged["accuracy_per_label"]["01"]["correct"], 2)
            self.assertEqual(merged["mean_accuracy"], 0.5)


if __name__ == "__main__":
    unittest.main()
