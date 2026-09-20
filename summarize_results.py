"""各 fold の Top-1 予測を、左右の手を同一視して集計する。"""

import argparse
from pathlib import Path

from evaluate_models import accuracy_report
from training_utils import DEFAULT_OUTPUT_DIR, read_json, write_json


def remove_hand_suffix(label):
    """01r / 01l を 01 にまとめる。"""
    return label[:-1] if label.endswith(("r", "l")) else label


def summarize(evaluation_dir):
    prediction_files = sorted(evaluation_dir.glob("fold_*/top_10_predictions.json"))
    if not prediction_files:
        raise FileNotFoundError(f"No fold predictions found in: {evaluation_dir}")

    records = []
    for path in prediction_files:
        for entry in read_json(path):
            records.append({
                "true_label": remove_hand_suffix(entry["true_label"]),
                "top_10": [
                    {**prediction, "label": remove_hand_suffix(prediction["label"])}
                    for prediction in entry["top_10"]
                ],
            })
    classes = sorted({entry["true_label"] for entry in records})
    report = accuracy_report(records, classes)
    # 旧 k3.py の出力キー・ファイル名を維持する。
    report["variance"] = report.pop("variance_accuracy")
    output_path = evaluation_dir / "rl_result.json"
    write_json(output_path, report)
    print(f"Top-1 accuracy (ignoring hand side): {report['mean_accuracy']:.4f}")
    print(f"Saved: {output_path}")
    return report


def main():
    parser = argparse.ArgumentParser(description="Summarize Top-1 predictions while ignoring left/right hand suffixes.")
    parser.add_argument("--evaluation-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "evaluation")
    args = parser.parse_args()
    summarize(args.evaluation_dir)


if __name__ == "__main__":
    main()
