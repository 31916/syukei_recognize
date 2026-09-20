"""MP4 動画から手の2次元座標・20関節角度・画像を抽出する。"""

import argparse
from pathlib import Path

import numpy as np

from training_utils import DEFAULT_OUTPUT_DIR, PROJECT_ROOT, write_json

# 各組の中央の点を頂点とする角度。親指から小指まで各4角度。
ANGLE_LANDMARK_TRIPLETS = (
    (5, 0, 1), (0, 1, 2), (1, 2, 3), (2, 3, 4),
    (9, 0, 5), (0, 5, 6), (5, 6, 7), (6, 7, 8),
    (13, 0, 9), (0, 9, 10), (9, 10, 11), (10, 11, 12),
    (17, 0, 13), (0, 13, 14), (13, 14, 15), (14, 15, 16),
    (17, 0, 1), (0, 17, 18), (17, 18, 19), (18, 19, 20),
)


def calculate_joint_angles(positions):
    """余弦定理で20角度（度）を求める。縮退した辺は NaN とする。"""
    angles = []
    for first, center, last in ANGLE_LANDMARK_TRIPLETS:
        point1, point2, point3 = positions[[first, center, last]]
        length1 = np.linalg.norm(point2 - point1)
        length2 = np.linalg.norm(point3 - point2)
        opposite = np.linalg.norm(point1 - point3)
        if not np.isfinite(length1 + length2) or length1 == 0 or length2 == 0:
            angles.append(float("nan"))
            continue
        cosine = (length1**2 + length2**2 - opposite**2) / (2 * length1 * length2)
        angles.append(float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))))
    return angles


def hand_direction(positions):
    """手首→人差し指先端の画像平面上の方向（ラジアン）。"""
    direction = positions[8] - positions[0]
    return float(np.arctan2(direction[1], direction[0]))


def landmark_positions(hand, image, width, height):
    """MediaPipe の手ランドマークを画素座標に変換し、画像にも描画する。"""
    if hand is None:
        return np.full((21, 2), np.nan)
    import mediapipe as mp

    mp.solutions.drawing_utils.draw_landmarks(
        image, hand, mp.solutions.holistic.HAND_CONNECTIONS,
    )
    return np.asarray([[width * point.x, height * point.y] for point in hand.landmark])


def maximum_movement(current, previous):
    valid = np.isfinite(current).all(axis=1) & np.isfinite(previous).all(axis=1)
    if not valid.any():
        return 0.0
    return float(np.max(np.linalg.norm(current[valid] - previous[valid], axis=1)))


def process_video(video_path, output_dir, movement_threshold=15):
    import cv2
    import mediapipe as mp

    video_name = video_path.stem
    frame_dir = output_dir / "frame" / f"{video_name}_frames"
    landmark_dir = output_dir / "landmark" / f"{video_name}_landmarks"
    frame_dir.mkdir(parents=True, exist_ok=True)
    landmark_dir.mkdir(parents=True, exist_ok=True)
    hands = {"right": [], "left": []}
    positions = []
    previous = None

    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        print(f"Processing {video_path.name} (FPS: {capture.get(cv2.CAP_PROP_FPS):g})")
        # 動画ごとにトラッカーを初期化し、別動画の状態を持ち越さない。
        with mp.solutions.holistic.Holistic(
            static_image_mode=False, min_detection_confidence=0.5
        ) as detector:
            frame_number = 0
            while True:
                success, frame = capture.read()
                if not success:
                    break
                height, width = frame.shape[:2]
                results = detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                annotated = frame.copy()
                right = landmark_positions(results.right_hand_landmarks, annotated, width, height)
                left = landmark_positions(results.left_hand_landmarks, annotated, width, height)
                current = (right, left)

                if all(np.isnan(hand).all() for hand in current):
                    frame_number += 1
                    continue

                # 元の研究コードでは角度を先に記録する。
                # 移動量の閾値が適用されるのは画像・座標の保存だけ。
                for name, side, hand in (("right", 1, right), ("left", 0, left)):
                    angles = calculate_joint_angles(hand)
                    if np.isfinite(angles).all():
                        hands[name].append({
                            "frame": frame_number, "angles": angles,
                            "yaw": hand_direction(hand), "rl": side,
                        })

                moving = previous is not None and any(
                    maximum_movement(hand, old) > movement_threshold
                    for hand, old in zip(current, previous)
                )
                previous = current
                if not moving:
                    positions.append({
                        "frame": frame_number,
                        "right_hand_pos": right.tolist() if np.isfinite(right).all() else None,
                        "left_hand_pos": left.tolist() if np.isfinite(left).all() else None,
                    })
                    filename = f"{video_name}_{frame_number:04d}.jpg"
                    for path, image in ((frame_dir / filename, frame), (landmark_dir / filename, annotated)):
                        if not cv2.imwrite(str(path), image):
                            raise OSError(f"Could not save image: {path}")
                frame_number += 1
    finally:
        capture.release()

    # 検出が0件でも空配列を書き、前回の特徴量が残ることを防ぐ。
    write_json(output_dir / "hand_info" / f"{video_name}r.json", hands["right"])
    write_json(output_dir / "hand_info" / f"{video_name}l.json", hands["left"])
    write_json(output_dir / "pos" / f"{video_name}.json", positions)


def main():
    parser = argparse.ArgumentParser(description="Extract hand landmarks and 20 joint angles from MP4 videos.")
    parser.add_argument("--video-dir", type=Path, default=PROJECT_ROOT / "data" / "video")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--movement-threshold", type=float, default=15.0, help="Maximum movement in pixels for saving images and positions")
    args = parser.parse_args()
    if not np.isfinite(args.movement_threshold) or args.movement_threshold < 0:
        parser.error("--movement-threshold must be finite and non-negative")
    videos = sorted(path for path in args.video_dir.glob("*") if path.suffix.lower() == ".mp4")
    if not videos:
        parser.error(f"No MP4 videos found in {args.video_dir}")
    for video_path in videos:
        process_video(video_path, args.output_dir, args.movement_threshold)


if __name__ == "__main__":
    main()
