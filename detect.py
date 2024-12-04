import mediapipe as mp
import cv2
import numpy as np
import os
import json
import glob
import tensorflow as tf

tf.get_logger().setLevel('ERROR')

# 初期設定
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    static_image_mode=False,
    min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# ブレを判定するスレッショルド値
THRESHOLD = 5  # フレーム間での許容移動距離

def process_video(video_path, output_dir_frame, output_dir_hand_info, output_dir_landmark, output_dir_pos):
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # 出力ディレクトリのパスを設定
    save_frame_dir = os.path.join(output_dir_frame, f'{video_name}_frames')
    save_hand_info_path_r = os.path.join(output_dir_hand_info, f'{video_name}r.json')
    save_hand_info_path_l = os.path.join(output_dir_hand_info, f'{video_name}l.json')
    save_landmark_dir = os.path.join(output_dir_landmark, f'{video_name}_landmarks')
    save_pos_path = os.path.join(output_dir_pos, f'{video_name}.json')

    # 既存の出力ファイルがあれば削除
    if os.path.exists(save_hand_info_path_r):
        os.remove(save_hand_info_path_r)
    if os.path.exists(save_hand_info_path_l):
        os.remove(save_hand_info_path_l)
    if os.path.exists(save_pos_path):
        os.remove(save_pos_path)

    os.makedirs(save_frame_dir, exist_ok=True)
    os.makedirs(output_dir_hand_info, exist_ok=True)
    os.makedirs(save_landmark_dir, exist_ok=True)
    os.makedirs(output_dir_pos, exist_ok=True)

    hand_info_r = []
    hand_info_l = []
    pos_data = []

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    prev_pos_r, prev_pos_l = None, None  # 前フレームの座標を保持する変数

    while True:
        ret, frame = cap.read()
        if not ret: break

        height, width, _ = frame.shape
        pos_r, pos_l, landmark_bgr = landmark(frame, height, width)

        # 前フレームと比較してブレをチェック
        if prev_pos_r is not None and prev_pos_l is not None:
            diff_r = np.nanmax(np.linalg.norm(pos_r - prev_pos_r, axis=1))
            diff_l = np.nanmax(np.linalg.norm(pos_l - prev_pos_l, axis=1))
            if diff_r > THRESHOLD or diff_l > THRESHOLD:
                print(f'Skipping frame {frame_count} due to excessive movement.')
                frame_count += 1
                prev_pos_r, prev_pos_l = pos_r, pos_l  # 次のフレームのために現在の座標を保存
                continue  # このフレームの処理をスキップ

        degree_r, degree_l, rad_r, rad_l, hand_orientation_r, hand_orientation_l = angle(pos_r, pos_l)

        if np.any(np.isnan(pos_r)) or np.any(np.isnan(pos_l)) or np.any(np.isnan(degree_r)) or np.any(np.isnan(degree_l)):
            print(f'Skipping frame {frame_count} due to NaN values.')
            frame_count += 1
            continue

        # 右手の情報
        hand_info_r.append({
            "frame": frame_count,
            "angles": degree_r,
            "palm_orientation": hand_orientation_r["palm_orientation"],
            "yaw": hand_orientation_r["yaw"],
            "rl": 1
        })

        # 左手の情報
        hand_info_l.append({
            "frame": frame_count,
            "angles": degree_l,
            "palm_orientation": hand_orientation_l["palm_orientation"],
            "yaw": hand_orientation_l["yaw"],
            "rl" : 0
        })

        pos_info = {
            "frame": frame_count,
            "right_hand_pos": pos_r.tolist(),
            "left_hand_pos": pos_l.tolist()
        }
        pos_data.append(pos_info)

        frame_filename = os.path.join(save_frame_dir, f"{video_name}_{frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)

        landmark_filename = os.path.join(save_landmark_dir, f"{video_name}_{frame_count:04d}.jpg")
        cv2.imwrite(landmark_filename, landmark_bgr.astype(np.uint8))

        frame_count += 1
        prev_pos_r, prev_pos_l = pos_r, pos_l  # 現在の座標を保存して次のフレームへ

    with open(save_hand_info_path_r, 'w') as json_file:
        json.dump(hand_info_r, json_file, indent=4)

    with open(save_hand_info_path_l, 'w') as json_file:
        json.dump(hand_info_l, json_file, indent=4)

    with open(save_pos_path, 'w') as json_file:
        json.dump(pos_data, json_file, indent=4)

    cap.release()


def main():
    # 動画が入っているディレクトリのパス
    video_dir = r'./do/data/video/'

    # 出力ディレクトリ
    output_dir_frame = r'./do/data/output/frame/'
    output_dir_hand_info = r'./do/data/output/hand_info/'
    output_dir_landmark = r'./do/data/output/landmark/'
    output_dir_pos = r'./do/data/output/pos/'

    # ディレクトリ内のすべての動画ファイルを取得
    video_paths = glob.glob(os.path.join(video_dir, '*.mp4'))

    # 各動画に対して処理を実行
    for video_path in video_paths:
        print(f'Processing video: {video_path}')
        process_video(video_path, output_dir_frame, output_dir_hand_info, output_dir_landmark, output_dir_pos)


# 右手のランドマーク取得
def r_hand(results, annotated_image, height, width, pos):
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=results.right_hand_landmarks,
            connections=mp_holistic.HAND_CONNECTIONS)
                
        for index, landmark in enumerate(results.right_hand_landmarks.landmark):
            x = width * landmark.x
            y = height * landmark.y
            pos = np.vstack((pos, [x, y]))  # 新しい座標を追加

    else:
        for index in range(21):
            pos = np.vstack((pos, [np.nan, np.nan]))  # NaNを追加
    return pos

# 左手のランドマーク取得
def l_hand(results, annotated_image, height, width, pos):
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=results.left_hand_landmarks,
            connections=mp_holistic.HAND_CONNECTIONS)
        
        for index, landmark in enumerate(results.left_hand_landmarks.landmark):
            x = width * landmark.x
            y = height * landmark.y
            pos = np.vstack((pos, [x, y]))  # 新しい座標を追加
    else:
        for index in range(21):
            pos = np.vstack((pos, [np.nan, np.nan]))  # NaNを追加
    return pos


# ランドマークの取得
def landmark(image, height, width):
    results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    annotated_image = image.copy()
    pos_r = np.empty((0,2), dtype=float)
    pos_l = np.empty((0,2), dtype=float)
    
    # 右手、左手の順にランドマーク取得
    pos_r = r_hand(results, annotated_image, height, width, pos_r)
    pos_l = l_hand(results, annotated_image, height, width, pos_l)

    return pos_r, pos_l, annotated_image


def angle(pos_r, pos_l):
    # 角度の配列
    degree_r = []
    degree_l = []
    rad_r = []
    rad_l = []

    # 角度計算用
    angle_landmark_set = [
    (5, 0, 1), (0, 1, 2), (1, 2, 3), (2, 3, 4),  # 親指
    (9, 0, 5), (0, 5, 6), (5, 6, 7), (6, 7, 8),  # 人差し指
    (13, 0, 9), (0, 9, 10), (9, 10, 11), (10, 11, 12),  # 中指
    (17, 0, 13), (0, 13, 14), (13, 14, 15), (14, 15, 16),  # 薬指
    (17, 0, 1), (0, 17, 18), (17, 18, 19), (18, 19, 20)  # 小指
    ]

    def calculate_angle(p1, p2, p3):
        a = np.linalg.norm(p2 - p1)
        b = np.linalg.norm(p3 - p2)
        c = np.linalg.norm(p1 - p3)
        angle = np.arccos((a**2 + b**2 - c**2) / (2 * a * b))
        return np.degrees(angle)

    def get_angles(landmarks):
        angles = []
        for i1, i2, i3 in angle_landmark_set:
            angle = calculate_angle(landmarks[i1], landmarks[i2], landmarks[i3])
            angles.append(angle)
        return angles

    degree_r = get_angles(pos_r)
    degree_l = get_angles(pos_l)

    # 右手の向き
    hand_orientation_r = get_hand_orientation(pos_r)
    # 左手の向き
    hand_orientation_l = get_hand_orientation(pos_l)

    return degree_r, degree_l, rad_r, rad_l, hand_orientation_r, hand_orientation_l


def get_hand_orientation(pos):
    # 手首と親指の先端（もしくは中指の先端）の位置から手の向きを判定
    wrist = pos[0]  # 手首のランドマーク
    index_finger_tip = pos[8]  # 人差し指の先端（例）
    
    if np.isnan(wrist[0]) or np.isnan(index_finger_tip[0]):
        return {"palm_orientation": None, "yaw": None}
    
    # 手のひらがカメラ向きか手の甲がカメラ向きかを判断
    palm_orientation = np.sign(index_finger_tip[1] - wrist[1])  # 上向きか下向きか
    yaw = np.arctan2(index_finger_tip[1] - wrist[1], index_finger_tip[0] - wrist[0])

    return {"palm_orientation": palm_orientation, "yaw": yaw}

if __name__ == '__main__':
    main()
