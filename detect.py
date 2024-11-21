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
THRESHOLD = 8  # フレーム間での許容移動距離

def process_video(video_path, output_dir_frmae, output_dir_angle, output_dir_landmark, output_dir_pos):
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    save_frame_dir = os.path.join(output_dir_frmae, f'{video_name}_frames')
    save_angle_path = os.path.join(output_dir_angle, f'{video_name}.json')
    save_landmark_dir = os.path.join(output_dir_landmark, f'{video_name}_landmarks')
    save_pos_path = os.path.join(output_dir_pos, f'{video_name}.json')

    if os.path.exists(save_angle_path):
        os.remove(save_angle_path)
    if os.path.exists(save_pos_path):
        os.remove(save_pos_path)
    
    os.makedirs(save_frame_dir, exist_ok=True)
    os.makedirs(output_dir_angle, exist_ok=True)
    os.makedirs(save_landmark_dir, exist_ok=True)
    os.makedirs(output_dir_pos, exist_ok=True)

    angles_data = []
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

        degree_r, degree_l, rad_r, rad_l, orientation_r, orientation_l = angle(pos_r, pos_l)

        if np.any(np.isnan(pos_r)) or np.any(np.isnan(pos_l)) or np.any(np.isnan(degree_r)) or np.any(np.isnan(degree_l)):
            print(f'Skipping frame {frame_count} due to NaN values.')
            frame_count += 1
            continue

        angle_info = {
            "frame": frame_count,
            "right_hand_info": degree_r + [orientation_r],
            "left_hand_info": degree_l + [orientation_l]
        }
        angles_data.append(angle_info)

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

    with open(save_angle_path, 'w') as json_file:
        json.dump(angles_data, json_file, indent=4)

    with open(save_pos_path, 'w') as json_file:
        json.dump(pos_data, json_file, indent=4)

    cap.release()


def main():
    # 動画が入っているディレクトリのパス
    video_dir = r'./do/data/video/'
    
    # 出力ディレクトリ
    output_dir_frame = r'./do/data/output/frame/'
    output_dir_angle = r'./do/data/output/angle/'
    output_dir_landmark = r'./do/data/output/landmark/'
    output_dir_pos = r'./do/data/output/pos/'
    
    # ディレクトリ内のすべての動画ファイルを取得
    video_paths = glob.glob(os.path.join(video_dir, '*.mp4'))

    # 各動画に対して処理を実行
    for video_path in video_paths:
        print(f'Processing video: {video_path}')
        process_video(video_path, output_dir_frame, output_dir_angle, output_dir_landmark, output_dir_pos)


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
    #角度の配列
    degree_r = []
    degree_l = []
    rad_r = []
    rad_l = []
   
    #角度計算用
    angle_landmark_set = [
        (5, 0, 1), (0, 1, 2), (1, 2, 3), (2, 3, 4), # 親指
        (9, 0, 5), (0, 5, 6), (5, 6, 7), (6, 7, 8), # 人差し指
        (13, 0, 9), (0, 9, 10), (9, 10, 11), (10, 11, 12), # 中指
        (17, 0, 13), (0, 13, 14), (13, 14, 15), (14, 15, 16), # 薬指
        (17, 0, 1), (0, 17, 18), (17, 18, 19), (18, 19, 20) # 小指
    ]

    #角度の計算
    #右手
    for i in range(len(angle_landmark_set)):
        vec_a = pos_r[angle_landmark_set[i][0]] - pos_r[angle_landmark_set[i][1]]
        vec_b = pos_r[angle_landmark_set[i][2]] - pos_r[angle_landmark_set[i][1]]

        # コサインの計算
        length_vec_a = np.linalg.norm(vec_a)
        length_vec_c = np.linalg.norm(vec_b)
        inner_product = np.inner(vec_a, vec_b)
        cos = inner_product / (length_vec_a * length_vec_c)

        # 角度（ラジアン）の計算
        rad = np.arccos(cos)

        # 弧度法から度数法（rad ➔ 度）への変換
        degree_r.append(np.rad2deg(rad))

        #弧度法
        rad_r.append(np.arccos(cos))

    #左手
    for j in range(len(angle_landmark_set)):
        vec_a = pos_l[angle_landmark_set[j][0]] - pos_l[angle_landmark_set[j][1]]
        vec_b = pos_l[angle_landmark_set[j][2]] - pos_l[angle_landmark_set[j][1]]

        # コサインの計算
        length_vec_a = np.linalg.norm(vec_a)
        length_vec_c = np.linalg.norm(vec_b)
        inner_product = np.inner(vec_a, vec_b)
        cos = inner_product / (length_vec_a * length_vec_c)

        # 角度（ラジアン）の計算
        rad = np.arccos(cos)

        # 弧度法から度数法（rad ➔ 度）への変換
        degree_l.append(np.rad2deg(rad))

        #弧度法
        rad_l.append(np.arccos(cos))

    # 手のひらの向きの判定
    orientation_r = 0 if pos_r[0][0] > pos_r[4][0] else 1  # 手のひらが上 (0) or 下 (1)
    orientation_l = 0 if pos_l[0][0] > pos_l[4][0] else 1  # 手のひらが上 (0) or 下 (1)

    return degree_r, degree_l, rad_r, rad_l, orientation_r, orientation_l


if __name__ == '__main__':
    main()
