import cv2
import mediapipe as mp

#描画ツール設定
mp_drawing = mp.solutions.drawing_utils     
mp_drawing_styles = mp.solutions.drawing_styles
#姿勢推定ツール設定
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence = 0.5,                                                                                     #検出信頼度
    min_tracking_confidence = 0.7                                                                                       #追跡信頼度
)
#手形推定ツール設定
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands = 2,                                                                                                  #最大検出数
    min_detection_confidence = 0.7,                                                                                     #検出信頼度
    min_tracking_confidence = 0.7                                                                                       #追跡信頼度
)
# landmarkの繋がり表示用
landmark_line_ids = [ 
    (0, 1), (1, 5), (5, 9), (9, 13), (13, 17), (17, 0),  # 掌
    (1, 2), (2, 3), (3, 4),         # 親指
    (5, 6), (6, 7), (7, 8),         # 人差し指
    (9, 10), (10, 11), (11, 12),    # 中指
    (13, 14), (14, 15), (15, 16),   # 薬指
    (17, 18), (18, 19), (19, 20),   # 小指
]

#camera
def camera():
    cap = cv2.VideoCapture(0)
    try:    #エラーの起こる可能性がある処理（カメラ）
        while cap.isOpened():
            ret,img = cap.read()
            if not ret:
                print("カメラ映像を取得できませんでした\n")
                continue
            img = cv2.flip(img,1)                                                                                       #画像を左右反転
            img_h, img_w, _ = img.shape                                                                                 #サイズ取得

            # OpenCVとMediaPipeでRGBの並びが違うため、
            # 処理前に変換しておく。
            # CV2:BGR → MediaPipe:RGB
            image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            pose_results = pose.process(image)                                                                          #姿勢推定
            hands_results = hands.process(image)                                                                        #手形推定

            #姿勢描画
            mp_drawing.draw_landmarks(
                img,
                pose_results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing_styles.get_default_pose_landmarks_style()
            )

            #手形描画
            if hands_results.multi_hand_landmarks:                
                for h_id, hand_landmarks in enumerate(hands_results.multi_hand_landmarks):             
                    #通常の描画
                    # mp_drawing.draw_landmarks(
                    #     image,
                    #     landmarks,
                    #     mp_hands.HAND_CONNECTIONS,
                    #     mp_drawing_styles.get_default_hand_landmarks_style(),
                    #     mp_drawing_styles.get_default_hand_connections_style()
                    # )

                    #改良版の描画
                    for line_id in landmark_line_ids:                                                                   #線の描画
                        lm = hand_landmarks.landmark[line_id[0]]                                                        #1点目の座標取得
                        lm_pos1 = (int(lm.x * img_w), int(lm.y * img_h))
                        lm = hand_landmarks.landmark[line_id[1]]                                                        #2点目の座標取得
                        lm_pos2 = (int(lm.x * img_w), int(lm.y * img_h))
                        cv2.line(img, lm_pos1, lm_pos2, (128, 0, 0), 1)                                                 #line描画
                    z_list = [lm.z for lm in hand_landmarks.landmark]                   
                    z_min = min(z_list)
                    z_max = max(z_list)
                    for lm in hand_landmarks.landmark:                                                                  #点の描画
                        lm_pos = (int(lm.x * img_w), int(lm.y * img_h))                                                 #ランドマークの座標取得
                        lm_z = int((lm.z - z_min) / (z_max - z_min) * 255)                                              #点の色のグラデーション
                        cv2.circle(img, lm_pos, 3, (255, lm_z, lm_z), -1)                                               #circle描画


                    #テキスト情報出力
                    hand_texts = []
                    for c_id, hand_class in enumerate(hands_results.multi_handedness[h_id].classification):
                        hand_texts.append("- Label:%s" % (hand_class.label))
                    lm = hand_landmarks.landmark[0]
                    lm_x = int(lm.x * img_w) - 50
                    lm_y = int(lm.y * img_h) - 10
                    lm_c = (64, 0, 0)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    for cnt, text in enumerate(hand_texts):
                        cv2.putText(img, text, (lm_x, lm_y + 10 * cnt), font, 0.3, lm_c, 1)

            
            #ディスプレイ表示
            cv2.imshow('test',img)

            #終了キー
            key = cv2.waitKey(1)
            if key == 27:                                       #ECSキーで終了
                print("終了")
                break            
    finally:
        cap.release()
        cv2.destroyAllWindows()
            
#main        
if __name__ == '__main__':  #呼び出し先の命令でなければ（mainであれば）実行
    camera()
