import sys
import time

import cv2
import mediapipe as mp
import numpy as np

from camera_stream import CameraStream
from config import MP_WIDTH, MP_HEIGHT, HAND_CONNECTIONS, PINCH_CLOSE, PINCH_OPEN

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    min_hand_presence_confidence=0.5,
)
detector = HandLandmarker.create_from_options(options)

stream = CameraStream(src=0, w=640, h=480, fps=30)

fps_timer = time.time()
fps_count = 0
frame_idx = 0
last_processed_id = -1
pinch_active = False

last_len = 0

def status_line(text):
    global last_len
    pad = max(0, last_len - len(text))
    sys.stdout.write("\r" + text + " " * pad)
    sys.stdout.flush()
    last_len = len(text)

try:
    while True:
        frame, frame_id = stream.read_latest(last_processed_id)

        if frame is None:
            time.sleep(0.001)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        last_processed_id = frame_id
        frame_idx += 1

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        small = cv2.resize(frame, (MP_WIDTH, MP_HEIGHT), interpolation=cv2.INTER_AREA)
        rgb_small = cv2.cvtColor(
            small,
            cv2.COLOR_BGR2RGB
        )
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_small)

        timestamp_ms = int(frame_idx * (1000.0 / 30.0))
        result = detector.detect_for_video(mp_image, timestamp_ms)

        if result.hand_landmarks:
            hand_landmarks = result.hand_landmarks[0]

            for start_idx, end_idx in HAND_CONNECTIONS:
                start = hand_landmarks[start_idx]
                end = hand_landmarks[end_idx]
                start_px = (int(start.x * w), int(start.y * h))
                end_px = (int(end.x * w), int(end.y * h))
                cv2.line(frame, start_px, end_px, (0, 255, 0), 2)

            for lm in hand_landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

            index = hand_landmarks[8]
            thumb = hand_landmarks[4]

            dist = float(np.hypot(index.x - thumb.x, index.y - thumb.y))

            if dist < PINCH_CLOSE and not pinch_active:
                pinch_active = True
                # print("пальцы сомкнулись")

            elif dist > PINCH_OPEN and pinch_active:
                pinch_active = False
                # print("пальцы разомкнулись")


        fps_count += 1
        if time.time() - fps_timer >= 1.0:
            # print(f"FPS: {fps_count}")
            fps_count = 0
            fps_timer = time.time()

        cv2.imshow("Gesture Control", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        status_line(
            f"FPS: {fps_count:3d} | pinch: {pinch_active}"
        )

finally:
    stream.release()
    cv2.destroyAllWindows()
    detector.close()
