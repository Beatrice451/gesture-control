import math
import sys
import time

import cv2
import mediapipe as mp
import numpy as np

from camera_stream import CameraStream
from config import MP_WIDTH, MP_HEIGHT, HAND_CONNECTIONS, PINCH_CLOSE, PINCH_OPEN, PINCH_RIGHT_CLOSE, DRAG_HOLD_TIME, \
    REFERENCE_HAND_SCALE

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

size_k = 0.0
scroll_active = False
right_pinch_active = False
dragging = False

pinch_start_time = 0.0

last_len = 0

def is_scroll_gesture(landmarks):
    ext = []
    for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        tip_d = np.hypot(landmarks[tip].x - landmarks[0].x,
                         landmarks[tip].y - landmarks[0].y)
        pip_d = np.hypot(landmarks[pip].x - landmarks[0].x,
                         landmarks[pip].y - landmarks[0].y)
        ext.append(tip_d > pip_d * 1.15)
    return ext[0] and ext[1] and not ext[2] and not ext[3]

# функция, которая проверяет раскрыта ли ладонь
# спросите меня что тут происходит и я вам отвечу - без понятия.
def is_open_palm(landmarks):
    fingers = [
        (8, 6),  # указательный
        (12, 10),  # средний
        (16, 14),  # я забыл как он называется блять
        (20, 18)  # мизинец
    ]

    extended = 0

    for tip, pip in fingers:
        tip_distance = np.hypot(
            landmarks[tip].x - landmarks[0].x,
            landmarks[tip].y - landmarks[0].y
        )

        pip_distance = np.hypot(
            landmarks[pip].x - landmarks[0].x,
            landmarks[pip].y - landmarks[0].y
        )

        if tip_distance > pip_distance * 1.15:  # а шо тут значит 1.15? Хз
            extended += 1

    return extended >= 4

def is_fist(landmarks):
    # все 4 пальца согнуты: кончик ближе к запястью, чем PIP
    folded = 0
    for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        tip_d = np.hypot(landmarks[tip].x - landmarks[0].x,
                         landmarks[tip].y - landmarks[0].y)
        pip_d = np.hypot(landmarks[pip].x - landmarks[0].x,
                         landmarks[pip].y - landmarks[0].y)
        if tip_d < pip_d:
            folded += 1
    return folded >= 4

def status_line(text):
    global last_len
    pad = max(0, last_len - len(text))
    sys.stdout.write("\r" + text + " " * pad)
    sys.stdout.flush()
    last_len = len(text)

def hand_scale(landmarks):
    """Устойчивый эталон размера кисти.
    Усредняем расстояния от запястья (0) до оснований пальцев (5, 9, 13, 17).
    Берём в нормализованных координатах — этого достаточно, поскольку
    коэффициент в дальнейшем используется как отношение."""
    refs = (5, 9, 13, 17)
    d = 0.0
    for i in refs:
        d += math.hypot(
            landmarks[i].x - landmarks[0].x,
            landmarks[i].y - landmarks[0].y,
        )
    return (d / len(refs)) or 1e-6


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

            # толщину линии и радиус точек масштабируем тем же size_k
            line_th = max(1, int(round(2 * size_k)))
            dot_r = int(np.clip(round(5 * size_k), 2, 10))

            for start_idx, end_idx in HAND_CONNECTIONS:
                start = hand_landmarks[start_idx]
                end = hand_landmarks[end_idx]
                start_px = (int(start.x * w), int(start.y * h))
                end_px = (int(end.x * w), int(end.y * h))
                cv2.line(frame, start_px, end_px, (0, 255, 0), line_th)

            for lm in hand_landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), dot_r, (255, 0, 0), -1)

            index = hand_landmarks[8]
            thumb = hand_landmarks[4]
            middle = hand_landmarks[12]

            palm_x = (
                             hand_landmarks[0].x + hand_landmarks[5].x + hand_landmarks[9].x
                             + hand_landmarks[13].x + hand_landmarks[17].x
                     ) / 5.0
            palm_y = (
                             hand_landmarks[0].y + hand_landmarks[5].y + hand_landmarks[9].y
                             + hand_landmarks[13].y + hand_landmarks[17].y
                     ) / 5.0

            scale = hand_scale(hand_landmarks)
            # >1 — рука ближе эталона, <1 — дальше. Клипуем, чтобы шум позы
            # (например, при сильном наклоне кисти) не ломал пороги.
            size_k = float(np.clip(scale / REFERENCE_HAND_SCALE, 0.4, 2.5))

            dist = float(np.hypot(index.x - thumb.x, index.y - thumb.y)) / size_k
            dist_right = float(np.hypot(thumb.x - middle.x, thumb.y - middle.y)) / size_k
            now_t = time.time()

            fist = is_fist(hand_landmarks)
            open_palm = is_open_palm(hand_landmarks)

            input_locked = fist

            # если пользователь сжал кулак посреди drag/pinch/скролла — аккуратно всё отпускаем,
            # чтобы ничего не осталось зажатым на время записи
            if input_locked:
                if dragging:
                    dragging = False
                pinch_active = False
                right_pinch_active = False
                scroll_active = False


            scroll_gesture = (
                    is_scroll_gesture(hand_landmarks)
                    and not pinch_active
                    and not right_pinch_active
                    and not input_locked
            )


            open_palm = is_open_palm(hand_landmarks)

            if open_palm and not pinch_active and not scroll_gesture:
                swipe_mode = True

            else:
                swipe_mode = False



            # --------------------------------------------------
            # PINCH / КЛИК / СКРИНШОТ
            # --------------------------------------------------

            # ---------- ЛЕВЫЙ PINCH: КЛИК / DRAG ----------
            if not input_locked:
                if not pinch_active and not right_pinch_active:
                    if dist < PINCH_CLOSE:
                        pinch_active = True
                        pinch_start_time = now_t
                        dragging = False

                elif pinch_active:
                    if dist < PINCH_OPEN:
                        hold_time = now_t - pinch_start_time
                        if hold_time >= DRAG_HOLD_TIME and not dragging:
                            dragging = True
                    else:
                        if dragging:
                            dragging = False

                        pinch_active = False

            # ---------- ПРАВЫЙ PINCH: ПКМ ----------
            if not input_locked:
                if not pinch_active and not right_pinch_active:
                    if dist_right < PINCH_RIGHT_CLOSE:
                        right_pinch_active = True

                elif right_pinch_active:
                    right_pinch_active = False

            # ---------- СКРОЛЛ ----------
            if scroll_gesture:

                if not scroll_active:
                    scroll_active = True

            else:
                if scroll_active:
                    scroll_active = False


        fps_count += 1
        if time.time() - fps_timer >= 1.0:
            # print(f"FPS: {fps_count}")
            fps_count = 0
            fps_timer = time.time()
            status_line(f"left pinch: {pinch_active} | right pinch: {right_pinch_active} | scroll gesture: {scroll_active} | dragging: {dragging}")

        cv2.imshow("Gesture Control", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    stream.release()
    cv2.destroyAllWindows()
    detector.close()
