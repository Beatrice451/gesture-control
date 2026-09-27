import math
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import pyperclip
from pynput.mouse import Controller, Button

from camera_stream import CameraStream
from config import *
from filters import OneEuroFilter
from screenshot_selector import ScreenshotSelector
from smooth import SmoothMouse
from voice_input import VoiceInput



pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0

swipe_history = deque(maxlen=20)
last_swipe = 0

pinch_mouse_x = 0
pinch_mouse_y = 0

# Временно убрал скрины
# screenshot_selector = ScreenshotSelector()

pinch_active = False
pinch_start_time = 0.0
# screenshot_mode = False

voice = VoiceInput(model_size="small", language=None)
fist_since = None
palm_since = None

dragging = False
last_click = 0.0

right_pinch_active = False
right_pinch_start_time = 0.0
last_right_click = 0.0

last_click_time = 0.0

scroll_active = False
scroll_prev_y = None
scroll_prev_t = 0.0
scroll_velocity = 0.0  # кликов/сек, сглаженная
scroll_inertia_velocity = 0.0
scroll_inertia_last_t = 0.0
scroll_frac = 0.0  # накопитель дробной части клика

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

screen_w, screen_h = pyautogui.size()
mouse = Controller()

smooth_mouse = SmoothMouse(
    mouse,
    smoothing=0.2,
    fps=60
)

stream = CameraStream(src=0, w=640, h=480, fps=30)

filter_x = OneEuroFilter(freq=30.0, mincutoff=1.2, beta=0.02, dcutoff=1.0)
filter_y = OneEuroFilter(freq=30.0, mincutoff=1.2, beta=0.02, dcutoff=1.0)

raw_positions = deque(maxlen=3)

prev_mouse_x, prev_mouse_y = mouse.position
last_click = 0.0

fps_timer = time.time()
fps_count = 0
frame_idx = 0
last_processed_id = -1


# Для получения "эталона" масштабов
def hand_scale(landmarks):
    # wrist(0) -> middle_mcp(9)
    return float(np.hypot(
        landmarks[9].x - landmarks[0].x,
        landmarks[9].y - landmarks[0].y,
    )) or 1e-6


def is_scroll_gesture(landmarks):
    ext = []
    for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        tip_d = np.hypot(landmarks[tip].x - landmarks[0].x,
                         landmarks[tip].y - landmarks[0].y)
        pip_d = np.hypot(landmarks[pip].x - landmarks[0].x,
                         landmarks[pip].y - landmarks[0].y)
        ext.append(tip_d > pip_d * 1.15)
    return ext[0] and ext[1] and not ext[2] and not ext[3]


def apply_scroll_clicks(clicks_float):
    global scroll_frac
    scroll_frac += clicks_float
    whole = int(scroll_frac)
    if whole != 0:
        mouse.scroll(0, whole)
        scroll_frac -= whole


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


def process_swipe(landmarks, now_t):
    global last_swipe

    palm_x = landmarks[9].x

    swipe_history.append((now_t, palm_x))

    # недостаточно истории
    if len(swipe_history) < 2:
        return False

    if now_t - last_swipe < SWIPE_COOLDOWN:
        return False

    old_time, old_x = swipe_history[0]

    for timestamp, x in swipe_history:
        if now_t - timestamp <= SWIPE_MAX_TIME:
            old_time = timestamp
            old_x = x
            break

    dx = palm_x - old_x
    elapsed = now_t - old_time

    # если прошло отрицательное количество времени?..
    if elapsed <= 0:
        return False

    # слишком медленное движение не считается свайпом
    if abs(dx) < SWIPE_THRESHOLD:
        return False

    if dx > 0:
        pyautogui.hotkey('alt', 'tab')
        print("делаем альт-таб")

    else:
        pyautogui.hotkey('alt', 'shift', 'tab')
        print("делаем альт шифт таб")

    last_swipe = now_t
    swipe_history.clear()
    return True


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
            middle = hand_landmarks[12]

            palm_x = (
                             hand_landmarks[0].x + hand_landmarks[5].x + hand_landmarks[9].x
                             + hand_landmarks[13].x + hand_landmarks[17].x
                     ) / 5.0
            palm_y = (
                             hand_landmarks[0].y + hand_landmarks[5].y + hand_landmarks[9].y
                             + hand_landmarks[13].y + hand_landmarks[17].y
                     ) / 5.0

            # scale = hand_scale(hand_landmarks) # TODO починить масштаб
            scale = 1  # пока что затычка
            dist = float(np.hypot(index.x - thumb.x, index.y - thumb.y)) / scale
            dist_right = float(np.hypot(thumb.x - middle.x, thumb.y - middle.y)) / scale
            now_t = time.time()

            fist = is_fist(hand_landmarks)
            open_palm = is_open_palm(hand_landmarks)

            input_locked = fist

            # если пользователь сжал кулак посреди drag/pinch/скролла — аккуратно всё отпускаем,
            # чтобы ничего не осталось зажатым на время записи
            if input_locked:
                if dragging:
                    mouse.release(Button.left)
                    dragging = False
                pinch_active = False
                right_pinch_active = False
                scroll_active = False
                scroll_inertia_velocity = 0.0
                swipe_history.clear()

            # --- старт записи ---
            if fist:
                if fist_since is None:
                    fist_since = now_t
                palm_since = None
                if not voice.is_recording and (now_t - fist_since) >= FIST_HOLD_TO_START:
                    voice.start()
                    print("[voice] ● запись")
            else:
                fist_since = None

            # --- стоп записи ---
            if open_palm and voice.is_recording:
                if palm_since is None:
                    palm_since = now_t
                if (now_t - palm_since) >= PALM_HOLD_TO_STOP:
                    voice.stop()
                    print("[voice] ■ стоп, отправлено в whisper")
                    palm_since = None
            else:
                if not open_palm:
                    palm_since = None

            scroll_gesture = (
                    is_scroll_gesture(hand_landmarks)
                    and not pinch_active
                    and not right_pinch_active
                    and not input_locked
            )

            raw_positions.append((palm_x, palm_y))
            avg_x_norm = sum(p[0] for p in raw_positions) / len(raw_positions)
            avg_y_norm = sum(p[1] for p in raw_positions) / len(raw_positions)

            target_x = np.interp(avg_x_norm, [X_MIN, X_MAX], [0, screen_w])
            target_y = np.interp(avg_y_norm, [Y_MIN, Y_MAX], [0, screen_h])
            target_x = float(np.clip(target_x, 0, screen_w - 1))
            target_y = float(np.clip(target_y, 0, screen_h - 1))

            now = time.time()
            smooth_x = filter_x(target_x, now)
            smooth_y = filter_y(target_y, now)

            if abs(smooth_x - prev_mouse_x) < DEADZONE_PX:
                smooth_x = prev_mouse_x
            if abs(smooth_y - prev_mouse_y) < DEADZONE_PX:
                smooth_y = prev_mouse_y

            if (not pinch_active or dragging) and not scroll_gesture and not input_locked:
                smooth_mouse.set_target(int(smooth_x), int(smooth_y))
                prev_mouse_x, prev_mouse_y = smooth_x, smooth_y

            open_palm = is_open_palm(hand_landmarks)

            if open_palm and not pinch_active and not scroll_gesture:
                process_swipe(hand_landmarks, now_t)

                swipe_mode = True

            else:
                swipe_mode = False

            if not open_palm:
                swipe_history.clear()
            
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
                            mouse.press(Button.left)
                            dragging = True
                    else:
                        if dragging:
                            mouse.release(Button.left)
                            dragging = False
                        else:
                            if (now_t - last_click) > CLICK_COOLDOWN:
                                mouse.click(Button.left, 1)
                                last_click = now_t

                        pinch_active = False

            # ---------- ПРАВЫЙ PINCH: ПКМ ----------
            if not input_locked:
                if not pinch_active and not right_pinch_active:
                    if dist_right < PINCH_RIGHT_CLOSE:
                        right_pinch_active = True
                        right_pinch_start_time = now_t

                elif right_pinch_active:
                    if dist_right > PINCH_RIGHT_OPEN:
                        if (now_t - right_pinch_start_time) <= RIGHT_CLICK_MAX_HOLD:
                            if (now_t - last_right_click) > RIGHT_CLICK_COOLDOWN:
                                mouse.click(Button.right, 1)
                                last_right_click = now_t

                        right_pinch_active = False

            # ---------- СКРОЛЛ ----------
            if scroll_gesture:
                swipe_history.clear()
                y_norm = (index.y + middle.y) / 2.0

                if not scroll_active:
                    scroll_active = True
                    scroll_prev_y = y_norm
                    scroll_prev_t = now_t
                    scroll_velocity = 0.0
                    scroll_inertia_velocity = 0.0  # новый жест гасит прошлую инерцию
                else:
                    dt = now_t - scroll_prev_t
                    if dt > 0:
                        dy = y_norm - scroll_prev_y
                        if abs(dy) > SCROLL_DEADZONE:
                            apply_scroll_clicks(dy * SCROLL_GAIN)

                            inst_vel = (dy / dt) * SCROLL_GAIN
                            scroll_velocity = (
                                    (1 - SCROLL_VEL_SMOOTH) * scroll_velocity
                                    + SCROLL_VEL_SMOOTH * inst_vel
                            )
                        scroll_prev_y = y_norm
                        scroll_prev_t = now_t
            else:
                if scroll_active:
                    scroll_active = False
                    scroll_inertia_velocity = scroll_velocity
                    scroll_inertia_last_t = now_t
                    scroll_prev_y = None

        # если рука ушла из кадра во время записи — не оставляем её висеть
        if not result.hand_landmarks and voice.is_recording:
            voice.stop()
            fist_since = None
            palm_since = None

        # забрать всё, что успел расшифровать Whisper
        for text in voice.poll():
            print("🎤", text)
            pyperclip.copy(text)
            if AUTO_PASTE:
                pyautogui.hotkey("ctrl", "v")

        # ---------- ИНЕРЦИЯ СКРОЛЛА ----------
        if scroll_inertia_velocity != 0.0 and not scroll_active:
            now_i = time.time()
            dt = now_i - scroll_inertia_last_t
            scroll_inertia_last_t = now_i
            if dt > 0:
                apply_scroll_clicks(scroll_inertia_velocity * dt)
                scroll_inertia_velocity *= math.exp(-dt / SCROLL_INERTIA_TAU)
                if abs(scroll_inertia_velocity) < SCROLL_INERTIA_MIN:
                    scroll_inertia_velocity = 0.0

        fps_count += 1
        if time.time() - fps_timer >= 1.0:
            # print(f"FPS: {fps_count}")
            fps_count = 0
            fps_timer = time.time()

        cv2.imshow("Gesture Control", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    voice.shutdown()
    stream.release()
    cv2.destroyAllWindows()
    detector.close()
