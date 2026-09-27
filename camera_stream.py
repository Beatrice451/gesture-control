import sys
import threading
import time
import cv2


class CameraStream:
    """Фоновый поток читает кадры с камеры, основной берёт самый свежий."""

    def __init__(self, src=0, w=640, h=480, fps=30, backend=None):
        # Автовыбор бэкенда в зависимости от ОС
        if backend is None:
            if sys.platform.startswith("linux"):
                backend = cv2.CAP_V4L2
            elif sys.platform == "win32":
                backend = cv2.CAP_DSHOW   # DirectShow — быстрее и стабильнее на Windows
            elif sys.platform == "darwin":
                backend = cv2.CAP_AVFOUNDATION  # macOS
            else:
                backend = cv2.CAP_ANY

        self.cap = cv2.VideoCapture(src, backend)

        if not self.cap.isOpened():
            raise RuntimeError(f"Не удалось открыть камеру {src} (backend={backend})")

        # MJPG имеет смысл ставить только на Linux — там это реально снижает задержку.
        # На Windows многие камеры его не поддерживают через DirectShow.
        if sys.platform.startswith("linux"):
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG"))

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._frame = None
        self._frame_id = 0
        self._lock = threading.Lock()
        self._running = True

        self._thread = threading.Thread(target=self._update, daemon=True)
        self._thread.start()

    def _update(self):
        while self._running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.001)
                continue
            with self._lock:
                self._frame = frame
                self._frame_id += 1

    def read_latest(self, last_id):
        """Возвращает (frame, frame_id) если кадр новее last_id, иначе (None, last_id)."""
        with self._lock:
            if self._frame is None or self._frame_id == last_id:
                return None, last_id
            return self._frame.copy(), self._frame_id

    def release(self):
        self._running = False
        self._thread.join(timeout=1.0)
        self.cap.release()
