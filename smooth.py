import threading
import time


class SmoothMouse:
    def __init__(self, mouse, smoothing=0.25, fps=60):
        self.mouse = mouse
        self.smoothing = smoothing
        self.interval = 1.0 / fps

        self.lock = threading.Lock()

        self.target_x, self.target_y = mouse.position
        self.current_x, self.current_y = mouse.position

        self.running = True

        self.thread = threading.Thread(
            target=self._update,
            daemon=True
        )
        self.thread.start()

    def freeze(self):
        with self.lock:
            self.target_x = self.current_x
            self.target_y = self.current_y

    def set_target(self, x, y):
        with self.lock:
            self.target_x = x
            self.target_y = y

    def _update(self):
        while self.running:
            with self.lock:
                target_x = self.target_x
                target_y = self.target_y

            # Плавно приближаемся к целевой позиции
            self.current_x += (
                target_x - self.current_x
            ) * self.smoothing

            self.current_y += (
                target_y - self.current_y
            ) * self.smoothing

            self.mouse.position = (
                int(self.current_x),
                int(self.current_y)
            )

            time.sleep(self.interval)

    def stop(self):
        self.running = False
        self.thread.join(timeout=1.0)
