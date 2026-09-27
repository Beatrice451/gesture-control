# screenshot_selector.py

import io
from datetime import datetime

import pyautogui

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import QApplication, QWidget


class ScreenshotOverlay(QWidget):
    def __init__(self, start_x: int, start_y: int):
        super().__init__()

        self.start_x = start_x
        self.start_y = start_y
        self.end_x = start_x
        self.end_y = start_y

        self.setWindowFlags(
            Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
            | Qt.Tool
            | Qt.WindowTransparentForInput
        )

        self.setAttribute(Qt.WA_TranslucentBackground)

        self.showFullScreen()

    def update_selection(self, x: int, y: int):
        self.end_x = x
        self.end_y = y
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)

        left = min(self.start_x, self.end_x)
        top = min(self.start_y, self.end_y)
        right = max(self.start_x, self.end_x)
        bottom = max(self.start_y, self.end_y)

        # Затемняем всё вокруг выделенной области.
        overlay_color = QColor(0, 0, 0, 100)
        painter.fillRect(0, 0, self.width(), top, overlay_color)
        painter.fillRect(
            0,
            bottom,
            self.width(),
            self.height() - bottom,
            overlay_color,
        )
        painter.fillRect(
            0,
            top,
            left,
            bottom - top,
            overlay_color,
        )
        painter.fillRect(
            right,
            top,
            self.width() - right,
            bottom - top,
            overlay_color,
        )

        # Рамка выделения.
        pen = QPen(QColor(255, 255, 255), 2)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)

        painter.drawRect(
            left,
            top,
            right - left,
            bottom - top,
        )


class ScreenshotSelector:
    def __init__(self):
        self.app = QApplication.instance()

        if self.app is None:
            self.app = QApplication([])

        self.overlay = None

        self.start_x = 0
        self.start_y = 0

    def start(self, x: int, y: int):
        """
        Начинает выделение области скриншота.
        """

        self.start_x = x
        self.start_y = y

        self.overlay = ScreenshotOverlay(x, y)
        self.overlay.show()

        self.app.processEvents()

    def update(self, x: int, y: int):
        """
        Обновляет текущую область выделения.
        """

        if self.overlay is None:
            return

        self.overlay.update_selection(x, y)
        self.app.processEvents()

    def finish(self, x: int, y: int):
        """
        Завершает выделение, сохраняет скриншот
        и копирует его в буфер обмена.
        """

        if self.overlay is None:
            return

        left = min(self.start_x, x)
        top = min(self.start_y, y)
        right = max(self.start_x, x)
        bottom = max(self.start_y, y)

        width = right - left
        height = bottom - top

        # Убираем оверлей до создания скриншота.
        self.overlay.hide()
        self.app.processEvents()

        # Небольшая пауза нужна, чтобы окно гарантированно
        # исчезло с экрана до захвата.
        import time
        time.sleep(0.05)

        if width <= 0 or height <= 0:
            self.overlay = None
            return

        screenshot = pyautogui.screenshot(
            region=(left, top, width, height)
        )

        # Сохраняем в файл.
        # filename = datetime.now().strftime(
        #     "screenshot_%Y%m%d_%H%M%S.png"
        # )
        #
        # screenshot.save(filename)

        # Копируем изображение в буфер обмена.
        buffer = io.BytesIO()
        screenshot.save(buffer, format="PNG")

        qimage = QImage.fromData(
            buffer.getvalue(),
            "PNG",
        )

        self.app.clipboard().setImage(qimage)

        # print(f"Screenshot saved: {filename}")
        print(f"Screenshot saved to buffer")
        print(f"Size: {width}x{height}")

        self.overlay.deleteLater()
        self.overlay = None

        self.app.processEvents()

    def cancel(self):
        """
        Отменяет текущее выделение.
        """

        if self.overlay is None:
            return

        self.overlay.hide()
        self.overlay.deleteLater()
        self.overlay = None

        self.app.processEvents()
