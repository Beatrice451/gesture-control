from PyQt6.QtWidgets import QApplication, QWidget
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPainter, QColor, QPen


class Overlay(QWidget):
    def __init__(self, screen_w: int, screen_h: int):
        super().__init__()
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.dot_x = 0
        self.dot_y = 0

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool                        # не показывать в таскбаре
            | Qt.WindowType.WindowTransparentForInput   # клики/тапы проходят сквозь
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)
        self.setGeometry(0, 0, screen_w, screen_h)
        self.show()

    def update_dot(self, x: float, y: float) -> None:
        self.dot_x = int(x)
        self.dot_y = int(y)
        self.update()   # попросить Qt перерисовать виджет

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # жёлтое кольцо + красная заливка
        painter.setPen(QPen(QColor(255, 255, 0), 2))
        painter.setBrush(QColor(255, 0, 0))
        painter.drawEllipse(self.dot_x - 8, self.dot_y - 8, 16, 16)