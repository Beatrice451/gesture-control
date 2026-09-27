import time

from PySide6.QtWidgets import QApplication

from screenshot_selector import ScreenshotSelector


app = QApplication.instance() or QApplication([])

selector = ScreenshotSelector()

print("Starting selection...")

selector.start(300, 200)

for x in range(300, 900, 10):
    y = 200 + int((x - 300) * 0.5)

    selector.update(x, y)

    time.sleep(0.01)

selector.finish(900, 500)

print("Done.")