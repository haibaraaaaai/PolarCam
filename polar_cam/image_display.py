from PySide6.QtWidgets import QGraphicsView, QGraphicsScene, QWidget
from PySide6.QtGui import QImage, QPainter, QMouseEvent
from PySide6.QtCore import QRectF, Slot
from polar_cam.utils import calculate_image_rect

class Display(QGraphicsView):
    def __init__(self, parent: QWidget = None):
        super().__init__(parent)
        self._scene = CustomGraphicsScene(self)
        self.setScene(self._scene)
        self.setMouseTracking(True)
        self.mouse_press_callback = None

    @Slot(QImage)
    def on_image_received(self, image: QImage):
        self._scene.set_image(image)
        self.update()

    def set_mouse_press_callback(self, callback):
        self.mouse_press_callback = callback

class CustomGraphicsScene(QGraphicsScene):
    def __init__(self, parent: Display = None):
        super().__init__(parent)
        self._parent = parent
        self._image = QImage()
        self.im_coord = None

    def set_image(self, image: QImage):
        self._image = image
        self.update()

    def drawBackground(self, painter: QPainter, rect: QRectF):
        display_width = self._parent.width()
        display_height = self._parent.height()
        image_width = self._image.width()
        image_height = self._image.height()

        self.im_coord = calculate_image_rect(
            display_width, display_height, image_width, image_height)
        if self.im_coord is not None:
            image_pos_x, image_pos_y, image_width, image_height = self.im_coord
            rect = QRectF(image_pos_x, image_pos_y, image_width, image_height)
            painter.drawImage(rect, self._image)
            print(f"Image drawn at ({image_pos_x}, {image_pos_y}) with size ({image_width}, {image_height})")

    def mousePressEvent(self, event: QMouseEvent):
        raw_pos = event.pos().toPoint()
        print(f"Raw event position: (x, y) = ({raw_pos.x()}, {raw_pos.y()})")
        super().mousePressEvent(event)
