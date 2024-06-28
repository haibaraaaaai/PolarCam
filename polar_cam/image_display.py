from PySide6.QtWidgets import QGraphicsView, QGraphicsScene, QWidget
from PySide6.QtGui import QImage, QPainter, QMouseEvent
from PySide6.QtCore import QRectF, Slot
from polar_cam.utils import calculate_image_rect

class Display(QGraphicsView):
    def __init__(self, parent: QWidget=None):
        super().__init__(parent)
        self._scene = CustomGraphicsScene(self)
        self.setScene(self._scene)
        self.mouse_press_callback = None

    @Slot(QImage)
    def on_image_received(self, image: QImage):
        self._scene.set_image(image)
        self.update()

    def set_mouse_press_callback(self, callback):
        self.mouse_press_callback = callback

    def mousePressEvent(self, event):
        if self.mouse_press_callback:
            scene_pos = self.mapToScene(event.pos()).toPoint()
            self.mouse_press_callback(scene_pos.x(), scene_pos.y())
        super().mousePressEvent(event)

class CustomGraphicsScene(QGraphicsScene):
    def __init__(self, parent: Display = None):
        super().__init__(parent)
        self._parent = parent
        self._image = QImage()

    def set_image(self, image: QImage):
        self._image = image
        self.update()

    def drawBackground(self, painter: QPainter, rect: QRectF):
        display_width = self._parent.width()
        display_height = self._parent.height()
        image_width = self._image.width()
        image_height = self._image.height()

        coords = calculate_image_rect(
            display_width, display_height, image_width, image_height
        )
        if coords is not None:
            image_pos_x, image_pos_y, image_width, image_height = coords
            rect = QRectF(image_pos_x, image_pos_y, image_width, image_height)
            painter.drawImage(rect, self._image)

    def mousePressEvent(self, event: QMouseEvent):
        pos = event.pos()
        scene_pos = self._parent.mapToScene(pos).toPoint()
        pixel_x, pixel_y = scene_pos.x(), scene_pos.y()
        if self._parent.mousePressEventCallback:
            self._parent.mousePressEventCallback(pixel_x, pixel_y)
        super().mousePressEvent(event)