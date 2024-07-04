from PySide6.QtWidgets import QGraphicsView, QGraphicsScene, QWidget
from PySide6.QtGui import QImage, QPainter, QMouseEvent
from PySide6.QtCore import QRectF, Slot
from polar_cam.utils import calculate_image_rect

class Display(QGraphicsView):
    def __init__(self, parent: QWidget = None):
        super().__init__(parent)
        self._scene = CustomGraphicsScene(self)
        self.setScene(self._scene)
        self.mouse_press_callback = None

    @Slot(QImage)
    def on_image_received(self, image: QImage):
        print("on_image_received called.")
        if image.isNull():
            print("Received image is null.")
        else:
            print(f"Received image size: {image.width()}x{image.height()}")
        self._scene.set_image(image)
        self._scene.invalidate(QRectF(), QGraphicsScene.BackgroundLayer)
        self._scene.update()
        self.update()
        print("Display updated with new image.")

    def update_display(self, qimage):
        print("Update display called.")
        self.on_image_received(qimage)
        self._scene.invalidate(QRectF(), QGraphicsScene.BackgroundLayer)  # Force full scene update
        self.viewport().update()  # Ensure the viewport is fully redrawn
        self.repaint()  # Force redraw of the entire display
        print("Display repaint called.")

    def set_mouse_press_callback(self, callback):
        self.mouse_press_callback = callback

class CustomGraphicsScene(QGraphicsScene):
    def __init__(self, parent: Display = None):
        super().__init__(parent)
        self._parent = parent
        self._image = QImage()

    def set_image(self, image: QImage):
        print("set_image called.")
        if image.isNull():
            print("Image set to scene is null.")
        else:
            print(f"Image set to scene size: {image.width()}x{image.height()}")
        self._image = image
        self.clear()  # Clear the scene to remove old items
        self.invalidate(QRectF(), QGraphicsScene.BackgroundLayer)
        self.update(QRectF())  # Ensure full scene update
        print("Scene image updated.")

    def drawBackground(self, painter: QPainter, rect: QRectF):
        print("drawBackground called.")
        if self._image.isNull():
            print("No image to draw.")
            return

        display_width = self._parent.width()
        display_height = self._parent.height()
        image_width = self._image.width()
        image_height = self._image.height()

        print(f"Display size: {display_width}x{display_height}")
        print(f"Image size: {image_width}x{image_height}")

        coords = calculate_image_rect(display_width, display_height, image_width, image_height)
        if coords is not None:
            image_pos_x, image_pos_y, image_width, image_height = coords
            rect = QRectF(image_pos_x, image_pos_y, image_width, image_height)
            painter.drawImage(rect, self._image)
            print(f"Background drawn with image at {rect}.")
        else:
            print("Invalid coordinates for drawing image.")

    def mousePressEvent(self, event: QMouseEvent):
        if self._parent.mouse_press_callback:
            scene_pos = self._parent.mapToScene(event.pos().toPoint())
            self._parent.mouse_press_callback(scene_pos.x(), scene_pos.y())
        super().mousePressEvent(event)
