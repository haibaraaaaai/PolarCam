import math
from PySide6.QtWidgets import QGraphicsView, QGraphicsScene, QWidget
from PySide6.QtGui import QImage, QPainter
from PySide6.QtCore import QRectF, Slot

class Display(QGraphicsView):
    def __init__(self, parent: QWidget=None):
        super().__init__(parent)
        self._scene = CustomGraphicsScene(self)
        self.setScene(self._scene)

    @Slot(QImage)
    def on_image_received(self, image: QImage):
        self._scene.set_image(image)
        self.update()

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
        
        if image_width == 0 or image_height == 0:
            return
            
        ratio1 = display_width / display_height
        ratio2 = image_width / image_height
        
        if ratio1 > ratio2:
            image_width = display_height * ratio2
            image_height = display_height
        else:
            image_width = display_width
            image_height = display_height / ratio2

        image_pos_x = -1.0 * (image_width / 2.0)
        image_pos_y = -1.0 * (image_height / 2.0)
        image_pos_x = math.trunc(image_pos_x)
        image_pos_y = math.trunc(image_pos_y)
        
        rect = QRectF(image_pos_x, image_pos_y, image_width, image_height)
        painter.drawImage(rect, self._image)
