from PySide6.QtWidgets import QApplication
from polar_cam import CameraControl, ImageProcessor, DataAnalyzer, MainWindow
import sys

def main():
    app = QApplication(sys.argv)
   
    camera_control = CameraControl()
    image_processor = ImageProcessor()
    data_analyzer = DataAnalyzer()

    main_window = MainWindow(camera_control, image_processor, data_analyzer)
    main_window.show()
    
    camera_control.initialize_and_start_acquisition()

    sys.exit(app.exec())

if __name__ == "__main__":
    main()
