from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QPushButton, QLabel, QLineEdit,
    QHBoxLayout, QDockWidget, QFormLayout, QGroupBox, QMessageBox,
    QStatusBar, QFileDialog, QScrollArea, QApplication, QInputDialog
)
from PySide6.QtCore import Qt, QTimer, Slot
import cv2
import os
import numpy as np
import time
from polar_cam.image_display import Display
from polar_cam.utils import adjust_for_increment, adjust_rectangle

class MainWindow(QMainWindow):
    def __init__(self, camera_control, image_processor, data_analyzer):
        super().__init__()
        self.camera_control = camera_control
        self.image_processor = image_processor
        self.data_analyzer = data_analyzer

        self.is_recording = False
        self.spots = []
        self.blobs = []
        self.recorded_frames = []
        self.spots_to_process = []
        self.spot_timestamps_storage = {}
        self.spot_intensities_storage = {}
        self.spot_image = None
        self.start_time = None
        self.original_settings = None
        self.current_spot_id = None
        self.data_directory = None
        self.sample_counter = 1
        self.current_roi = None

        self.init_camera_parameters()
        self.setup_ui()
        self.connect_signals()
        self.adjust_to_screen_size()
        self.prompt_for_data_directory()

    def init_camera_parameters(self):
        self.min_framerate = None
        self.max_framerate = None
        self.current_framerate = None
        self.min_exposure = None
        self.max_exposure = None
        self.current_exposure = None
        self.min_w = None
        self.max_w = None
        self.inc_w = None
        self.current_w = None
        self.min_h = None
        self.max_h = None
        self.inc_h = None
        self.current_h = None
        self.min_x = None
        self.max_x = None
        self.inc_x = None
        self.current_x = None
        self.min_y = None
        self.max_y = None
        self.inc_y = None
        self.current_y = None
        self.min_analog_gain = None
        self.max_analog_gain = None
        self.current_analog_gain = None
        self.min_digital_gain = None
        self.max_digital_gain = None
        self.current_digital_gain = None

    def setup_ui(self):
        self.setWindowTitle("Camera Control Interface")

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        self.image_display = Display(self)
        layout.addWidget(self.image_display)

        self.setup_parameter_sidebar()
        self.create_statusbar()

        buttons_layout = QHBoxLayout()

        self.start_pause_button = QPushButton("Start Acquisition", self)
        self.start_pause_button.clicked.connect(self.toggle_acquisition)
        self.start_pause_button.setFixedSize(150, 30)
        buttons_layout.addWidget(self.start_pause_button)

        self.record_button = QPushButton("Start Recording", self)
        self.record_button.clicked.connect(self.toggle_recording)
        self.record_button.setFixedSize(150, 30)
        buttons_layout.addWidget(self.record_button)

        toggle_params_button = QPushButton("Toggle Parameters", self)
        toggle_params_button.clicked.connect(self.toggle_parameter_sidebar)
        toggle_params_button.setFixedSize(150, 30)
        buttons_layout.addWidget(toggle_params_button)

        detect_spots_button = QPushButton("Detect Spots", self)
        detect_spots_button.clicked.connect(self.on_detect_spots)
        detect_spots_button.setFixedSize(150, 30)
        buttons_layout.addWidget(detect_spots_button)

        add_spot_button = QPushButton("Add Spot", self)
        add_spot_button.clicked.connect(self.on_add_spot)
        add_spot_button.setFixedSize(150, 30)
        buttons_layout.addWidget(add_spot_button)

        remove_spot_button = QPushButton("Remove Spot", self)
        remove_spot_button.clicked.connect(self.on_remove_spot)
        remove_spot_button.setFixedSize(150, 30)
        buttons_layout.addWidget(remove_spot_button)

        scan_spot_button = QPushButton("Scan Spot", self)
        scan_spot_button.clicked.connect(self.on_scan_spot)
        scan_spot_button.setFixedSize(150, 30)
        buttons_layout.addWidget(scan_spot_button)

        clear_spot_list_button = QPushButton("Clear Spot List", self)
        clear_spot_list_button.clicked.connect(self.on_clear_spot_list)
        clear_spot_list_button.setFixedSize(150, 30)
        buttons_layout.addWidget(clear_spot_list_button)

        layout.addLayout(buttons_layout)

    def prompt_for_data_directory(self):
        directory = QFileDialog.getExistingDirectory(
            self, "Select Directory to Save Data", 
            os.path.join(os.getcwd(), "PolarCam", "data")
        )
        if directory:
            self.data_directory = directory
            QMessageBox.information(
                self, "Directory Selected",
                f"Data will be saved in: {self.data_directory}"
            )
        else:
            QMessageBox.warning(
                self, "No Directory Selected", "No directory was selected.")

    def setup_parameter_sidebar(self):
        self.parameter_sidebar = QDockWidget("Parameters", self)
        self.parameter_sidebar.setAllowedAreas(
            Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

        sidebar_scroll_area = QScrollArea()  
        sidebar_widget = QWidget()
        sidebar_scroll_area.setWidget(sidebar_widget)
        sidebar_scroll_area.setWidgetResizable(True)

        self.parameter_sidebar.setWidget(sidebar_scroll_area)
        sidebar_layout = QVBoxLayout(sidebar_widget)

        self.create_framerate_group(sidebar_layout)
        self.create_exposure_group(sidebar_layout)
        self.create_roi_group(sidebar_layout)
        self.create_roi_offset_group(sidebar_layout)
        self.create_gain_group(sidebar_layout)
        self.create_spot_detection_group(sidebar_layout)

        self.addDockWidget(Qt.LeftDockWidgetArea, self.parameter_sidebar)

    def adjust_to_screen_size(self):
        screen_geometry = QApplication.primaryScreen().availableGeometry()
        self.setGeometry(
            0, 0, screen_geometry.width(), screen_geometry.height())

    def toggle_parameter_sidebar(self):
        self.parameter_sidebar.setVisible(
            not self.parameter_sidebar.isVisible())

    def create_statusbar(self):
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.label_infos = QLabel()
        self.status_bar.addPermanentWidget(self.label_infos)

    def toggle_acquisition(self):
        if self.camera_control.acquisition_running:
            self.spot_image = self.camera_control.get_current_frame()
            self.camera_control.stop_acquisition()
            self.start_pause_button.setText("Start Acquisition")
        else:
            if self.camera_control.start_acquisition():
                self.start_pause_button.setText("Pause Acquisition")

    def toggle_recording(self):
        if self.is_recording:
            self.is_recording = False
            self.record_button.setText("Start Recording")
            self.save_recording()
        else:
            if not self.camera_control.acquisition_running:
                self.toggle_acquisition()
            self.is_recording = True
            self.record_button.setText("Stop Recording")
            self.recorded_frames = []

    def save_recording(self):
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Video", "", "Video Files (*.avi)")
        if not filepath:
            return
                    
        height, width = self.recorded_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = round(self.current_framerate, 2)
        
        video = cv2.VideoWriter(
            filepath, fourcc, fps, (width, height), isColor=False)

        for _, frame in enumerate(self.recorded_frames):            
            video.write(frame)
        
        video.release()

    def on_frame_captured(self, image_np_array):
        if self.is_recording and self.current_spot_id is None:
            self.recorded_frames.append(np.copy(image_np_array))
            
        if self.is_recording and self.current_spot_id is not None:
            elapsed_time = time.perf_counter() - self.start_time

            self.spot_timestamps_storage.setdefault(
                self.current_spot_id, []).append(elapsed_time)

            extracted = self.image_processor.extract_polar_inten(
                image_np_array, self.current_roi)
            for key in self.spot_intensities_storage[
                self.current_spot_id].keys():
                self.spot_intensities_storage[
                    self.current_spot_id][key].append(extracted[key])

    def create_framerate_group(self, layout):
        self.framerate_group = QGroupBox("AcquisitionFrameRate")
        form_layout = QFormLayout()

        self.min_framerate_label = QLabel("Min AcquisitionFrameRate:")
        self.max_framerate_label = QLabel("Max AcquisitionFrameRate:")
        self.framerate_input = QLineEdit()
        self.framerate_input.textChanged.connect(self.validate_framerate_input)
        self.framerate_set_button = QPushButton("Set AcquisitionFrameRate")
        self.framerate_set_button.clicked.connect(self.on_apply_framerate)

        form_layout.addRow(self.min_framerate_label)
        form_layout.addRow(self.max_framerate_label)
        form_layout.addRow("AcquisitionFrameRate", self.framerate_input)
        form_layout.addRow(self.framerate_set_button)

        self.framerate_group.setLayout(form_layout)
        layout.addWidget(self.framerate_group)

    def create_exposure_group(self, layout):
        self.exposure_group = QGroupBox("ExposureTime")
        form_layout = QFormLayout()

        self.min_exposure_label = QLabel("Min ExposureTime:")
        self.max_exposure_label = QLabel("Max ExposureTime:")
        self.exposure_input = QLineEdit()
        self.exposure_input.textChanged.connect(self.validate_exposure_input)
        self.exposure_set_button = QPushButton("Set ExposureTime (ms)")
        self.exposure_set_button.clicked.connect(self.on_apply_exposure)

        form_layout.addRow(self.min_exposure_label)
        form_layout.addRow(self.max_exposure_label)
        form_layout.addRow("ExposureTime (ms)", self.exposure_input)
        form_layout.addRow(self.exposure_set_button)

        self.exposure_group.setLayout(form_layout)
        layout.addWidget(self.exposure_group)

    def create_roi_group(self, layout):
        self.roi_group = QGroupBox("ROI")
        form_layout = QFormLayout()
        
        self.min_w_label = QLabel("Min Width:")
        self.max_w_label = QLabel("Max Width:")
        self.w_input = QLineEdit()
        self.w_input.textChanged.connect(self.validate_roi_input)
        self.min_h_label = QLabel("Min Height:")
        self.max_h_label = QLabel("Max Height:")
        self.h_input = QLineEdit()
        self.h_input.textChanged.connect(self.validate_roi_input)
        self.roi_set_button = QPushButton("Set ROI")
        self.roi_set_button.clicked.connect(self.on_apply_roi)

        form_layout.addRow(self.min_w_label)
        form_layout.addRow(self.max_w_label)
        form_layout.addRow("Width", self.w_input)
        form_layout.addRow(self.min_h_label)
        form_layout.addRow(self.max_h_label)
        form_layout.addRow("Height", self.h_input)
        form_layout.addRow(self.roi_set_button)

        self.roi_group.setLayout(form_layout)
        layout.addWidget(self.roi_group)

    def create_roi_offset_group(self, layout):   
        self.roi_offset_group = QGroupBox("ROI offset")
        form_layout = QFormLayout()
        
        self.min_x_label = QLabel("Min OffsetX:")
        self.max_x_label = QLabel("Max OffsetX:")
        self.x_input = QLineEdit()
        self.x_input.textChanged.connect(self.validate_roi_offset_input)
        self.min_y_label = QLabel("Min OffsetY:")
        self.max_y_label = QLabel("Max OffsetY:")
        self.y_input = QLineEdit()
        self.y_input.textChanged.connect(self.validate_roi_offset_input)
        self.roi_offset_set_button = QPushButton("Set ROI offset")
        self.roi_offset_set_button.clicked.connect(self.on_apply_roi_offset)

        form_layout.addRow(self.min_x_label)
        form_layout.addRow(self.max_x_label)
        form_layout.addRow("OffsetX", self.x_input)
        form_layout.addRow(self.min_y_label)
        form_layout.addRow(self.max_y_label)
        form_layout.addRow("OffsetY", self.y_input)
        form_layout.addRow(self.roi_offset_set_button)

        self.roi_offset_group.setLayout(form_layout)
        layout.addWidget(self.roi_offset_group)

    def create_gain_group(self, layout):
        self.gain_group = QGroupBox("Gain")
        form_layout = QFormLayout()
        
        self.min_analog_gain_label = QLabel("Min AnalogGain:")
        self.max_analog_gain_label = QLabel("Max AnalogGain:")
        self.analog_gain_input = QLineEdit()
        self.analog_gain_input.textChanged.connect(self.validate_gain_input)
        self.min_digital_gain_label = QLabel("Min DigitalGain:")
        self.max_digital_gain_label = QLabel("Max DigitalGain:")
        self.digital_gain_input = QLineEdit()
        self.digital_gain_input.textChanged.connect(self.validate_gain_input)
        self.gain_set_button = QPushButton("Set Gain")
        self.gain_set_button.clicked.connect(self.on_apply_gain)

        form_layout.addRow(self.min_analog_gain_label)
        form_layout.addRow(self.max_analog_gain_label)
        form_layout.addRow("AnalogGain", self.analog_gain_input)
        form_layout.addRow(self.min_digital_gain_label)
        form_layout.addRow(self.max_digital_gain_label)
        form_layout.addRow("DigitalGain", self.digital_gain_input)
        form_layout.addRow(self.gain_set_button)

        self.gain_group.setLayout(form_layout)
        layout.addWidget(self.gain_group)

    def create_spot_detection_group(self, layout):
        self.spot_detection_group = QGroupBox("Spot Detection Parameters")
        form_layout = QFormLayout()

        self.min_sigma_input = QLineEdit("10")
        self.max_sigma_input = QLineEdit("30")
        self.num_sigma_input = QLineEdit("10")
        self.threshold_input = QLineEdit("0.3")
        self.reset_spot_detection_button = QPushButton("Reset to Defaults")
        self.reset_spot_detection_button.clicked.connect(
            self.reset_spot_detection_parameters)

        form_layout.addRow("Min Sigma", self.min_sigma_input)
        form_layout.addRow("Max Sigma", self.max_sigma_input)
        form_layout.addRow("Number Sigma", self.num_sigma_input)
        form_layout.addRow("Threshold", self.threshold_input)
        form_layout.addRow(self.reset_spot_detection_button)

        self.spot_detection_group.setLayout(form_layout)
        layout.addWidget(self.spot_detection_group)

    def on_apply_framerate(self):
        try:
            input_framerate = float(self.framerate_input.text())
            self.camera_control.set_parameters(
                {"AcquisitionFrameRate": {"current": input_framerate}})
        except Exception as e:
            QMessageBox.critical(
                self, "Error Setting AcquisitionFrameRate", 
                str(e), QMessageBox.Ok)

    def validate_framerate_input(self):
        self.framerate_set_button.setEnabled(False)

        if self.min_framerate is None or self.max_framerate is None:
            return

        try:
            input_framerate = float(self.framerate_input.text())
            is_framerate_valid = (
                self.min_framerate <= input_framerate <= self.max_framerate)

            is_decimal_valid = True
            if "." in self.framerate_input.text():
                fraction_part = self.framerate_input.text().split(".")[1]
                is_decimal_valid = len(fraction_part) <= 2

            if is_framerate_valid and is_decimal_valid:
                self.framerate_set_button.setEnabled(True)
        except ValueError:
            pass

    def on_apply_exposure(self):
        try:
            input_exposure = float(self.exposure_input.text())
            input_exposure_us = input_exposure * 1000
            self.camera_control.set_parameters(
                {"ExposureTime": {"current": input_exposure_us}})
        except Exception as e:
            QMessageBox.critical(
                self, "Error Setting ExposureTime", str(e), QMessageBox.Ok)

    def validate_exposure_input(self):
        self.exposure_set_button.setEnabled(False)

        if self.min_exposure is None or self.max_exposure is None:
            return

        try:
            input_exposure = float(self.exposure_input.text())
            is_exposure_valid = (
                self.min_exposure <= input_exposure <= self.max_exposure)

            is_decimal_valid = True
            if "." in self.exposure_input.text():
                fraction_part = self.exposure_input.text().split(".")[1]
                is_decimal_valid = len(fraction_part) <= 2

            if is_exposure_valid and is_decimal_valid:
                self.exposure_set_button.setEnabled(True)
        except ValueError:
            pass

    def on_apply_roi(self):
        try:
            w = int(self.w_input.text())
            h = int(self.h_input.text())
            
            adjusted_w = adjust_for_increment(w, self.inc_w, self.max_w)
            adjusted_h = adjust_for_increment(h, self.inc_h, self.max_h)
            
            self.w_input.setText(str(adjusted_w))
            self.h_input.setText(str(adjusted_h))
            
            self.camera_control.set_parameters({
                "Width": {"current": adjusted_w}, 
                "Height": {"current": adjusted_h}
            })
        except Exception as e:
            QMessageBox.critical(
                self, "Error Setting ROI", str(e), QMessageBox.Ok)

    def validate_roi_input(self):
        self.roi_set_button.setEnabled(False)

        if any(value is None for value in [
                self.min_w, self.max_w, self.inc_w,
                self.min_h, self.max_h, self.inc_h
            ]):
            return

        try:
            w = int(self.w_input.text())
            h = int(self.h_input.text())
            is_w_valid = self.min_w <= w <= self.max_w
            is_h_valid = self.min_h <= h <= self.max_h

            if is_w_valid and is_h_valid:
                self.roi_set_button.setEnabled(True)
        except ValueError:
            pass

    def on_apply_roi_offset(self):
        try:
            x = int(self.x_input.text())
            y = int(self.y_input.text())
            
            adjusted_x = adjust_for_increment(x, self.inc_x, self.max_x)
            adjusted_y = adjust_for_increment(y, self.inc_y, self.max_y)
            
            self.x_input.setText(str(adjusted_x))
            self.y_input.setText(str(adjusted_y))
            
            self.camera_control.set_parameters({
                "OffsetX": {"current": adjusted_x}, 
                "OffsetY": {"current": adjusted_y}
            })
        except Exception as e:
            QMessageBox.critical(
                self, "Error Setting ROI offset", str(e), QMessageBox.Ok)

    def validate_roi_offset_input(self):
        self.roi_offset_set_button.setEnabled(False)

        if any(value is None for value in [
            self.min_x, self.max_x, self.inc_x, 
            self.min_y, self.max_y, self.inc_y
        ]):
            return

        try:
            x = int(self.x_input.text())
            y = int(self.y_input.text())
            is_x_valid = self.min_x <= x <= self.max_x
            is_y_valid = self.min_y <= y <= self.max_y

            if is_x_valid and is_y_valid:
                self.roi_offset_set_button.setEnabled(True)
        except ValueError:
            pass

    def on_apply_gain(self):
        try:
            input_analog_gain = float(self.analog_gain_input.text())
            input_digital_gain = float(self.digital_gain_input.text())
            
            self.camera_control.set_parameters({
                "AnalogGain": {"current": input_analog_gain}, 
                "DigitalGain": {"current": input_digital_gain}
            })
        except Exception as e:
            QMessageBox.critical(
                self, "Error Setting Gain", str(e), QMessageBox.Ok)

    def validate_gain_input(self):
        self.gain_set_button.setEnabled(False)
        
        if (self.min_analog_gain is None or self.max_analog_gain is None or 
            self.min_digital_gain is None or self.max_digital_gain is None):
            return
        
        try:
            input_analog_gain = float(self.analog_gain_input.text())
            input_digital_gain = float(self.digital_gain_input.text())
            
            is_analog_valid = (
                self.min_analog_gain <= input_analog_gain 
                <= self.max_analog_gain
            )
            is_digital_valid = (
                self.min_digital_gain <= input_digital_gain 
                <= self.max_digital_gain
            )
            
            is_analog_decimal_valid = True
            is_digital_decimal_valid = True
            if "." in self.analog_gain_input.text():
                fraction_part_analog = (
                    self.analog_gain_input.text().split(".")[1])
                is_analog_decimal_valid = len(fraction_part_analog) <= 2
            if "." in self.digital_gain_input.text():
                fraction_part_digital = (
                    self.digital_gain_input.text().split(".")[1])
                is_digital_decimal_valid = len(fraction_part_digital) <= 2
            
            if (is_analog_valid and is_digital_valid and 
                is_analog_decimal_valid and is_digital_decimal_valid):
                self.gain_set_button.setEnabled(True)
        except ValueError:
            pass

    def on_detect_spots(self):
        if self.camera_control.acquisition_running:
            self.toggle_acquisition()

        min_sigma = float(self.min_sigma_input.text())
        max_sigma = float(self.max_sigma_input.text())
        num_sigma = int(self.num_sigma_input.text())
        threshold = float(self.threshold_input.text())

        print(f"Detecting spots")

        blobs = self.image_processor.detect_spots(
            self.spot_image, min_sigma, max_sigma, num_sigma, threshold)

        print(f"Detected {len(blobs)} spots")

        self.image_processor.generate_highlighted_image(
            self.spot_image, blobs, self.data_directory)

        if not blobs:
            QMessageBox.information(self, "Info", "No valid spots detected.")
            return

        self.blobs = blobs
        self.spots = self.convert_blobs_to_spots(blobs)

    def convert_blobs_to_spots(self, blobs):
        spots = []
        unique_id = 0
        for blob in blobs:
            y, x, r = blob
            w = h = int(r * 2)
            x = int(x - r)
            y = int(y - r)
            rect_x, rect_y, rect_w, rect_h = adjust_rectangle(x, y, w, h)
            spots.append({
                'id': unique_id, 
                'x': rect_x, 
                'y': rect_y, 
                'width': rect_w, 
                'height': rect_h
            })
            unique_id += 1
        return spots

    def on_add_spot(self):
        if self.camera_control.acquisition_running:
            self.toggle_acquisition()
        
        self.image_display.set_mouse_press_callback(self.add_spot_at)
        self.status_bar.showMessage("Click on the image to add a spot.")

    def on_remove_spot(self):
        if self.camera_control.acquisition_running or not self.blobs:
            return

        self.image_display.set_mouse_press_callback(self.remove_spot_at)
        self.status_bar.showMessage("Click on the spot to remove.")

    def add_spot_at(self, x, y):
        square_size = 32
        half_square = square_size // 2
        y_start = int(max(y - half_square, 0))
        y_end = int(min(y + half_square, self.spot_image.shape[0]))
        x_start = int(max(x - half_square, 0))
        x_end = int(min(x + half_square, self.spot_image.shape[1]))
        
        roi = self.spot_image[y_start:y_end, x_start:x_end]

        test_params = [(10, 30, 10, 0.3), (5, 40, 5, 0.2), (2, 50, 2, 0.1)]
        for params in test_params:
            blobs = self.image_processor.detect_spots(roi, *params)

            if blobs:
                for blob in blobs:
                    blob[0] += y_start
                    blob[1] += x_start
                
                self.blobs.extend(blobs)
                self.spots = self.convert_blobs_to_spots(self.blobs)
                self.image_processor.generate_highlighted_image(
                    self.spot_image, self.blobs, self.data_directory)

                self.status_bar.showMessage("Spot added successfully.")
                self.image_display.set_mouse_press_callback(None)
                return

        QMessageBox.information(
            self, "Info", "No spot detected at the selected location.")
        self.image_display.set_mouse_press_callback(None)

    def remove_spot_at(self, x, y):
        if not self.blobs:
            self.image_display.set_mouse_press_callback(None)
            return
        
        closest_blob = min(
            self.blobs, key=lambda blob: (blob[1] - x) ** 2 + 
            (blob[0] - y) ** 2)
        self.blobs.remove(closest_blob)
        self.spots = self.convert_blobs_to_spots(self.blobs)
        
        self.image_processor.generate_highlighted_image(
            self.spot_image, self.blobs, self.data_directory)
        self.status_bar.showMessage("Spot removed successfully.")
        
        self.image_display.set_mouse_press_callback(None)

    def on_clear_spot_list(self):
        self.blobs.clear()
        self.spots.clear()
        self.status_bar.showMessage("Spot list cleared.")
        self.toggle_acquisition()

    def reset_spot_detection_parameters(self):
        self.min_sigma_input.setText("10")
        self.max_sigma_input.setText("30")
        self.num_sigma_input.setText("10")
        self.threshold_input.setText("0.3")

    def on_scan_spot(self):
        if not self.spots:
            QMessageBox.warning(
                self, "Warning", 
                "No spots detected or spot detection not yet performed.")
            return

        if not self.camera_control.acquisition_running:
            self.toggle_acquisition()

        self.original_settings = self.save_original_camera_settings()
        self.spots_to_process = self.spots.copy()
        self.sample_folder = os.path.join(
            self.data_directory, f"Sample {self.sample_counter}")
        os.makedirs(self.sample_folder, exist_ok=True)

        duration, ok = QInputDialog.getInt(
            self, "Scan Duration", "Enter the scan duration in milliseconds:", 
            value=10000, min=1000, max=6000000, step=1000
        )

        if ok:
            self.scan_duration = duration
            self.process_next_spot()
        else:
            QMessageBox.information(self, "Info", "Spot scanning canceled.")

    def process_next_spot(self):
        if self.spots_to_process:
            spot = self.spots_to_process.pop(0)
            print(f"Processing spot: {spot}")
            QTimer.singleShot(
                100, lambda: self.start_spot_recording(spot['id'])
            )
        else:
            self.restore_camera_settings(self.original_settings)
            self.analyze_all_spot_data()
            self.sample_counter += 1
            self.spots_to_process = []
            self.spot_timestamps_storage = {}
            self.spot_intensities_storage = {}
            self.current_roi = None
            QMessageBox.information(self, "Info", "Spot scanning completed.")

    def save_original_camera_settings(self):
        original_settings = {}
        try:
            original_settings['roi_x'] = self.current_x
            original_settings['roi_y'] = self.current_y
            original_settings['roi_width'] = self.current_w
            original_settings['roi_height'] = self.current_h
            original_settings['exposure'] = self.current_exposure
            original_settings['framerate'] = self.current_framerate
        except ValueError as e:
            QMessageBox.critical(
                self, "Error", "Invalid camera settings in input fields.")
            return None

        return original_settings

    def adjust_camera_to_spot(self, spot):
        FULL_FRAME_WIDTH = 2464
        FULL_FRAME_HEIGHT = 2056
        MIN_ROI_WIDTH = 256
        MIN_ROI_HEIGHT = 2
        STEP_INCREMENT_X = 4
        STEP_INCREMENT_Y = 2

        desired_width = max(MIN_ROI_WIDTH, spot['width'])
        desired_height = max(MIN_ROI_HEIGHT, spot['height'])
        desired_width = (
            ((desired_width + STEP_INCREMENT_X - 1) // STEP_INCREMENT_X) 
            * STEP_INCREMENT_X
        )
        desired_height = (
            ((desired_height + STEP_INCREMENT_Y - 1) // STEP_INCREMENT_Y)
            * STEP_INCREMENT_Y
        )
        desired_width = min(desired_width, FULL_FRAME_WIDTH)
        desired_height = min(desired_height, FULL_FRAME_HEIGHT)

        offset_x = max(
            0, 
            spot['x'] + self.original_settings['roi_x'] - 
            (desired_width - spot['width']) // 2
        )
        offset_y = max(
            0, 
            spot['y'] + self.original_settings['roi_y'] - 
            (desired_height - spot['height']) // 2
        )
        offset_x -= offset_x % STEP_INCREMENT_X
        offset_y -= offset_y % STEP_INCREMENT_Y
        offset_x = min(offset_x, FULL_FRAME_WIDTH - desired_width)
        offset_y = min(offset_y, FULL_FRAME_HEIGHT - desired_height)

        self.camera_control.set_parameters({
            "OffsetX": {"current": self.min_x},
            "OffsetY": {"current": self.min_y}
        })
        self.camera_control.set_parameters({
            "Width": {"current": self.min_w},
            "Height": {"current": self.min_h}
        })

        self.camera_control.set_parameters({
            "Width": {"current": desired_width}, 
            "Height": {"current": desired_height}
        })
        
        self.camera_control.set_parameters({
            "OffsetX": {"current": offset_x},
            "OffsetY": {"current": offset_y}
        })

    def minimize_exposure(self):
        self.camera_control.set_parameters({
            "ExposureTime": {"current": self.min_exposure * 1000}})

    def maximize_framerate(self):
        self.camera_control.set_parameters({
            "AcquisitionFrameRate": {"current": self.max_framerate}})

    def adjust_gain_to_target(self, max_pixel_value, target_value=150):
        if max_pixel_value == 0:
            return

        adjustment_made = False

        while True:
            current_analog_gain = self.current_analog_gain
            current_digital_gain = self.current_digital_gain
            print(
                "Current gains: "
                f"Analog Gain = {round(current_analog_gain, 2)}, "
                f"Digital Gain = {round(current_digital_gain, 2)}"
            )
            current_total_gain = current_analog_gain * current_digital_gain
            required_total_gain = round(
                (target_value / max_pixel_value) * current_total_gain, 2)

            new_analog_gain = min(
                max(required_total_gain, self.min_analog_gain), 
                self.max_analog_gain
            )
            new_digital_gain = round(required_total_gain / new_analog_gain, 2)
            new_digital_gain = min(
                max(new_digital_gain, self.min_digital_gain), 
                self.max_digital_gain
            )

            if (
                current_analog_gain == new_analog_gain 
                and current_digital_gain == new_digital_gain
            ):
                break

            self.camera_control.set_parameters({
                "AnalogGain": {"current": new_analog_gain},
                "DigitalGain": {"current": new_digital_gain}
            })

            time.sleep(0.5)
            
            print(
                f"Adjusted gains: Analog Gain = {round(new_analog_gain, 2)}, "
                f"Digital Gain = {new_digital_gain}"
            )

            frame = self.camera_control.get_current_frame()
            max_pixel_value = np.max(frame)
            print(f"New max pixel value after adjustment: {max_pixel_value}")

            adjustment_made = True

            if max_pixel_value <= 220:
                break

            if (
                new_analog_gain <= self.min_analog_gain 
                and new_digital_gain <= self.min_digital_gain 
                and max_pixel_value < target_value
            ):
                print(
                    "Both gains are at their minimum values and intensity is "
                    "below target. Exiting adjustment loop."
                )
                break

            if (
                new_analog_gain >= self.max_analog_gain 
                and new_digital_gain >= self.max_digital_gain 
                and max_pixel_value > target_value
            ):
                print(
                    "Both gains are at their maximum values and intensity is "
                    "above target. Exiting adjustment loop."
                )
                break

        if not adjustment_made:
            print(f"Max pixel value already optimal: {max_pixel_value}")
        print(f"Final max pixel value after adjustments: {max_pixel_value}")

    def scan_roi_and_adjust_gain(self, num_frames=10, target_value=150):
        max_pixel_value = 0

        for _ in range(num_frames):
            frame = self.camera_control.get_current_frame()
            max_pixel_value = max(max_pixel_value, np.max(frame))
            time.sleep(0.1)

        print(f"Max pixel value after {num_frames} frames: {max_pixel_value}")

        self.adjust_gain_to_target(max_pixel_value, target_value)
        
        self.start_time = time.perf_counter()
        self.is_recording = True
        QTimer.singleShot(
            self.scan_duration, 
            lambda: self.stop_spot_recording(self.current_spot_id))

    def start_spot_recording(self, spot_id):
        if self.current_spot_id is not None:
            return

        self.current_spot_id = spot_id
        self.spot_timestamps_storage[spot_id] = []
        self.spot_intensities_storage[spot_id] = {
            '90': [], '45': [], '135': [], '0': []}

        spot = next((s for s in self.spots if s['id'] == spot_id), None)
        if spot:
            self.adjust_camera_to_spot(spot)
            self.minimize_exposure()
            self.maximize_framerate()

            calculated_x = spot['x'] + self.original_settings[
                'roi_x'] - self.current_x
            calculated_y = spot['y'] + self.original_settings[
                'roi_y'] - self.current_y
            self.current_roi = {
                'x': calculated_x,
                'y': calculated_y,
                'width': spot['width'],
                'height': spot['height']
            }

        QTimer.singleShot(1000, self.scan_roi_and_adjust_gain)

    def stop_spot_recording(self, spot_id):
        if self.current_spot_id == spot_id:
            spot = next((s for s in self.spots if s['id'] == spot_id), None)
            if not spot:
                print(f"Spot {spot_id} not found.")
                return
            
            timestamps = self.spot_timestamps_storage.get(spot_id, [])
            intensities = self.spot_intensities_storage.get(spot_id, {})

            self.save_spot_data(spot_id, intensities, timestamps)

            self.current_spot_id = None
            self.is_recording = False
            self.process_next_spot()

    def save_spot_data(self, spot_id, intensities, timestamps):
        raw_data_file = os.path.join(
            self.sample_folder, f"spot_{spot_id}_data.npz")
        
        np.savez_compressed(
            raw_data_file, intensities=intensities, timestamps=timestamps)

    def analyze_all_spot_data(self):
        for file in os.listdir(self.sample_folder):
            if file.endswith('_data.npz'):
                raw_data_file = os.path.join(self.sample_folder, file)
                data = np.load(raw_data_file)
                intensities = data['intensities'].item()
                timestamps = data['timestamps']
                spot_id = int(file.split('_')[1])

                self.data_analyzer.analyze(
                    intensities, timestamps, spot_id, self.sample_folder)
        
        print("All data analyzed.")
        QMessageBox.information(self, "Info", "All data analyzed.")

    def restore_camera_settings(self, settings):
        required_keys = [
            'roi_x', 'roi_y', 'roi_width', 'roi_height', 
            'exposure', 'framerate'
        ]
        if not all(key in settings for key in required_keys):
            QMessageBox.critical(
                self, "Error", "Incomplete camera settings to restore."
            )
            return
        
        try:
            self.camera_control.set_parameters({
                "OffsetX": {"current": self.min_x},
                "OffsetY": {"current": self.min_y}
            })
            self.camera_control.set_parameters({
                "Width": {"current": self.min_w},
                "Height": {"current": self.min_h}
            })
            
            self.camera_control.set_parameters({
                "OffsetX": {"current": settings['roi_x']},
                "OffsetY": {"current": settings['roi_y']}
            })            
            self.camera_control.set_parameters({
                "Width": {"current": settings['roi_width']}, 
                "Height": {"current": settings['roi_height']}
            })
        
            self.camera_control.set_parameters({
                "ExposureTime": {
                    "current": settings['exposure'] * 1000
                }
            })
            self.camera_control.set_parameters({
                "AcquisitionFrameRate": {
                    "current": settings['framerate']
                }
            })
        except ValueError as e:
            QMessageBox.critical(
                self, "Error", f"Failed to restore camera settings: {e}")

    def connect_signals(self):
        self.camera_control.image_acquired.connect(
            self.image_display.on_image_received)
        self.camera_control.parameter_updated.connect(
            self.on_parameter_updated)
        self.camera_control.acquisition_updated.connect(
            self.on_acquisition_updated)
        self.camera_control.camera_error.connect(self.on_camera_error)
        self.camera_control.acquisition_started.connect(
            self.on_acquisition_started)
        self.camera_control.acquisition_stopped.connect(
            self.on_acquisition_stopped)
        self.camera_control.frame_captured.connect(self.on_frame_captured)
        self.image_processor.image_processed.connect(self.image_display.on_image_received)

    def closeEvent(self, event):
        self.cleanup()
        event.accept()

    def cleanup(self):
        self.camera_control.destroy_all()
        QApplication.instance().quit()

    @Slot(dict)
    def on_parameter_updated(self, parameters):
        try:
            framerate_info = parameters.get("AcquisitionFrameRate", {})
            self.min_framerate = framerate_info.get('min', None)
            self.max_framerate = framerate_info.get('max', None)
            self.current_framerate = framerate_info.get('current', None)
            self.min_framerate_label.setText(
                f"Min Frame Rate: {round(self.min_framerate, 2)} fps")
            self.max_framerate_label.setText(
                f"Max Frame Rate: {round(self.max_framerate, 2)} fps")
            self.framerate_input.setText(str(round(self.current_framerate, 2)))
            
            exposure_info = parameters.get("ExposureTime", {})
            self.min_exposure = exposure_info.get('min', None) / 1000
            self.max_exposure = exposure_info.get('max', None) / 1000
            self.min_exposure_label.setText(
                f"Min Exposure: {round(self.min_exposure, 2)} ms")
            self.max_exposure_label.setText(
                f"Max Exposure: {round(self.max_exposure, 2)} ms")
            self.current_exposure = exposure_info.get('current', None) / 1000
            self.exposure_input.setText(str(round(self.current_exposure, 2)))
            
            w_info = parameters.get("Width", {})
            self.min_w = w_info.get('min', None)
            self.max_w = w_info.get('max', None)
            self.inc_w = w_info.get('increment', None)
            self.current_w = w_info.get('current', None)
            self.min_w_label.setText(f"Min Width: {self.min_w}")
            self.max_w_label.setText(f"Max Width: {self.max_w}")
            self.w_input.setText(str(self.current_w))
            
            h_info = parameters.get("Height", {})
            self.min_h = h_info.get('min', None)
            self.max_h = h_info.get('max', None)
            self.inc_h = h_info.get('increment', None)
            self.current_h = h_info.get('current', None)
            self.min_h_label.setText(f"Min Height: {self.min_h}")
            self.max_h_label.setText(f"Max Height: {self.max_h}")
            self.h_input.setText(str(self.current_h))
            
            x_info = parameters.get("OffsetX", {})
            self.min_x = x_info.get('min', None)
            self.max_x = x_info.get('max', None)
            self.inc_x = x_info.get('increment', None)
            self.current_x = x_info.get('current', None)
            self.min_x_label.setText(f"Min OffsetX: {self.min_x}")
            self.max_x_label.setText(f"Max OffsetX: {self.max_x}")
            self.x_input.setText(str(self.current_x))
            
            y_info = parameters.get("OffsetY", {})
            self.min_y = y_info.get('min', None)
            self.max_y = y_info.get('max', None)
            self.inc_y = y_info.get('increment', None)
            self.current_y = y_info.get('current', None)
            self.min_y_label.setText(f"Min OffsetY: {self.min_y}")
            self.max_y_label.setText(f"Max OffsetY: {self.max_y}")
            self.y_input.setText(str(self.current_y))
            
            analog_info = parameters.get("AnalogGain", {})
            self.min_analog_gain = analog_info.get('min', None)
            self.max_analog_gain = analog_info.get('max', None)
            self.current_analog_gain = analog_info.get('current', None)
            self.min_analog_gain_label.setText(
                f"Min AnalogGain: {round(self.min_analog_gain, 2)}")
            self.max_analog_gain_label.setText(
                f"Max AnalogGain: {round(self.max_analog_gain, 2)}")
            self.analog_gain_input.setText(
                str(round(self.current_analog_gain, 2)))
            
            digital_info = parameters.get("DigitalGain", {})
            self.min_digital_gain = digital_info.get('min', None)
            self.max_digital_gain = digital_info.get('max', None)
            self.current_digital_gain = digital_info.get('current', None)
            self.min_digital_gain_label.setText(
                f"Min DigitalGain: {round(self.min_digital_gain, 2)}")
            self.max_digital_gain_label.setText(
                f"Max DigitalGain: {round(self.max_digital_gain, 2)}")
            self.digital_gain_input.setText(
                str(round(self.current_digital_gain, 2)))
        except Exception as e:
            QMessageBox.critical(
                self, "Parameter Update Error",
                f"Error updating UI with new parameters: {str(e)}",
                QMessageBox.Ok
            )

    @Slot(int, int)
    def on_acquisition_updated(self, frame_counter, error_counter):
        fps = round(self.current_framerate, 2)
        self.label_infos.setText(
            f"Acquired: {frame_counter}, Errors: {error_counter}, fps: {fps}")

    @Slot(str)
    def on_camera_error(self, error_message):
        QMessageBox.critical(
            self, "Camera Error", error_message, QMessageBox.Ok)

    @Slot()
    def on_acquisition_started(self):
        self.start_pause_button.setText("Pause Acquisition")

    @Slot()
    def on_acquisition_stopped(self):
        self.start_pause_button.setText("Start Acquisition")
