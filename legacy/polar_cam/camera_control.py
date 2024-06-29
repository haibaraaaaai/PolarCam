from PySide6.QtCore import QObject, Signal, QTimer
import numpy as np
from ids_peak import ids_peak, ids_peak_ipl_extension
from PySide6.QtGui import QImage
from .utils import configure_device_component, fetch_camera_parameters

class CameraControl(QObject):
    image_acquired = Signal(QImage)
    frame_captured = Signal(object)
    camera_error = Signal(str)
    acquisition_updated = Signal(int, int)
    parameter_updated = Signal(dict)
    acquisition_started = Signal()
    acquisition_stopped = Signal()

    def __init__(self):
        super().__init__()
        self.device = None
        self.node_map_remote_device = None
        self.datastream = None
        self.latest_frame = None
        self.acquisition_timer = QTimer()
        self.frame_counter = 0
        self.error_counter = 0
        self.acquisition_running = False
        self.parameters = {
            "AcquisitionFrameRate": {"min": 0, "max": 0, "current": 0},
            "ExposureTime": {"min": 0, "max": 0, "current": 0},
            "OffsetX": {"min": 0, "max": 0, "current": 0, "increment": 0},
            "OffsetY": {"min": 0, "max": 0, "current": 0, "increment": 0},
            "Width": {"min": 0, "max": 0, "current": 0, "increment": 0},
            "Height": {"min": 0, "max": 0, "current": 0, "increment": 0},
            "AnalogGain": {"min": 0, "max": 0, "current": 0},
            "DigitalGain": {"min": 0, "max": 0, "current": 0},
        }
        ids_peak.Library.Initialize()

    def initialize_and_start_acquisition(self):
        if self.open_device():
            self.start_acquisition()
        else:
            self.camera_error.emit("Could not initialize camera.")
            self.close_device()

    def __del__(self):
        self.destroy_all()

    def destroy_all(self):
        self.stop_acquisition()
        self.close_device()
        ids_peak.Library.Close()

    def open_device(self):
        try:
            device_manager = ids_peak.DeviceManager.Instance()
            device_manager.Update()

            if device_manager.Devices().empty():
                self.camera_error.emit("No device found!")
                return False

            for device in device_manager.Devices():
                if device.IsOpenable():
                    self.device = device.OpenDevice(
                        ids_peak.DeviceAccessType_Control)
                    break

            if self.device is None:
                self.camera_error.emit("Device could not be opened!")
                return False

            self.node_map_remote_device = (
                self.device.RemoteDevice().NodeMaps()[0])

            configure_device_component(
                self.node_map_remote_device, self.camera_error)

            datastreams = self.device.DataStreams()
            if datastreams.empty():
                self.camera_error.emit("Device has no DataStream!")
                self.device = None
                return False

            self.datastream = datastreams[0].OpenDataStream()
            self.fetch_initial_parameters()

            return True
        except Exception as e:
            self.camera_error.emit(f"Failed to open device: {str(e)}")
            return False

    def fetch_initial_parameters(self):
        self.parameters = fetch_camera_parameters(
            self.node_map_remote_device, 
            self.parameters, 
            self.camera_error
        )
        if self.parameters and self.acquisition_running:
            current_framerate = self.parameters[
                "AcquisitionFrameRate"]["current"]
            self.acquisition_timer.setInterval(
                1000 / current_framerate if current_framerate > 0 else 1000)

        self.parameter_updated.emit(self.parameters)

    def start_acquisition(self):
        if (
            self.device is None or 
            self.acquisition_running or 
            self.datastream is None
        ):
            return False

        payload_size = (
            self.node_map_remote_device.FindNode("PayloadSize").Value())
        buffer_count_max = self.datastream.NumBuffersAnnouncedMinRequired()
        for _ in range(buffer_count_max):
            buffer = self.datastream.AllocAndAnnounceBuffer(payload_size)
            self.datastream.QueueBuffer(buffer)

        current_framerate = self.parameters["AcquisitionFrameRate"]["current"]
        self.acquisition_timer.setInterval(
            1000 / current_framerate if current_framerate > 0 else 1000)
        self.acquisition_timer.setSingleShot(False)
        self.acquisition_timer.timeout.connect(self.fetch_frame)

        try:
            self.datastream.StartAcquisition()
            self.node_map_remote_device.FindNode("AcquisitionStart").Execute()
            self.node_map_remote_device.FindNode(
                "AcquisitionStart").WaitUntilDone()
        except Exception as e:
            self.camera_error.emit(f"Exception in start_acquisition: {str(e)}")
            return False

        self.acquisition_timer.start()
        self.acquisition_running = True
        self.acquisition_started.emit()
        return True

    def stop_acquisition(self):
        if self.device is None or not self.acquisition_running:
            return

        try:
            self.node_map_remote_device.FindNode("AcquisitionStop").Execute()
            self.datastream.StopAcquisition(
                ids_peak.AcquisitionStopMode_Default)
            self.datastream.Flush(ids_peak.DataStreamFlushMode_DiscardAll)

            for buffer in self.datastream.AnnouncedBuffers():
                self.datastream.RevokeBuffer(buffer)

            self.acquisition_running = False
            self.acquisition_timer.stop()
            self.acquisition_stopped.emit()
        except Exception as e:
            self.camera_error.emit(f"Error stopping acquisition: {str(e)}")

    def close_device(self):
        if self.datastream is not None:
            try:
                for buffer in self.datastream.AnnouncedBuffers():
                    self.datastream.RevokeBuffer(buffer)
            except Exception as e:
                self.camera_error.emit(f"Error closing device: {str(e)}")

        self.device = None

    def fetch_frame(self):
        try:
            buffer = self.datastream.WaitForFinishedBuffer(5000)
            ipl_image = ids_peak_ipl_extension.BufferToImage(buffer)
            self.datastream.QueueBuffer(buffer)

            image_np_array = ipl_image.get_numpy_1D().reshape(
                (ipl_image.Height(), ipl_image.Width()))
            image_qt = QImage(
                image_np_array.data,
                image_np_array.shape[1],
                image_np_array.shape[0],
                QImage.Format_Grayscale8,
            )

            self.latest_frame = np.copy(image_np_array)

            self.image_acquired.emit(image_qt.copy())
            self.frame_captured.emit(self.latest_frame)
            self.frame_counter += 1
            self.acquisition_updated.emit(
                self.frame_counter, self.error_counter)
        except Exception as e:
            self.error_counter += 1
            self.camera_error.emit(f"Error fetching frame: {str(e)}")

    def set_gain(self, gain_type, value):
        try:
            if gain_type in ["AnalogGain", "DigitalGain"]:
                gain_selector_value = (
                    "AnalogAll" if gain_type == "AnalogGain" else "DigitalAll"
                )
                self.node_map_remote_device.FindNode("GainSelector")\
                    .SetCurrentEntry(gain_selector_value)
                self.node_map_remote_device.FindNode("Gain").SetValue(value)
                self.parameters[gain_type]["current"] = value
            else:
                raise ValueError("Invalid gain type specified.")
        except Exception as e:
            self.camera_error.emit(f"Error setting {gain_type}: {str(e)}")

    def set_parameters(self, updates):
        try:
            self.stop_acquisition()
            for parameter_name, values in updates.items():
                if parameter_name in ["AnalogGain", "DigitalGain"]:
                    self.set_gain(parameter_name, values['current'])
                elif parameter_name in self.parameters:
                    current_value = values['current']
                    node = self.node_map_remote_device.FindNode(parameter_name)
                    if node is not None:
                        node.SetValue(current_value)
                    self.parameters[parameter_name]["current"] = current_value
            self.fetch_initial_parameters()
            self.parameter_updated.emit(self.parameters)
            self.start_acquisition()
        except Exception as e:
            self.camera_error.emit(f"Error setting parameters: {str(e)}")

    def get_current_frame(self):
        self.fetch_frame()
        return self.latest_frame
