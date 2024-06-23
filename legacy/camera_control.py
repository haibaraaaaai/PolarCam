from PySide6.QtCore import QObject, Signal, QTimer
import numpy as np
from ids_peak_ipl import ids_peak_ipl
from ids_peak import ids_peak, ids_peak_ipl_extension
from PySide6.QtGui import QImage
import cv2

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
        """Initialize the camera and start acquisition."""
        if self.open_device():
            self.start_acquisition()
        else:
            self.camera_error.emit("Could not initialize camera.")
            self.close_device()

    def __del__(self):
        """Destructor to ensure resources are released."""
        self.destroy_all()

    def destroy_all(self):
        """Release all resources and close the library."""
        self.stop_acquisition()
        self.close_device()
        ids_peak.Library.Close()

    def open_device(self):
        """Open the camera device and configure initial parameters."""
        try:
            device_manager = ids_peak.DeviceManager.Instance()
            device_manager.Update()

            if device_manager.Devices().empty():
                self.camera_error.emit("No device found!")
                return False

            for device in device_manager.Devices():
                if device.IsOpenable():
                    self.device = device.OpenDevice(ids_peak.DeviceAccessType_Control)
                    break

            if self.device is None:
                self.camera_error.emit("Device could not be opened!")
                return False

            self.node_map_remote_device = self.device.RemoteDevice().NodeMaps()[0]

            component_selector = self.node_map_remote_device.FindNode("ComponentSelector").CurrentEntry().SymbolicValue()
            component_enable = self.node_map_remote_device.FindNode("ComponentEnable").Value()

            if component_selector != "Raw" or not component_enable:
                try:
                    self.node_map_remote_device.FindNode("ComponentSelector").SetCurrentEntry("Raw")
                    self.node_map_remote_device.FindNode("ComponentEnable").SetValue(True)
                except Exception as e:
                    self.camera_error.emit(f"Warning: Error Setting Component: {e}")

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
        """Fetch initial parameters from the camera."""
        try:
            self.parameters["AcquisitionFrameRate"]["min"] = self.node_map_remote_device.FindNode("AcquisitionFrameRate").Minimum()
            self.parameters["AcquisitionFrameRate"]["max"] = self.node_map_remote_device.FindNode("AcquisitionFrameRate").Maximum()
            self.parameters["AcquisitionFrameRate"]["current"] = self.node_map_remote_device.FindNode("AcquisitionFrameRate").Value()

            self.parameters["ExposureTime"]["min"] = self.node_map_remote_device.FindNode("ExposureTime").Minimum()
            self.parameters["ExposureTime"]["max"] = self.node_map_remote_device.FindNode("ExposureTime").Maximum()
            self.parameters["ExposureTime"]["current"] = self.node_map_remote_device.FindNode("ExposureTime").Value()

            self.parameters["Width"]["min"] = self.node_map_remote_device.FindNode("Width").Minimum()
            self.parameters["Width"]["max"] = self.node_map_remote_device.FindNode("Width").Maximum()
            self.parameters["Width"]["current"] = self.node_map_remote_device.FindNode("Width").Value()
            self.parameters["Width"]["increment"] = self.node_map_remote_device.FindNode("Width").Increment()
            self.parameters["Height"]["min"] = self.node_map_remote_device.FindNode("Height").Minimum()
            self.parameters["Height"]["max"] = self.node_map_remote_device.FindNode("Height").Maximum()
            self.parameters["Height"]["current"] = self.node_map_remote_device.FindNode("Height").Value()
            self.parameters["Height"]["increment"] = self.node_map_remote_device.FindNode("Height").Increment()

            self.parameters["OffsetX"]["min"] = self.node_map_remote_device.FindNode("OffsetX").Minimum()
            self.parameters["OffsetX"]["max"] = self.node_map_remote_device.FindNode("OffsetX").Maximum()
            self.parameters["OffsetX"]["current"] = self.node_map_remote_device.FindNode("OffsetX").Value()
            self.parameters["OffsetX"]["increment"] = self.node_map_remote_device.FindNode("OffsetX").Increment()
            self.parameters["OffsetY"]["min"] = self.node_map_remote_device.FindNode("OffsetY").Minimum()
            self.parameters["OffsetY"]["max"] = self.node_map_remote_device.FindNode("OffsetY").Maximum()
            self.parameters["OffsetY"]["current"] = self.node_map_remote_device.FindNode("OffsetY").Value()
            self.parameters["OffsetY"]["increment"] = self.node_map_remote_device.FindNode("OffsetY").Increment()

            self.node_map_remote_device.FindNode("GainSelector").SetCurrentEntry("AnalogAll")
            self.parameters["AnalogGain"]["min"] = self.node_map_remote_device.FindNode("Gain").Minimum()
            self.parameters["AnalogGain"]["max"] = self.node_map_remote_device.FindNode("Gain").Maximum()
            self.parameters["AnalogGain"]["current"] = self.node_map_remote_device.FindNode("Gain").Value()
            self.node_map_remote_device.FindNode("GainSelector").SetCurrentEntry("DigitalAll")
            self.parameters["DigitalGain"]["min"] = self.node_map_remote_device.FindNode("Gain").Minimum()
            self.parameters["DigitalGain"]["max"] = self.node_map_remote_device.FindNode("Gain").Maximum()
            self.parameters["DigitalGain"]["current"] = self.node_map_remote_device.FindNode("Gain").Value()

            if self.acquisition_running:
                current_framerate = self.parameters["AcquisitionFrameRate"]["current"]
                self.acquisition_timer.setInterval(1000 / current_framerate if current_framerate > 0 else 1000)

            self.parameter_updated.emit(self.parameters)
        except Exception as e:
            self.camera_error.emit(f"Failed to fetch initial parameters: {str(e)}")

    def start_acquisition(self):
        """Start the image acquisition process."""
        if self.device is None or self.acquisition_running or self.datastream is None:
            return False

        payload_size = self.node_map_remote_device.FindNode("PayloadSize").Value()
        buffer_count_max = self.datastream.NumBuffersAnnouncedMinRequired()
        for i in range(buffer_count_max):
            buffer = self.datastream.AllocAndAnnounceBuffer(payload_size)
            self.datastream.QueueBuffer(buffer)

        current_framerate = self.parameters["AcquisitionFrameRate"]["current"]
        self.acquisition_timer.setInterval(1000 / current_framerate if current_framerate > 0 else 1000)
        self.acquisition_timer.setSingleShot(False)
        self.acquisition_timer.timeout.connect(self.fetch_frame)

        try:
            self.datastream.StartAcquisition()
            self.node_map_remote_device.FindNode("AcquisitionStart").Execute()
            self.node_map_remote_device.FindNode("AcquisitionStart").WaitUntilDone()
        except Exception as e:
            self.camera_error.emit(f"Exception in start_acquisition: {str(e)}")
            return False

        self.acquisition_timer.start()
        self.acquisition_running = True
        self.acquisition_started.emit()
        return True

    def stop_acquisition(self):
        """Stop the image acquisition process."""
        if self.device is None or not self.acquisition_running:
            return

        try:
            self.node_map_remote_device.FindNode("AcquisitionStop").Execute()
            self.datastream.StopAcquisition(ids_peak.AcquisitionStopMode_Default)
            self.datastream.Flush(ids_peak.DataStreamFlushMode_DiscardAll)

            for buffer in self.datastream.AnnouncedBuffers():
                self.datastream.RevokeBuffer(buffer)

            self.acquisition_running = False
            self.acquisition_timer.stop()
            self.acquisition_stopped.emit()
        except Exception as e:
            self.camera_error.emit(f"Error stopping acquisition: {str(e)}")

    def close_device(self):
        """Close the camera device and release resources."""
        self.stop_acquisition()

        if self.datastream is not None:
            try:
                for buffer in self.datastream.AnnouncedBuffers():
                    self.datastream.RevokeBuffer(buffer)
            except Exception as e:
                self.camera_error.emit(f"Error closing device: {str(e)}")

        self.device = None

    def fetch_frame(self):
        """Fetch a single frame from the camera."""
        try:
            buffer = self.datastream.WaitForFinishedBuffer(5000)
            ipl_image = ids_peak_ipl_extension.BufferToImage(buffer)
            self.datastream.QueueBuffer(buffer)

            image_np_array = ipl_image.get_numpy_1D().reshape((ipl_image.Height(), ipl_image.Width()))
            image_qt = QImage(image_np_array.data, image_np_array.shape[1], image_np_array.shape[0], QImage.Format_Grayscale8)

            self.latest_frame = np.copy(image_np_array)

            self.image_acquired.emit(image_qt.copy())
            self.frame_captured.emit(self.latest_frame)
            self.frame_counter += 1
            self.acquisition_updated.emit(self.frame_counter, self.error_counter)
        except Exception as e:
            self.error_counter += 1
            self.camera_error.emit(f"Error fetching frame: {str(e)}")

    def set_gain(self, gain_type, value):
        """Set the analog or digital gain value."""
        try:
            if gain_type in ["AnalogGain", "DigitalGain"]:
                gain_selector_value = "AnalogAll" if gain_type == "AnalogGain" else "DigitalAll"
                self.node_map_remote_device.FindNode("GainSelector").SetCurrentEntry(gain_selector_value)
                self.node_map_remote_device.FindNode("Gain").SetValue(value)
                self.parameters[gain_type]["current"] = value
            else:
                raise ValueError("Invalid gain type specified.")
        except Exception as e:
            self.camera_error.emit(f"Error setting {gain_type}: {str(e)}")

    def set_parameters(self, updates):
        """Set multiple camera parameters."""
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
        """Force fetch a new frame from the camera."""
        self.fetch_frame()
        return self.latest_frame
