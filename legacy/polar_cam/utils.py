import numpy as np

def adjust_rectangle(x, y, w, h):
    if x % 2 != 0: x += 1
    if y % 2 != 0: y += 1
    if w % 2 != 0: w += 1
    if h % 2 != 0: h += 1
    return (x, y, w, h)

def blobs_overlap(blob1, blob2):
    y1, x1, r1 = blob1
    y2, x2, r2 = blob2

    distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance < (r1 + r2)

def adjust_for_increment(value, increment, max_value):
    if value % increment == 0:
        return value
    else:
        rounded_value = round(value / increment) * increment
        if rounded_value > max_value:
            return rounded_value - increment
        else:
            return rounded_value

def calculate_image_rect(
    display_width, display_height, image_width, image_height
):
    if image_width == 0 or image_height == 0:
        return None

    ratio1 = display_width / display_height
    ratio2 = image_width / image_height

    if ratio1 > ratio2:
        image_width = display_height * ratio2
        image_height = display_height
    else:
        image_width = display_width
        image_height = display_width / ratio2

    image_pos_x = -1.0 * (image_width / 2.0)
    image_pos_y = -1.0 * (image_height / 2.0)
    image_pos_x = int(image_pos_x)
    image_pos_y = int(image_pos_y)

    return image_pos_x, image_pos_y, image_width, image_height

def configure_device_component(node_map_remote_device, camera_error):
    component_selector = (
        node_map_remote_device.FindNode("ComponentSelector")
        .CurrentEntry()
        .SymbolicValue()
    )
    component_enable = (
        node_map_remote_device.FindNode("ComponentEnable")
        .Value()
    )

    if component_selector != "Raw" or not component_enable:
        try:
            node_map_remote_device.FindNode("ComponentSelector")\
                .SetCurrentEntry("Raw")
            node_map_remote_device.FindNode("ComponentEnable")\
                .SetValue(True)
        except Exception as e:
            camera_error.emit(f"Warning: Error Setting Component: {e}")


def fetch_camera_parameters(node_map_remote_device, parameters, camera_error):
    try:
        param_names = [
            "AcquisitionFrameRate", "ExposureTime", "Width", "Height",
            "OffsetX", "OffsetY", "AnalogGain", "DigitalGain"
        ]
        for param in param_names:
            parameters[param]["min"] = (
                node_map_remote_device.FindNode(param).Minimum()
            )
            parameters[param]["max"] = (
                node_map_remote_device.FindNode(param).Maximum()
            )
            parameters[param]["current"] = (
                node_map_remote_device.FindNode(param).Value()
            )
            if param in ["Width", "Height", "OffsetX", "OffsetY"]:
                parameters[param]["increment"] = (
                    node_map_remote_device.FindNode(param).Increment()
                )
        return parameters
    except Exception as e:
        camera_error.emit(f"Failed to fetch initial parameters: {str(e)}")
        return None
