import cv2
import os
import numpy as np
from PySide6.QtGui import QImage
from PySide6.QtCore import QObject, Signal
from skimage.feature import blob_log
from skimage import exposure
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import datetime
from polar_cam.utils import blobs_overlap

class ImageProcessor(QObject):
    image_processed = Signal(QImage)
    
    def __init__(self, display):
        super().__init__()
        self.display = display

    def preprocess_image(self, image):
        image = exposure.equalize_adapthist(image, clip_limit=0.03)
        image = cv2.medianBlur((image * 255).astype(np.uint8), 5)
        return image

    def detect_spots_log(
            self, image, min_sigma, max_sigma, num_sigma, threshold):
        blobs = blob_log(image, min_sigma, max_sigma, num_sigma, threshold)
        blobs[:, 2] = blobs[:, 2] * np.sqrt(2)
        return blobs

    def shape_check(self, blob, image):
        y, x, r = blob
        minr, minc, maxr, maxc = int(y - r), int(x - r), int(y + r), int(x + r)
        if (minr < 0 or minc < 0 or maxr > image.shape[0] or 
            maxc > image.shape[1]):
            return False

        roi = image[minr:maxr, minc:maxc]
        blob_area = np.sum(roi)
        bounding_box_area = roi.shape[0] * roi.shape[1]
        circularity = blob_area / bounding_box_area

        return circularity >= 0.8

    def detect_spots(self, image, min_sigma, max_sigma, num_sigma, threshold):
        preprocessed_image = self.preprocess_image(image)
        blobs = self.detect_spots_log(
            preprocessed_image, min_sigma, max_sigma, num_sigma, threshold)

        valid_blobs = [blob for blob in blobs 
                       if self.shape_check(blob, preprocessed_image)]

        self.check_for_overlaps(valid_blobs)

        return valid_blobs

    def check_for_overlaps(self, blobs):
        for i in range(len(blobs)):
            for j in range(i + 1, len(blobs)):
                if blobs_overlap(blobs[i], blobs[j]):
                    print(f"Blobs {i} and {j} overlap.")

    def generate_highlighted_image(self, image, blobs, output_directory):
        height, width = image.shape[:2]

        dpi = 100
        fig = plt.figure(
            frameon=False, figsize=(width / dpi, height / dpi), dpi=dpi)

        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)

        ax.imshow(image, cmap='gray', aspect='auto', vmin=0, vmax=255)
        for index, blob in enumerate(blobs):
            y, x, r = blob
            c = plt.Circle((x, y), r, color='red', linewidth=2, fill=False)
            ax.add_patch(c)
            ax.text(
                x + r, y, str(index), color='yellow', 
                fontsize=30, ha='left', va='center'
            )

        fig.canvas.draw()

        buffer = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        buffer = buffer.reshape((height, width, 3))
        qimage = QImage(
            buffer.data, width, height, 3 * width, QImage.Format_RGB888)

        self.image_processed.emit(qimage)

        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        filename = os.path.join(output_directory, f'blobs_{timestamp}.png')
        fig.savefig(filename, bbox_inches='tight', pad_inches=0, dpi=dpi)

        plt.close(fig)

    def extract_polar_inten(self, image, roi):
        x, y, width, height = roi['x'], roi['y'], roi['width'], roi['height']
        roi_image = image[y:y+height, x:x+width]

        sum_intensities = {'90': 0, '45': 0, '135': 0, '0': 0}
        count_intensities = {'90': 0, '45': 0, '135': 0, '0': 0}

        for i in range(0, height, 2):
            for j in range(0, width, 2):
                sum_intensities['90'] += roi_image[i, j]
                sum_intensities['45'] += roi_image[i, j + 1]
                sum_intensities['135'] += roi_image[i + 1, j]
                sum_intensities['0'] += roi_image[i + 1, j + 1]

                count_intensities['90'] += 1
                count_intensities['45'] += 1
                count_intensities['135'] += 1
                count_intensities['0'] += 1

        avg_intensities = {angle: sum_intensity / count_intensities[angle]
                           for angle, sum_intensity in sum_intensities.items()}

        return avg_intensities
