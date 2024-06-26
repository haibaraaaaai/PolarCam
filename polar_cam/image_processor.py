import cv2
import os
import numpy as np
from skimage.feature import blob_log
from skimage import exposure
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import datetime
from polar_cam.utils import blobs_overlap

class ImageProcessor:
    def __init__(self):
        pass

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

    def detect_spots(self, image, output_directory, min_sigma, 
                     max_sigma, num_sigma, threshold):
        preprocessed_image = self.preprocess_image(image)
        blobs = self.detect_spots_log(
            preprocessed_image, min_sigma, max_sigma, num_sigma, threshold)

        valid_blobs = [blob for blob in blobs 
                       if self.shape_check(blob, preprocessed_image)]

        self.check_for_overlaps(valid_blobs)
        highlighted_image = self.save_blobs_image(
            preprocessed_image, valid_blobs, output_directory)

        return highlighted_image, valid_blobs

    def check_for_overlaps(self, blobs):
        for i in range(len(blobs)):
            for j in range(i + 1, len(blobs)):
                if blobs_overlap(blobs[i], blobs[j]):
                    print(f"Blobs {i} and {j} overlap.")

    def save_blobs_image(self, image, blobs, output_directory):
        plt.figure(figsize=(10, 10))
        plt.imshow(image, cmap='gray')
        plt.title("Detected Blobs with Laplacian of Gaussian")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.grid(True)

        for blob in blobs:
            y, x, r = blob
            c = plt.Circle((x, y), r, color='red', linewidth=2, fill=False)
            plt.gca().add_patch(c)

        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        filename = os.path.join(output_directory, f'blobs_{timestamp}.png')
        plt.savefig(filename)
        plt.close()

        detected_image = cv2.imread(filename)
        return detected_image

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
