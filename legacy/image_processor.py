import cv2
import numpy as np
from skimage.feature import blob_log
from skimage import exposure
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import datetime


class ImageProcessor:
    def __init__(self, min_sigma=10, max_sigma=30, num_sigma=10, threshold=0.3):
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.num_sigma = num_sigma
        self.threshold = threshold
        self.spots = []

    def preprocess_image(self, image):
        """Preprocess the image using adaptive histogram equalization and median filtering."""
        image = exposure.equalize_adapthist(image, clip_limit=0.03)
        
        image = cv2.medianBlur((image * 255).astype(np.uint8), 5)
        
        return image

    def detect_spots_log(self, image):
        """Detect spots using the Laplacian of Gaussian (LoG) method."""
        blobs = blob_log(image, min_sigma=self.min_sigma, max_sigma=self.max_sigma, num_sigma=self.num_sigma, threshold=self.threshold)
        blobs[:, 2] = blobs[:, 2] * np.sqrt(2)  # Convert standard deviation to radius
        return blobs

    def shape_check(self, blob, image):
        """Check the circularity of the detected blob."""
        y, x, r = blob
        minr, minc, maxr, maxc = int(y - r), int(x - r), int(y + r), int(x + r)
        if minr < 0 or minc < 0 or maxr > image.shape[0] or maxc > image.shape[1]:
            return False

        roi = image[minr:maxr, minc:maxc]
        blob_area = np.sum(roi)
        bounding_box_area = roi.shape[0] * roi.shape[1]
        circularity = blob_area / bounding_box_area

        if circularity < 0.8:
            return False

        return True

    def detect_spots(self, image):
        """Main method to detect spots and convert blobs to rectangles."""
        preprocessed_image = self.preprocess_image(image)
        blobs = self.detect_spots_log(preprocessed_image)

        unique_id = 0
        self.spots.clear()

        for blob in blobs:
            if self.shape_check(blob, preprocessed_image):
                y, x, r = blob
                w = h = int(r * 2)
                x = int(x - r)
                y = int(y - r)

                rect_x, rect_y, rect_w, rect_h = self.adjust_rectangle(x, y, w, h)
                self.spots.append({'id': unique_id, 'x': rect_x, 'y': rect_y, 'width': rect_w, 'height': rect_h})
                unique_id += 1

        self.check_for_overlaps(blobs)
        self.save_blobs_image(preprocessed_image, blobs)
        
        print(self.spots)
        
        return self.spots

    def adjust_rectangle(self, x, y, w, h):
        if x % 2 != 0: x += 1
        if y % 2 != 0: y += 1
        if w % 2 != 0: w += 1
        if h % 2 != 0: h += 1
        return (x, y, w, h)

    def get_spots(self):
        return self.spots

    def check_for_overlaps(self, blobs):
        for i in range(len(blobs)):
            for j in range(i + 1, len(blobs)):
                if self.blobs_overlap(blobs[i], blobs[j]):
                    print(f"Blobs {i} and {j} overlap.")

    def blobs_overlap(self, blob1, blob2):
        y1, x1, r1 = blob1
        y2, x2, r2 = blob2

        distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if distance < (r1 + r2):
            return True
        return False

    def save_blobs_image(self, image, blobs):
        """Save the detected blobs image using matplotlib and display with OpenCV."""
        plt.figure(figsize=(10, 10))
        plt.imshow(image, cmap='gray')
        plt.title("Detected Blobs with Laplacian of Gaussian")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.grid(True)

        for blob in blobs:
            if self.shape_check(blob, image):
                y, x, r = blob
                c = plt.Circle((x, y), r, color='red', linewidth=2, fill=False)
                plt.gca().add_patch(c)

        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f'blobs_{timestamp}.png'
        plt.savefig(filename)
        plt.close()

        detected_image = cv2.imread(filename)
        cv2.imshow("Detected Blobs", detected_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def extract_polarization_intensities(self, image, roi):
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
