import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

class DataAnalyzer:
    def __init__(self, nperseg=400, nfft=1600, windowsize=400, overlap=200):
        self.nperseg = nperseg
        self.nfft = nfft
        self.windowsize = windowsize
        self.overlap = overlap

    def analyze(self, intensities, timestamps, spot_id, output_directory):
        c90 = np.array(intensities['90'])
        c45 = np.array(intensities['45'])
        c135 = np.array(intensities['135'])
        c0 = np.array(intensities['0'])

        I0 = (c0 - c90) / (c0 + c90)
        I1 = (c45 - c135) / (c45 + c135)
        ANIS = I0 + 1j * I1
        ITOT = c90 + c0 + c45 + c135

        signals = {
            'I0': I0,
            'I1': I1,
            'ANIS': ANIS,
            'ITOT': ITOT
        }

        self.fft_welch(signals, timestamps, spot_id, output_directory)

    def fft_welch(
            self, signals, timestamps, spot_id, output_directory, threshold=1):
        plt.figure(figsize=(14, 8))

        for label, intensity in signals.items():
            n_seg = int((len(intensity) - self.overlap) / self.overlap)
            dom_freq = []
            current_time_segments = []

            for i in range(n_seg):
                start = i * self.overlap
                end = start + self.windowsize
                segment = intensity[start:end]

                if len(segment) < self.windowsize:
                    continue

                segment_timestamps = timestamps[start:end]
                fs = 1 / np.mean(np.diff(segment_timestamps))

                is_complex = np.iscomplexobj(segment)
                freqs, power = signal.welch(
                    segment, 
                    fs=fs, 
                    nperseg=self.nperseg, 
                    nfft=self.nfft, 
                    return_onesided=not is_complex
                )
                magnitude = np.abs(power)

                if np.max(magnitude) >= threshold:
                    dominant_frequency = freqs[np.argmax(magnitude)]
                    dom_freq.append(dominant_frequency)
                    current_time_segments.append(
                        (segment_timestamps[0] + segment_timestamps[-1]) / 2
                    )

            if dom_freq:
                plt.plot(current_time_segments, dom_freq, label=label)

        plt.title(f'Speed-Time Diagram for Spot {spot_id}')
        plt.xlabel('Time (s)')
        plt.ylabel('Dominant Frequency (Hz)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plot_filename = os.path.join(
            output_directory, f'speed_time_diagram_spot_{spot_id}.png')
        plt.savefig(plot_filename)
        plt.close()
