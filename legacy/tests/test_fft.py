import matplotlib.pyplot as plt
import os
import numpy as np
from scipy import signal
from test_calculator import select_and_calculate

def plot_results(file_path, results):
    timestamps = [result['Timestamp'] for result in results]
    I0 = [result['I0'] for result in results]
    I1 = [result['I1'] for result in results]
    ANIS = [result['ANIS'] for result in results]
    ITOT = [result['ITOT'] for result in results]

    signals = {
        'I0': I0,
        'I1': I1,
        'ANIS': ANIS,
        'ITOT': ITOT
    }

    signals_fft, seg_time = fft_welch(signals, timestamps)

    plt.figure(figsize=(10, 6))
    for s in signals_fft:
        plt.plot(seg_time, signals_fft[s], label=s)
    plt.xlabel('Time (s)')
    plt.ylabel('Speed (Hz)')
    plt.legend()
    plt.title('Speed time traces')

    base_filename = os.path.basename(file_path)
    base_name, _ = os.path.splitext(base_filename)
    save_path = os.path.join(os.path.dirname(file_path), 
                             f"{base_name}_fft.png")
    plt.savefig(save_path)
    plt.show()
    print(f"Plot saved as {save_path}")

def fft_welch(signals, timestamps, nperseg=400, 
              nfft=1600, windowsize=400, overlap=200):
    signals_fft = {}
    for label, intensity in signals.items():
        n_seg = int((len(intensity) - overlap) / overlap)
        dom_freq = []
        time_segments = []

        for i in range(n_seg):
            start = i * overlap
            end = start + windowsize
            segment = intensity[start:end]

            if len(segment) < windowsize:
                continue

            segment_timestamps = timestamps[start:end]
            fs = 1 / np.mean(np.diff(segment_timestamps))

            is_complex = np.iscomplexobj(segment)
            freqs, power = signal.welch(
                segment, 
                fs=fs, 
                nperseg=nperseg, 
                nfft=nfft, 
                return_onesided=not is_complex
            )
            magnitude = np.abs(power)

            dominant_frequency = freqs[np.argmax(magnitude)]
            dom_freq.append(dominant_frequency)
            time_segments.append(
                (segment_timestamps[0] + segment_timestamps[-1]) / 2
            )

        signals_fft[label] = dom_freq

    return signals_fft, time_segments

if __name__ == "__main__":
    file_path, results = select_and_calculate()
    if results:
        plot_results(file_path, results)
