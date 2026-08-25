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

    fft_results = fft_spectrum(signals, timestamps)

    plt.figure(figsize=(10, 6))
    for label, (freqs, magnitude) in fft_results.items():
        plt.plot(freqs, magnitude, label=label)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.title('Magnitude Spectrum of the First Segment')

    base_filename = os.path.basename(file_path)
    base_name, _ = os.path.splitext(base_filename)
    save_path = os.path.join(os.path.dirname(file_path), 
                             f"{base_name}_fft_spectrum.png")
    plt.savefig(save_path)
    plt.show()
    print(f"Plot saved as {save_path}")

def fft_spectrum(signals, timestamps, nperseg=400, nfft=1600, windowsize=400):
    fft_results = {}
    for label, intensity in signals.items():

        segment = intensity[:windowsize]

        if len(segment) < windowsize:
            print(f"Segment too short for {label}, skipping.")
            continue

        segment_timestamps = timestamps[:windowsize]
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

        fft_results[label] = (freqs, magnitude)

    return fft_results

if __name__ == "__main__":
    file_path, results = select_and_calculate()
    if results:
        plot_results(file_path, results)
