import matplotlib.pyplot as plt
import os
from scipy.signal import medfilt
from test_calculator import select_and_calculate

def plot_results(file_path, results, apply_median_filter=False, kernel_size=3):
    timestamps = [result['Timestamp'] for result in results]
    c0_values = [result['c0'] for result in results]
    c45_values = [result['c45'] for result in results]
    c90_values = [result['c90'] for result in results]
    c135_values = [result['c135'] for result in results]

    if apply_median_filter:
        c0_values = medfilt(c0_values, kernel_size=kernel_size)
        c45_values = medfilt(c45_values, kernel_size=kernel_size)
        c90_values = medfilt(c90_values, kernel_size=kernel_size)
        c135_values = medfilt(c135_values, kernel_size=kernel_size)

    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, c0_values, label='c0')
    plt.plot(timestamps, c45_values, label='c45')
    plt.plot(timestamps, c90_values, label='c90')
    plt.plot(timestamps, c135_values, label='c135')
    plt.xlabel('Timestamps')
    plt.ylabel('Values')
    plt.legend()
    plt.title('Raw Data over Time')

    base_filename = os.path.basename(file_path)
    base_name, _ = os.path.splitext(base_filename)
    save_path = os.path.join(os.path.dirname(file_path), 
                             f"{base_name}_raw.png")
    plt.savefig(save_path)
    plt.show()
    print(f"Plot saved as {save_path}")

if __name__ == "__main__":
    file_path, results = select_and_calculate()
    if results:
        plot_results(file_path, results, True, 35)
