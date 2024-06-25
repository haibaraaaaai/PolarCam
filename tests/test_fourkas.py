import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.interpolate import interp1d
from test_calculator import select_and_calculate

def plot_results(file_path, results):
    timestamps = [result['Timestamp'] for result in results]
    phi_values = [result['Phi'] for result in results]
    theta1_values = [result['Theta1'] for result in results]

    valid_indices = ~np.isnan(theta1_values)
    if np.any(valid_indices):
        f_interp = interp1d(
            np.array(timestamps)[valid_indices], 
            np.array(theta1_values)[valid_indices], 
            kind='linear', 
            fill_value="extrapolate"
        )
        theta1_values = f_interp(timestamps)

    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, phi_values, label='Phi')
    plt.plot(timestamps, theta1_values, label='Theta1')
    plt.xlabel('Timestamps')
    plt.ylabel('Values')
    plt.legend()
    plt.title('Phi and Theta1 over Time')

    base_filename = os.path.basename(file_path)
    base_name, _ = os.path.splitext(base_filename)
    save_path = os.path.join(os.path.dirname(file_path), 
                             f"{base_name}_fourkas.png")
    plt.savefig(save_path)
    plt.show()
    print(f"Plot saved as {save_path}")

if __name__ == "__main__":
    file_path, results = select_and_calculate()
    if results:
        plot_results(file_path, results)
