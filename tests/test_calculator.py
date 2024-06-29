import tkinter as tk
from tkinter import filedialog
import numpy as np

def parse_file(file_path):
    timestamps = []
    c90_values = []
    c45_values = []
    c135_values = []
    c0_values = []

    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if line.strip() and not line.startswith("Spot ID") and \
               not line.startswith("Timestamps"):
                parts = line.split(',')
                timestamps.append(float(parts[0].strip()))
                c90_values.append(float(parts[1].strip()))
                c45_values.append(float(parts[2].strip()))
                c135_values.append(float(parts[3].strip()))
                c0_values.append(float(parts[4].strip()))
    
    return timestamps, c90_values, c45_values, c135_values, c0_values

def parse_npz_file(file_path):
    data = np.load(file_path)
    timestamps = data['timestamps']
    c90_values = data['intensities']['c90']
    c45_values = data['intensities']['c45']
    c135_values = data['intensities']['c135']
    c0_values = data['intensities']['c0']
    
    return timestamps, c90_values, c45_values, c135_values, c0_values

class FourkasCalculator:
    def __init__(self, NA, nw, tweaktheta):
        self.NA = NA
        self.nw = nw
        self.tweaktheta = tweaktheta

    def calculate(self, c90, c45, c135, c0):
        alpha = np.arcsin(self.NA / self.nw)
        A = 1/6 - 1/4 * np.cos(alpha) + 1/12 * np.cos(alpha)**3
        B = 1/8 * np.cos(alpha) - 1/8 * np.cos(alpha)**3
        C = 7/48 - np.cos(alpha)/16 - np.cos(alpha)**2/16 - np.cos(alpha)**3/48
        phi = 0.5 * np.arctan2((c45 / 2 - c135 / 2), (c0 / 2 - c90 / 2))

        cs = np.cos(2 * phi)
        if cs == 0:
            Itots2thet = c0 / A
        else:
            Itots2thet = 1 / (2 * A) * ((1 - B / (C * cs)) * c0 + 
                                        (1 + B / (C * cs)) * c90)

        sqrt_arg = (c0 - c90) / (2 * self.tweaktheta * Itots2thet * C * cs)

        nan_marker = np.nan

        if sqrt_arg == 0:
            theta1 = 0
        elif sqrt_arg < 0 or sqrt_arg > 1:
            theta1 = nan_marker
        else:
            theta1 = np.arcsin(np.sqrt(sqrt_arg))

        I0 = (c0 - c90) / (c0 + c90)
        I1 = (c45 - c135) / (c45 + c135)
        ANIS = I0 + 1j * I1
        ITOT = c90 + c0 + c45 + c135

        return phi, theta1, I0, I1, ANIS, ITOT

def select_and_calculate(file_path=None):
    if file_path is None:
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(
            title="Select a file",
            filetypes=(("Text files", "*.txt"),
                        ("NPZ files", "*.npz"), ("All files", "*.*"))
        )
    if file_path:
        if file_path.endswith('.txt'):
            timestamps, c90, c45, c135, c0 = parse_file(file_path)
        elif file_path.endswith('.npz'):
            timestamps, c90, c45, c135, c0 = parse_npz_file(file_path)
        else:
            print("Unsupported file type")
            return None, None

        calculator = FourkasCalculator(NA=1.0, nw=1.33, tweaktheta=0.5)
        
        results = []
        for i in range(len(timestamps)):
            phi, theta1, I0, I1, ANIS, ITOT = calculator.calculate(
                c90[i], c45[i], c135[i], c0[i]
            )
            results.append({
                'Timestamp': timestamps[i], 'Phi': phi, 'Theta1': theta1,
                'I0': I0, 'I1': I1, 'ANIS': ANIS, 'ITOT': ITOT,
                'c90': c90[i], 'c45': c45[i], 'c135': c135[i], 'c0': c0[i]
            })
        return file_path, results
    else:
        print("No file selected")
        return None, None

if __name__ == "__main__":
    file_path, results = select_and_calculate()
    if results:
        for result in results:
            print(result)
