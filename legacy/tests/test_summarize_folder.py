import os
import tkinter as tk
from tkinter import filedialog
from test_i0_i1 import plot_results as plot_i0_i1_results
from test_fourkas import plot_results as plot_fourkas_results

def select_folder():
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(title="Select a folder")
    return folder_path

def process_files_in_folder(folder_path):
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.startswith('raw_data_spot_') and file.endswith('.txt'):
                file_path = os.path.join(root, file)
                print(f"Processing file: {file_path}")
                try:
                    print("Running test_i0_i1...")
                    plot_i0_i1_results(file_path, results_file(file_path))
                except Exception as e:
                    print(f"Error in test_i0_i1 for file {file_path}: {e}")
                try:
                    print("Running test_fourkas...")
                    plot_fourkas_results(file_path, results_file(file_path))
                except Exception as e:
                    print(f"Error in test_fourkas for file {file_path}: {e}")

def results_file(file_path):
    from test_calculator import select_and_calculate
    _, results = select_and_calculate(file_path=file_path)
    return results

if __name__ == "__main__":
    folder_path = select_folder()
    if folder_path:
        print(f"Selected folder: {folder_path}")
        process_files_in_folder(folder_path)
    else:
        print("No folder selected")
