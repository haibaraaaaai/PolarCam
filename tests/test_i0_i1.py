import matplotlib.pyplot as plt
import os
from test_calculator import select_and_calculate

def plot_results(file_path, results):
    timestamps = [result['Timestamp'] for result in results]
    i0_values = [result['I0'] for result in results]
    i1_values = [result['I1'] for result in results]

    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, i0_values, label='I0')
    plt.plot(timestamps, i1_values, label='I1')
    plt.xlabel('Timestamps')
    plt.ylabel('Values')
    plt.ylim(-1, 1)
    plt.legend()
    plt.title('I0 and I1 over Time')
    
    base_filename = os.path.basename(file_path)
    base_name, _ = os.path.splitext(base_filename)
    save_path_time = os.path.join(os.path.dirname(file_path), 
                                  f"{base_name}_i0_i1_time.png")
    plt.savefig(save_path_time)
    plt.show()
    print(f"Plot saved as {save_path_time}")

    plt.figure(figsize=(6, 6))
    plt.scatter(i0_values, i1_values, s=10)
    plt.xlabel('I0')
    plt.ylabel('I1')
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.title('I0 vs I1')
    
    save_path_i0_i1 = os.path.join(os.path.dirname(file_path), 
                                   f"{base_name}_i0_vs_i1.png")
    plt.savefig(save_path_i0_i1)
    plt.show()
    print(f"Plot saved as {save_path_i0_i1}")

if __name__ == "__main__":
    file_path, results = select_and_calculate()
    if results:
        plot_results(file_path, results)
