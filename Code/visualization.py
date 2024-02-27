import numpy as np
import matplotlib.pyplot as plt

def plot_signal(signal: np.ndarray, start_time: float = None, end_time: float = None) -> None:
    """
    Plots the signal over time, with an option to focus on a specific time range.

    Parameters:
    - signal (np.ndarray): Signal values.
    - start_time (float, optional): Start time for plotting a specific range.
    - end_time (float, optional): End time for plotting a specific range.
    """

    plt.figure(figsize=(12, 8), dpi=600)
    plt.style.use('default')
    plt.plot(signal, label="Signal", color="black", linewidth=2)

    if start_time is not None and end_time is not None:
        plt.xlim(start_time, end_time)
    plt.legend(loc="upper right", fontsize=20, frameon=True, edgecolor='black')
    plt.xlabel("Time (seconds)", fontsize=16, fontweight='bold', color='black')
    plt.ylabel("Signal value", fontsize=16, fontweight='bold', color='black')
    plt.xticks(fontsize=12, fontweight='bold', color='black')
    plt.yticks(fontsize=12, fontweight='bold', color='black')
    plt.tight_layout()
    plt.grid(True, color='gray', linewidth=1)

    for axis in ['top', 'bottom', 'left', 'right']:
        plt.gca().spines[axis].set_linewidth(2)

    plt.show()

    