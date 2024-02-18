import pandas as pd
import pickle
import tkinter as tk
from tkinter import filedialog

def view_results_gui():
    # Initialize a Tkinter root widget
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    # Open a file dialog to select a file
    file_path = filedialog.askopenfilename()

    # Check if a file was selected
    if not file_path:
        print("No file selected.")
        return

    # Load data from the selected file path
    with open(file_path, 'rb') as f:
        loaded_data = pickle.load(f)
        
        # Check the number of elements in loaded_data
        if len(loaded_data) == 3:
            setup, results, trial_times = loaded_data
        elif len(loaded_data) == 2:
            setup, results = loaded_data
            trial_times = None  # Assign a default value or handle accordingly
        else:
            print("Unexpected data format in the file.")
            return

    # Convert the setup dictionary to a DataFrame for better visualization
    setup_df = pd.DataFrame.from_dict(setup, orient='index', columns=['Value'])
    print("Setup:")
    print(setup_df)

    # Convert the results to a DataFrame
    results_df = pd.DataFrame(results, columns=['Steps', 'Found goal', 'Elapsed Time (s)'], index=range(1, len(results) + 1))
    print("\nResults:")
    print(results_df)

    # Calculate and print the metrics for Steps
    avg_steps = results_df['Steps'].mean()
    max_steps = results_df['Steps'].max()
    min_steps = results_df['Steps'].min()
    median_steps = results_df['Steps'].median()
    print(f"\nAverage Steps: {avg_steps:.4f}")
    print(f"Max Steps: {max_steps}")
    print(f"Min Steps: {min_steps}")
    print(f"Median Steps: {median_steps}")

    # Calculate and print the metrics for Elapsed Time
    avg_time = results_df['Elapsed Time (s)'].mean()
    max_time = results_df['Elapsed Time (s)'].max()
    min_time = results_df['Elapsed Time (s)'].min()
    median_time = results_df['Elapsed Time (s)'].median()
    print(f"\nAverage Time (s): {avg_time:.4f}")
    print(f"Max Time (s): {max_time:.4f}")
    print(f"Min Time (s): {min_time:.4f}")
    print(f"Median Time (s): {median_time:.4f}\n")

# Example usage
view_results_gui()
