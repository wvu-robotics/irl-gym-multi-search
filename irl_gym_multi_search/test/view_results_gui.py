import pandas as pd
import pickle
import tkinter as tk
import os
from tkinter import filedialog

def view_results_gui():
    # Initialize a Tkinter root widget
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    relative_path = os.path.join(os.getcwd(), 'irl_gym_multi_search', 'test', 'experiment data')

    # Open a file dialog to select a file
    file_path = filedialog.askopenfilename(initialdir=relative_path)

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

    print('\nLoaded path:\n', file_path)

    # Convert the setup dictionary to a DataFrame for better visualization
    setup_df = pd.DataFrame.from_dict(setup, orient='index', columns=['Value'])
    print("\nSetup:")
    print(setup_df)

    # Convert the results to a DataFrame
    results_df = pd.DataFrame(results, columns=['Steps', 'Found goal', 'Elapsed Time (s)'], index=range(1, len(results) + 1))
    print("Results:")
    print(results_df)

    true_count = results_df['Found goal'].sum()  # Assuming 'Found goal' column contains boolean values
    false_count = len(results_df) - true_count
    true_rate = (true_count / len(results_df)) * 100
    print(f"Trials that found the goal: {true_count}")
    print(f"Trials that reached max steps: {false_count}")
    print(f"Successful completion rate: {true_rate:.2f}%")

    # Calculate specified percentiles
    percentiles = results_df['Steps'].quantile([0.05, 0.25, 0.5, 0.75, 0.95])

    # Print out the percentile values
    print("\n\n\nPercentile Values for Steps:")
    print("                   Min:", results_df['Steps'].min())
    print(" 5th Percentile (0.05):", percentiles[0.05])
    print("25th Percentile (0.25):", percentiles[0.25])
    print("                Median:", percentiles[0.5])
    print("75th Percentile (0.75):", percentiles[0.75])
    print("95th Percentile (0.95):", percentiles[0.95])
    print("                   Max:", results_df['Steps'].max())

        # Calculate specified percentiles
    percentiles = results_df['Elapsed Time (s)'].quantile([0.05, 0.25, 0.5, 0.75, 0.95])

    # Print out the percentile values
    print("\n\nPercentile Values for Time:")
    print("                   Min:", results_df['Elapsed Time (s)'].min())
    print(" 5th Percentile (0.05):", percentiles[0.05])
    print("25th Percentile (0.25):", percentiles[0.25])
    print("                Median:", percentiles[0.5])
    print("75th Percentile (0.75):", percentiles[0.75])
    print("95th Percentile (0.95):", percentiles[0.95])
    print("                   Max:", results_df['Elapsed Time (s)'].max())

    print('\n\n')


    # Calculate and print the metrics for Elapsed Time
    avg_time = results_df['Elapsed Time (s)'].mean()
    max_time = results_df['Elapsed Time (s)'].max()
    min_time = results_df['Elapsed Time (s)'].min()
    median_time = results_df['Elapsed Time (s)'].median()
    print(f"\nAverage Time (s): {avg_time:.4f}")
    print(f"Max Time (s): {max_time:.4f}")
    print(f"Min Time (s): {min_time:.4f}")
    print(f"Median Time (s): {median_time:.4f}\n")

    total_trials = len(results_df)
    total_time_min = avg_time * total_trials / 60
    print(f"Total Time (min): {total_time_min:.4f}")

# Example usage
view_results_gui()
