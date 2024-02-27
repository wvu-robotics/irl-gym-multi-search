import pandas as pd
import pickle
import os
import sys  # Import the sys module

def view_results(file_path):
    # Check if the file path exists
    if not os.path.exists(file_path):
        print("File does not exist.")
        return

    # Load data from the provided file path
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

    # Calculate and print statistics from the results DataFrame
    true_count = results_df['Found goal'].sum()  # Assuming 'Found goal' column contains boolean values
    false_count = len(results_df) - true_count
    true_rate = (true_count / len(results_df)) * 100
    print(f"Trials that found the goal: {true_count}")
    print(f"Trials that reached max steps: {false_count}")
    print(f"Successful completion rate: {true_rate:.2f}%")

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

    total_trials = len(results_df)
    total_time_min = avg_time * total_trials / 60
    print(f"Total Time (min): {total_time_min:.4f}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script_name.py <file_path>")
    else:
        file_path = sys.argv[1]  # Get the file path from command line arguments
        view_results(file_path)
