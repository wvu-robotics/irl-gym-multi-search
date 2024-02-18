import pandas as pd
import pickle

def view_results(file_path):
    # Load data from the provided file path
    with open(file_path, 'rb') as f:
        setup, results, trial_times = pickle.load(f)

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
# folder_name = 'experiment data/'
# data_file_name = 'experiment_data_20230628-140954.pickle'
# file_path = folder_name + data_file_name

# view_results(file_path)