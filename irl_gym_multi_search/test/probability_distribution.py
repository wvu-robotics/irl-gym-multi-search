
# Generates multiple 2D Gaussian distributions as a 2D array

import numpy as np

def gaussian_filter(seed, size_x, size_y, num_goals, num_peaks_range, peak_height_range, peak_width_range_x, peak_width_range_y, peak_rot_range):
    rng = np.random.default_rng(seed=seed)  # Create a random number generator with the specified seed
    
    num_peaks = rng.integers(num_peaks_range[0], num_peaks_range[1])  # Randomly select the number of peaks within the specified range

    x, y = np.meshgrid(np.linspace(-1, 1, size_x),
                       np.linspace(-1, 1, size_y), indexing='ij')
    
    gauss = np.zeros((size_x, size_y))
    peak_centers = []

    peak_heights = rng.uniform(peak_height_range[0], peak_height_range[1], size=num_peaks)
    peak_widths_x = rng.uniform(peak_width_range_x[0], peak_width_range_x[1], size=num_peaks)
    peak_widths_y = rng.uniform(peak_width_range_y[0], peak_width_range_y[1], size=num_peaks)
    peak_rots = rng.uniform(peak_rot_range[0], peak_rot_range[1], size=num_peaks)

    # Place peaks randomly
    for i in range(num_peaks):
        peak_x = rng.integers(0, size_x)
        peak_y = rng.integers(0, size_y)
        peak_centers.append([peak_x, peak_y])

        max_peak_value = peak_heights[i]
        x_rot = (x - x[peak_x, peak_y]) * np.cos(peak_rots[i]) + (y - y[peak_x, peak_y]) * np.sin(peak_rots[i])
        y_rot = -(x - x[peak_x, peak_y]) * np.sin(peak_rots[i]) + (y - y[peak_x, peak_y]) * np.cos(peak_rots[i])
        gauss += max_peak_value * np.exp(-(((x_rot / peak_widths_x[i])**2) + ((y_rot / peak_widths_y[i])**2)) / 2.0)
    
    gauss /= np.sum(gauss) # Normalize the gauss array so that it sums to 1
    
    return gauss, num_peaks, peak_centers
