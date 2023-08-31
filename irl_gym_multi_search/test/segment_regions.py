
# Takes the search distribution and separates it into regions of interest (ROI)

from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.measure import find_contours
import matplotlib.pyplot as plt
import numpy as np

def segment_regions(search_distribution, minPeakHeight=0.001, minValue = 0.0001):
    # Threshold to exclude values under the threshold
    thresholded_distribution = (search_distribution >= minValue).astype(int)

    # Find local maxima (peaks)
    peak_coords = peak_local_max(search_distribution, min_distance=2, threshold_abs=minPeakHeight, exclude_border=False)
    markers = np.zeros_like(search_distribution, dtype=int)
    for i, (x, y) in enumerate(peak_coords):
        markers[x, y] = i + 1
    
    if len(peak_coords) == 0:
        print('No peaks found. No regions generated.')
        print('Consider lowering the minimum cell values ')

    # Watershed segmentation
    labeled_regions = watershed(-search_distribution, markers, mask=thresholded_distribution)
    labeled_regions[labeled_regions == 0] = len(peak_coords) + 1  # Color unused regions

    # Create a list to hold the regions and their outlines
    regions = []
    outlines = []

    for i in range(1, len(peak_coords) + 1):
        region_coords = np.column_stack(np.where(labeled_regions == i))
        weight = np.mean(search_distribution[tuple(region_coords.T)])
        total = np.sum(search_distribution[tuple(region_coords.T)])
        centroid_x = sum(x for x, y in region_coords) / len(region_coords)
        centroid_y = sum(y for x, y in region_coords) / len(region_coords)
        region_info = {
            'index': i - 1,
            'peak_x': peak_coords[i - 1][0],
            'peak_y': peak_coords[i - 1][1],
            'num_points': len(region_coords),
            'weight': weight,
            'total': total,
            'centroid': (centroid_x, centroid_y),
            'points': region_coords.tolist()
        }
        regions.append(region_info)
        
        region_mask = (labeled_regions.T == i)
        contours = find_contours(region_mask, 0.5)
        outlines.append(contours)

    # Plotting setup
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot search distribution and region outlines
    im = ax.imshow(search_distribution.T, cmap='viridis', origin='lower')
    ax.set_title('Search Distribution with region Outlines')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_ylim(top=0, bottom=search_distribution.shape[1])
    fig.colorbar(im, ax=ax)

    # Plot the contours of each region
    for contour in outlines:
        for segment in contour:
            ax.plot(segment[:, 1], segment[:, 0], linewidth=2, color='red')

    # Adjust the layout of the subplots
    plt.tight_layout()

    # plt.show()

    return regions, outlines
