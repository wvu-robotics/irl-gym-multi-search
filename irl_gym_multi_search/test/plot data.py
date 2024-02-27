import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# ################################################################## 50x50
# ##################################### steps

# # MCTS Regions Lite, 50x50
# mcts_regions_lite_50_50_min = 1
# mcts_regions_lite_50_50_p5 = 30.95
# mcts_regions_lite_50_50_p25 = 102.75
# mcts_regions_lite_50_50_median = 239
# mcts_regions_lite_50_50_p75 = 452.5
# mcts_regions_lite_50_50_p95 = 787.1
# mcts_regions_lite_50_50_max = 2500

# # MCTS Regions, 50x50 REDO
# mcts_regions_50_50_min = 30
# mcts_regions_50_50_p5 = 35
# mcts_regions_50_50_p25 = 140
# mcts_regions_50_50_median = 444.5
# mcts_regions_50_50_p75 = 806
# mcts_regions_50_50_p95 = 1205
# mcts_regions_50_50_max = 1403

# # MCTS, 50x50
# mcts_50_50_min = 1
# mcts_50_50_p5 = 26
# mcts_50_50_p25 = 155
# mcts_50_50_median = 722
# mcts_50_50_p75 = 2500
# mcts_50_50_p95 = 2500
# mcts_50_50_max = 2500

# # Receding Horizon, 50x50
# receding_horizon_50_50_min = 4
# receding_horizon_50_50_p5 = 41
# receding_horizon_50_50_p25 = 316.75
# receding_horizon_50_50_median = 2500
# receding_horizon_50_50_p75 = 2500
# receding_horizon_50_50_p95 = 2500
# receding_horizon_50_50_max = 2500

# # Breadth First Search, 50x50  NEEDS REDONE
# breadth_first_search_50_50_min = 1
# breadth_first_search_50_50_p5 = 24.95
# breadth_first_search_50_50_p25 = 85
# breadth_first_search_50_50_median = 298.5
# breadth_first_search_50_50_p75 = 640.5
# breadth_first_search_50_50_p95 = 2000
# breadth_first_search_50_50_max = 2500

# ##################################### time

# # MCTS Regions Lite, 50x50
# mcts_regions_lite_50_50_min_time = 1
# mcts_regions_lite_50_50_p5_time = 30.95
# mcts_regions_lite_50_50_p25_time = 102.75
# mcts_regions_lite_50_50_median_time = 239
# mcts_regions_lite_50_50_p75_time = 452.5
# mcts_regions_lite_50_50_p95_time = 787.1
# mcts_regions_lite_50_50_max_time = 2500

# # MCTS Regions, 50x50 REDO
# mcts_regions_50_50_min_time = 30
# mcts_regions_50_50_p5_time = 35
# mcts_regions_50_50_p25_time = 140
# mcts_regions_50_50_median_time = 444.5
# mcts_regions_50_50_p75_time = 806
# mcts_regions_50_50_p95_time = 1205
# mcts_regions_50_50_max_time = 1403

# # MCTS, 50x50
# mcts_50_50_min_time = 1
# mcts_50_50_p5_time = 26
# mcts_50_50_p25_time = 155
# mcts_50_50_median_time = 722
# mcts_50_50_p75_time = 2500
# mcts_50_50_p95_time = 2500
# mcts_50_50_max_time = 2500

# # Receding Horizon, 50x50
# receding_horizon_50_50_min_time = 4
# receding_horizon_50_50_p5_time = 41
# receding_horizon_50_50_p25_time = 316.75
# receding_horizon_50_50_median_time = 2500
# receding_horizon_50_50_p75_time = 2500
# receding_horizon_50_50_p95_time = 2500
# receding_horizon_50_50_max_time = 2500

# # Breadth First Search, 50x50  NEEDS REDONE
# breadth_first_search_50_50_min_time = 1
# breadth_first_search_50_50_p5_time = 24.95
# breadth_first_search_50_50_p25_time = 85
# breadth_first_search_50_50_median_time = 298.5
# breadth_first_search_50_50_p75_time = 640.5
# breadth_first_search_50_50_p95_time = 2000
# breadth_first_search_50_50_max_time = 2500

# ################################################################## 100x100
# ##################################### steps

# # MCTS Regions Lite, 100x100
# mcts_regions_lite_100_100_min = 4
# mcts_regions_lite_100_100_p5 = 88.95
# mcts_regions_lite_100_100_p25 = 412.25
# mcts_regions_lite_100_100_median = 1018
# mcts_regions_lite_100_100_p75 = 1918
# mcts_regions_lite_100_100_p95 = 3736.45
# mcts_regions_lite_100_100_max = 10000

# # MCTS Regions, 100x100
# mcts_regions_100_100_min = 0
# mcts_regions_100_100_p5 = 1
# mcts_regions_100_100_p25 = 2
# mcts_regions_100_100_median = 5
# mcts_regions_100_100_p75 = 8
# mcts_regions_100_100_p95 = 9
# mcts_regions_100_100_max = 10

# # MCTS, 100x100
# mcts_100_100_min = 0
# mcts_100_100_p5 = 1
# mcts_100_100_p25 = 2
# mcts_100_100_median = 5
# mcts_100_100_p75 = 8
# mcts_100_100_p95 = 9
# mcts_100_100_max = 10

# # Receding Horizon, 100x100
# receding_horizon_100_100_min = 0
# receding_horizon_100_100_p5 = 1
# receding_horizon_100_100_p25 = 2
# receding_horizon_100_100_median = 5
# receding_horizon_100_100_p75 = 8
# receding_horizon_100_100_p95 = 9
# receding_horizon_100_100_max = 10

# # Breadth First Search, 100x100
# breadth_first_search_100_100_min = 9
# breadth_first_search_100_100_p5 = 72.85
# breadth_first_search_100_100_p25 = 521.25
# breadth_first_search_100_100_median = 1973
# breadth_first_search_100_100_p75 = 3965.25
# breadth_first_search_100_100_p95 = 10000
# breadth_first_search_100_100_max = 10000

# ##################################### time

# # MCTS Regions Lite, 100x100
# mcts_regions_lite_100_100_min_time = 4
# mcts_regions_lite_100_100_p5_time = 88.95
# mcts_regions_lite_100_100_p25_time = 412.25
# mcts_regions_lite_100_100_median_time = 1018
# mcts_regions_lite_100_100_p75_time = 1918
# mcts_regions_lite_100_100_p95_time = 3736.45
# mcts_regions_lite_100_100_max_time = 10000

# # MCTS Regions, 100x100
# mcts_regions_100_100_min_time = 0
# mcts_regions_100_100_p5_time = 1
# mcts_regions_100_100_p25_time = 2
# mcts_regions_100_100_median_time = 5
# mcts_regions_100_100_p75_time = 8
# mcts_regions_100_100_p95_time = 9
# mcts_regions_100_100_max_time = 10

# # MCTS, 100x100
# mcts_100_100_min_time = 0
# mcts_100_100_p5_time = 1
# mcts_100_100_p25_time = 2
# mcts_100_100_median_time = 5
# mcts_100_100_p75_time = 8
# mcts_100_100_p95_time = 9
# mcts_100_100_max_time = 10

# # Receding Horizon, 100x100
# receding_horizon_100_100_min_time = 0
# receding_horizon_100_100_p5_time = 1
# receding_horizon_100_100_p25_time = 2
# receding_horizon_100_100_median_time = 5
# receding_horizon_100_100_p75_time = 8
# receding_horizon_100_100_p95_time = 9
# receding_horizon_100_100_max_time = 10

# # Breadth First Search, 100x100
# breadth_first_search_100_100_min_time = 9
# breadth_first_search_100_100_p5_time = 72.85
# breadth_first_search_100_100_p25_time = 521.25
# breadth_first_search_100_100_median_time = 1973
# breadth_first_search_100_100_p75_time = 3965.25
# breadth_first_search_100_100_p95_time = 10000
# breadth_first_search_100_100_max_time = 10000

################################################################## 200x200
##################################### steps

# MCTS Regions Lite, 200x200
mcts_regions_lite_200_200_min = 74
mcts_regions_lite_200_200_p5 = 349.05
mcts_regions_lite_200_200_p25 = 1746.5
mcts_regions_lite_200_200_median = 5521.5
mcts_regions_lite_200_200_p75 = 10930.75
mcts_regions_lite_200_200_p95 = 24853.8
mcts_regions_lite_200_200_max = 40000

# MCTS Regions, 200x200
mcts_regions_200_200_min = 434
mcts_regions_200_200_p5 = 560.8
mcts_regions_200_200_p25 = 10762
mcts_regions_200_200_median = 26185
mcts_regions_200_200_p75 = 40000
mcts_regions_200_200_p95 = 40000
mcts_regions_200_200_max = 40000

# MCTS, 200x200
mcts_200_200_min = 20
mcts_200_200_p5 = 421.95
mcts_200_200_p25 = 39655
mcts_200_200_median = 39611
mcts_200_200_p75 = 40000
mcts_200_200_p95 = 40000
mcts_200_200_max = 40000

# Receding Horizon, 200x200
receding_horizon_200_200_min = 165
receding_horizon_200_200_p5 = 444.4
receding_horizon_200_200_p25 = 2387
receding_horizon_200_200_median = 40000
receding_horizon_200_200_p75 = 40000
receding_horizon_200_200_p95 = 40000
receding_horizon_200_200_max = 40000

# Breadth First Search, 200x200
breadth_first_search_200_200_min = 72
breadth_first_search_200_200_p5 = 266.75
breadth_first_search_200_200_p25 = 2357.75
breadth_first_search_200_200_median = 8830.5
breadth_first_search_200_200_p75 = 40000
breadth_first_search_200_200_p95 = 40000
breadth_first_search_200_200_max = 40000

##################################### time

# MCTS Regions Lite, 200x200
mcts_regions_lite_200_200_min_time = 1.8499
mcts_regions_lite_200_200_p5_time = 12.557
mcts_regions_lite_200_200_p25_time = 81.513
mcts_regions_lite_200_200_median_time = 194.069
mcts_regions_lite_200_200_p75_time = 364.99
mcts_regions_lite_200_200_p95_time = 618.4405
mcts_regions_lite_200_200_max_time = 9414.657

# MCTS Regions, 200x200
mcts_regions_200_200_min_time = 6.43565
mcts_regions_200_200_p5_time = 31.411
mcts_regions_200_200_p25_time = 199.131
mcts_regions_200_200_median_time = 540.7519
mcts_regions_200_200_p75_time = 681.26496
mcts_regions_200_200_p95_time = 890.341
mcts_regions_200_200_max_time = 1670.9493

# MCTS, 200x200
mcts_200_200_min_time = 2.585
mcts_200_200_p5_time = 7.7558
mcts_200_200_p25_time = 753.598
mcts_200_200_median_time = 1000.133
mcts_200_200_p75_time = 1058.7025
mcts_200_200_p95_time = 1137.252
mcts_200_200_max_time = 1169.1148

# Receding Horizon, 200x200
receding_horizon_200_200_min_time = 9.74751
receding_horizon_200_200_p5_time = 24.5319
receding_horizon_200_200_p25_time = 140.609
receding_horizon_200_200_median_time = 1880.880
receding_horizon_200_200_p75_time = 2126.0149
receding_horizon_200_200_p95_time = 2339.5541
receding_horizon_200_200_max_time = 2349.3642

# Breadth First Search, 200x200
breadth_first_search_200_200_min_time = 1.6818
breadth_first_search_200_200_p5_time = 3.8077
breadth_first_search_200_200_p25_time = 32.5304
breadth_first_search_200_200_median_time = 116.784
breadth_first_search_200_200_p75_time = 432.740
breadth_first_search_200_200_p95_time = 928.819
breadth_first_search_200_200_max_time = 1004.99

################################################################## data

data_steps = {
    'Algorithm': [
        'MCTS Regions Lite', 'MCTS Regions', 'MCTS', 'Receding Horizon', 'Breadth First Search'
    ],
    '5th Percentile Steps': [
        mcts_regions_lite_200_200_p5, mcts_regions_200_200_p5, mcts_200_200_p5,
        receding_horizon_200_200_p5, breadth_first_search_200_200_p5
    ],
    '25th Percentile Steps': [
        mcts_regions_lite_200_200_p25, mcts_regions_200_200_p25, mcts_200_200_p25,
        receding_horizon_200_200_p25, breadth_first_search_200_200_p25
    ],
    'Median Steps': [
        mcts_regions_lite_200_200_median, mcts_regions_200_200_median, mcts_200_200_median,
        receding_horizon_200_200_median, breadth_first_search_200_200_median
    ],
    '75th Percentile Steps': [
        mcts_regions_lite_200_200_p75, mcts_regions_200_200_p75, mcts_200_200_p75,
        receding_horizon_200_200_p75, breadth_first_search_200_200_p75
    ],
    '95th Percentile Steps': [
        mcts_regions_lite_200_200_p95, mcts_regions_200_200_p95, mcts_200_200_p95,
        receding_horizon_200_200_p95, breadth_first_search_200_200_p95
    ]
}

data_time = {
    'Algorithm': [
        'MCTS Regions Lite', 'MCTS Regions', 'MCTS', 'Receding Horizon', 'Breadth First Search'
    ],
    '5th Percentile Time': [
        mcts_regions_lite_200_200_p5_time, mcts_regions_200_200_p5_time, mcts_200_200_p5_time,
        receding_horizon_200_200_p5_time, breadth_first_search_200_200_p5_time
    ],
    '25th Percentile Time': [
        mcts_regions_lite_200_200_p25_time, mcts_regions_200_200_p25_time, mcts_200_200_p25_time,
        receding_horizon_200_200_p25_time, breadth_first_search_200_200_p25_time
    ],
    'Median Time': [
        mcts_regions_lite_200_200_median_time, mcts_regions_200_200_median_time, mcts_200_200_median_time,
        receding_horizon_200_200_median_time, breadth_first_search_200_200_median_time
    ],
    '75th Percentile Time': [
        mcts_regions_lite_200_200_p75_time, mcts_regions_200_200_p75_time, mcts_200_200_p75_time,
        receding_horizon_200_200_p75_time, breadth_first_search_200_200_p75_time
    ],
    '95th Percentile Time': [
        mcts_regions_lite_200_200_p95_time, mcts_regions_200_200_p95_time, mcts_200_200_p95_time,
        receding_horizon_200_200_p95_time, breadth_first_search_200_200_p95_time
    ]
}


df_steps = pd.DataFrame(data_steps)
df_time = pd.DataFrame(data_time)

# Create a figure with subplots for steps and time
fig, axes = plt.subplots(1, 2, figsize=(10, 6))  # Adjusted for two subplots

# Use the values from the subplot configuration tool sliders
fig.subplots_adjust(left=0.064,
                    bottom=0.036,
                    right=0.8,
                    top=0.945,
                    wspace=0.225,
                    hspace=0.2)

# Titles for the plots
titles = ['Steps Per Trial', 'Time [s] Per Trial']

# Define a custom color palette for the box plots
custom_palette = ['deepskyblue', 'seagreen', 'firebrick', 'darkviolet', 'orange']

# DataFrames for 200x200 size are already filtered for this size, so we directly use them
# Plotting steps
box_data_steps = df_steps.iloc[:, 1:].values.tolist()  # Excluding the 'Algorithm' column
bp_steps = axes[0].boxplot(box_data_steps, positions=np.arange(len(box_data_steps)), patch_artist=True,
                           whis=9999999, medianprops=dict(color="black"))

# Plotting time
box_data_time = df_time.iloc[:, 1:].values.tolist()  # Excluding the 'Algorithm' column
bp_time = axes[1].boxplot(box_data_time, positions=np.arange(len(box_data_time)), patch_artist=True,
                          whis=9999999, medianprops=dict(color="black"))

# Set the custom colors for each box in both plots
for patch, color in zip(bp_steps['boxes'], custom_palette):
    patch.set_facecolor(color)
for patch, color in zip(bp_time['boxes'], custom_palette):
    patch.set_facecolor(color)

# Set titles, and adjust the appearance of each subplot
for i, ax in enumerate(axes):
    ax.set_title(titles[i])
    ax.set_xticks([])  # Removing x-ticks as we have the legend to indicate algorithms

# Create custom legends from the 'Algorithm' column of df_steps or df_time as they are the same
legend_patches = [plt.Line2D([0], [0], color=custom_palette[i], lw=4, label=df_steps['Algorithm'][i])
                  for i in range(len(df_steps['Algorithm']))]
fig.legend(handles=legend_patches, loc='center right', bbox_to_anchor=(1, 0.5), title="Algorithms")

# Show the plots
plt.show()