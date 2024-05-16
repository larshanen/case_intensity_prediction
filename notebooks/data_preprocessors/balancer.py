import pandas as pd
import numpy as np
from scipy.stats import skew
import matplotlib.pyplot as plt

# Define function to reduce skewness, assuming positive skewness starting the downsampling from class '0'
def balancer(X_in, y_in, final_heavy_class: int, downsampling_perc_heaviest):
    # ADDRESS IMBALANCE IN THE FIRST TARGET COLUMN
    # Make copies of X and y
    X = X_in.copy()
    y = y_in.copy()
    
    # Create series to store all frequencies
    classes = pd.Series(range(int(y['Dagen_tot_laatst'].max() + 1)))
    counts = y['Dagen_tot_laatst'].value_counts().sort_index()
    frequencies = counts.reindex(classes, fill_value=0)

    # Disregard classes exceeding final_heavy_class
    classes_to_downsample = frequencies[:final_heavy_class+1]

    # Calculate median class size for disregarded classes
    subseq_classes = frequencies[final_heavy_class+1:final_heavy_class+round(final_heavy_class*0.25)]
    median_subseq_class_size = np.median(subseq_classes.tolist())

    # Define downsampling percentages for all downsampling classes
    downsampling_percs = []
    indices_removed = []
    max_class_downsample = max(classes_to_downsample) - median_subseq_class_size
    for class_label, class_size in classes_to_downsample.items():
        if class_size > median_subseq_class_size:
            class_size = class_size - median_subseq_class_size
        downsampling_perc = (class_size * downsampling_perc_heaviest) / max_class_downsample
        downsampling_percs.append(downsampling_perc)

        # Calculate amount of instances to be removed
        n = round(class_size * downsampling_perc)

        # Randomly choose instances to remove
        class_indices_candidates = y[y['Dagen_tot_laatst'] == class_label].index
        np.random.seed(42)
        class_indices_remove = np.random.choice(class_indices_candidates, size=int(n), replace=False)
        indices_removed.extend(class_indices_remove)

    # Remove all indices in indices_to_remove from df with inplace=True argument
    y.drop(indices_removed, inplace=True)
    X.drop(indices_removed, inplace=True)

    # ADDRESS IMBALANCE IN THE SECOND TARGET COLUMN
    # Create series to store all frequencies
    classes_2 = pd.Series(range(int(y['No_interventions'].max() + 1)))
    counts_2 = y['No_interventions'].value_counts().sort_index()
    frequencies_2 = counts_2.reindex(classes_2, fill_value=0)

    # Disregard classes exceeding final_heavy_class
    classes_to_downsample_2 = frequencies_2[1:5] # Disregard class '0'

    # Calculate median class size for disregarded classes
    subseq_classes_2 = frequencies_2[5:8]
    median_subseq_class_size_2 = np.median(subseq_classes_2.tolist())

    # Define downsampling percentages for all downsampling classes
    downsampling_percs_2 = []
    indices_removed_2 = []
    max_class_downsample_2 = max(classes_to_downsample_2) - median_subseq_class_size_2
    for class_label_2, class_size_2 in classes_to_downsample_2.items():
        if class_size_2 > median_subseq_class_size_2:
            class_size_2 = class_size_2 - median_subseq_class_size_2
        downsampling_perc_2 = (class_size_2 * downsampling_perc_heaviest) / max_class_downsample_2
        downsampling_percs_2.append(downsampling_perc_2)

        # Calculate amount of instances to be removed
        n_2 = round(class_size_2 * downsampling_perc_2)

        # Randomly choose instances to remove
        class_indices_candidates_2 = y[y['No_interventions'] == class_label_2].index
        np.random.seed(42)
        class_indices_remove_2 = np.random.choice(class_indices_candidates_2, size=int(n_2), replace=False)
        indices_removed_2.extend(class_indices_remove_2)
    
    # Remove all indices in indices_to_remove from df with inplace=True argument
    y.drop(indices_removed_2, inplace=True)
    X.drop(indices_removed_2, inplace=True)

    return X, y

# Define the function to visualize the skewness of a column
def balance_visualizer(y, name: str, subplot_size=(6, 6)):
    # Calculate total figure size based on subplot size
    fig_width = subplot_size[0] * 2
    fig_height = subplot_size[1]

    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height))

    # Store frequencies
    frequencies_interventions = y['No_interventions'].value_counts().sort_index()
    frequencies_days = y['Dagen_tot_laatst'].value_counts().sort_index()

    ## INTERVENTIONS
    # Create a bar chart
    axes[0].bar(frequencies_interventions.index, frequencies_interventions)

    # Add skewness to the plot
    axes[0].text(0.95, 0.95, f'Skewness: {skew(y['No_interventions'], nan_policy='omit'):.2f}', transform=axes[0].transAxes, fontsize=18,
            verticalalignment='top', horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.8))

    # Add labels and title
    axes[0].set_xlabel('Interventions')
    axes[0].set_ylabel('Frequency')
    axes[0].set_xlim([-1, 25])
    axes[0].set_title('Amount of interventions until case closure')

    ## DAYS
    # Create a bar chart
    axes[1].bar(frequencies_days.index, frequencies_days)

    # Add skewness to the plot
    axes[1].text(0.95, 0.95, f'Skewness: {skew(y['Dagen_tot_laatst'], nan_policy='omit'):.2f}', transform=axes[1].transAxes, fontsize=18,
            verticalalignment='top', horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.8))

    # Add labels and title
    axes[1].set_xlabel('Business days')
    axes[1].set_ylabel('Frequency')
    axes[1].set_xlim([-1, 25])
    axes[1].set_title('Amount of business days until last intervention')

    # Show the plot
    plt.savefig(f'../visualizations/overleaf_final/label_distribution_{name}.png', bbox_inches='tight', transparent=True)
    plt.show()


# Define the function to visualize the skewness of a column
def balance_visualizer_log(y, name: str, subplot_size=(6, 6)):
    # Calculate total figure size based on subplot size
    fig_width = subplot_size[0] * 2
    fig_height = subplot_size[1]

    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height))

    # Store frequencies
    frequencies_interventions = round(y['No_interventions']).value_counts().sort_index()
    frequencies_days = round(y['Dagen_tot_laatst']).value_counts().sort_index()

    ## INTERVENTIONS
    # Create a bar chart
    axes[0].bar(frequencies_interventions.index, frequencies_interventions)

    # Add skewness to the plot
    axes[0].text(0.95, 0.95, f'Skewness: {skew(y['No_interventions'], nan_policy='omit'):.2f}', transform=axes[0].transAxes, fontsize=18,
            verticalalignment='top', horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.8))

    # Add labels and title
    axes[0].set_xlabel('Interventions')
    axes[0].set_ylabel('Frequency')
    axes[0].set_xlim([-1, 25])
    axes[0].set_title('Amount of interventions until case closure')

    ## DAYS
    # Create a bar chart
    axes[1].bar(frequencies_days.index, frequencies_days)

    # Add skewness to the plot
    axes[1].text(0.95, 0.95, f'Skewness: {skew(y['Dagen_tot_laatst'], nan_policy='omit'):.2f}', transform=axes[1].transAxes, fontsize=18,
            verticalalignment='top', horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.8))

    # Add labels and title
    axes[1].set_xlabel('Business days')
    axes[1].set_ylabel('Frequency')
    axes[1].set_xlim([-1, 25])
    axes[1].set_title('Amount of business days until last intervention')

    # Show the plot
    plt.savefig(f'../visualizations/overleaf_final/label_distribution_{name}.png', bbox_inches='tight', transparent=True)
    plt.show()