import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from math import pi
from mpl_toolkits.mplot3d import Axes3D


def plot_ap_importance(importance_scores):
    """
    Plots the importance scores for all APs as individual vertical bars.
    """
    if not importance_scores:
        print("No importance scores available to plot.")
        return

    # Get AP numbers and corresponding scores (keep original order or sort by AP number)
    ap_numbers = np.arange(len(importance_scores))
    scores = [score for _, score in importance_scores.items()]

    plt.figure(figsize=(10, 6))
    plt.bar(ap_numbers, scores, width=1.0, color='steelblue', edgecolor='none')
    plt.xlabel("AP Number", fontsize=16)
    plt.ylabel("Importance", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim([0, len(ap_numbers)])
    plt.tight_layout()
    plt.show()

def plot_redundancy_matrix(redundancy_matrix, importance_dict=None):
    """
    Plots the redundancy matrix as a heatmap with reduced axis labels.
    Optionally filters to show only APs with non-zero importance.

    Parameters:
    -----------
    redundancy_matrix : DataFrame
        The redundancy (correlation) matrix calculated for APs.
    importance_dict : dict, optional
        Dictionary mapping AP names to importance scores.
        If provided, only APs with non-zero importance will be shown.
    """
    if redundancy_matrix.empty:
        print("Redundancy matrix is empty. Nothing to plot.")
        return

    # Filter by importance if provided
    if importance_dict is not None:
        # Get APs with non-zero importance
        nonzero_aps = [ap for ap, score in importance_dict.items() if score > 0]

        # Filter redundancy matrix to only include non-zero importance APs
        redundancy_matrix = redundancy_matrix.loc[nonzero_aps, nonzero_aps]

        print(f"Filtered to {len(nonzero_aps)} APs with non-zero importance (from {len(importance_dict)} total APs)")

    if redundancy_matrix.empty:
        print("No APs with non-zero importance. Nothing to plot.")
        return

    plt.figure(figsize=(10, 8))
    sns.heatmap(redundancy_matrix, cmap='viridis', cbar_kws={'label': 'Absolute Pearson Correlation'})

    # Limit tick labels (e.g., show 5 evenly spaced labels)
    n_ticks = 10
    xticks = np.linspace(0, len(redundancy_matrix.columns) - 1, n_ticks, dtype=int)
    yticks = np.linspace(0, len(redundancy_matrix.index) - 1, n_ticks, dtype=int)

    plt.xticks(xticks, [redundancy_matrix.columns[i] for i in xticks], rotation=45, ha='right', fontsize=14)
    plt.yticks(yticks, [redundancy_matrix.index[i] for i in yticks], rotation=0, fontsize=14)

    plt.xlabel("Access Point (AP)", fontsize=16)
    plt.ylabel("Access Point (AP)", fontsize=16)
    plt.tight_layout()
    plt.show()


# Plots Results per Budget 
def plot_sa_error_vs_budget(df, ax=None):
    """
    Plot Simulated Annealing mean localization error vs. budget.

    Parameters:
    -----------
    df : pandas.DataFrame
        Summary DataFrame containing columns ['Budget', 'Solver', 'Mean_Error_m'].
    ax : matplotlib.axes._axes.Axes, optional
        Matplotlib Axes on which to plot. If None, a new figure and axes are created.

    Returns:
    --------
    matplotlib.axes._axes.Axes
        The axes containing the plot.
    """
    # Filter for Simulated Annealing solver
    df_sa = df[df['Solver'] == 'SA'].sort_values('Budget')

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    sns.lineplot(
        data=df_sa,
        x='Budget',
        y='Mean_Error_m',
        marker='o',
        color='tab:blue',
        ax=ax,
        linewidth=3,
        markersize=10
    )
    ax.set_xlabel('Budget (k)', fontsize=16)
    ax.set_ylabel('Mean Error (m)', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.grid(True)
    return ax

def plot_sa_floor_accuracy_vs_budget(df, ax=None):
    """
    Plot Simulated Annealing floor accuracy vs. budget.

    Parameters:
    -----------
    df : pandas.DataFrame
        Summary DataFrame containing columns ['Budget', 'Solver', 'Floor_Accuracy'].
    ax : matplotlib.axes._axes.Axes, optional
        Matplotlib Axes on which to plot. If None, a new figure and axes are created.

    Returns:
    --------
    matplotlib.axes._axes.Axes
        The axes containing the plot.
    """
    # Filter for Simulated Annealing solver
    df_sa = df[df['Solver'] == 'SA'].sort_values('Budget')

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    sns.lineplot(
        data=df_sa,
        x='Budget',
        y='Floor_Accuracy',
        marker='o',
        color='tab:green',
        ax=ax,
        linewidth=3,
        markersize=10
    )
    ax.set_xlabel('Budget (k)', fontsize=16)
    ax.set_ylabel('Floor Accuracy', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_ylim(0, 1)
    ax.grid(True)
    return ax

def plot_error_cdf_by_budget(df, budget, ax=None):
    """
    Plot Error_Meters vs. Cumulative_Percentage for all solvers at a given budget,
    converting percentages to fractions if needed, as line plots.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing columns ['Budget', 'Solver', 'Error_Meters', 'Cumulative_Percentage'].
    budget : int
        Budget value (k) to filter on.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, a new figure and axes are created.

    Returns:
    --------
    matplotlib.axes.Axes
        The axes containing the plot.
    """
    # Copy and filter by budget
    df_plot = df[df['Budget'] == budget].copy()

    # Convert percentage to fraction if necessary
    if df_plot['Cumulative_Percentage'].max() > 1:
        df_plot['Cumulative_Percentage'] /= 100.0

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    # Plot a line for each solver
    for solver, grp in df_plot.groupby('Solver'):
        sns.lineplot(
            data=grp,
            x='Error_Meters',
            y='Cumulative_Percentage',
            label=solver,
            ax=ax,
            linewidth=3
        )

    ax.set_xlabel('3D Localization Error (m)', fontsize=16)
    ax.set_ylabel('Cumulative Fraction', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1)
    ax.grid(True)
    ax.legend(title='Solver', fontsize=14, title_fontsize=14)
    plt.tight_layout()
    return ax








# Importance Plots for Comparison of Feature Selection Methods

def plot_error_comparison(results, save_path=None):
    """
    Compare 3D positioning errors (median only) across methods

    Parameters:
    results: dict with structure {method: {metric: value}}
    save_path: optional path to save the figure
    """
    methods = list(results.keys())
    median_errors = [results[method]['median_3d_error'] for method in methods]

    x = np.arange(len(methods))

    fig, ax = plt.subplots(figsize=(12, 7))

    bars = ax.bar(x, median_errors, width=0.6, label='Median 3D Error',
                  color='coral', linewidth=2, edgecolor='black')

    ax.set_xlabel('Feature Selection Method', fontsize=16, fontweight='bold')
    ax.set_ylabel('Error (meters)', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=14, fontweight='bold')
    ax.tick_params(axis='y', labelsize=14)
    ax.legend(fontsize=13, frameon=True, shadow=True)
    ax.grid(axis='y', alpha=0.3, linewidth=1.5)

    # Make tick marks thicker
    ax.tick_params(width=2, length=6)
    for spine in ax.spines.values():
        spine.set_linewidth(2)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.2f}',
               ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_floor_accuracy_comparison(results, save_path=None):
    """
    Compare floor accuracy across methods

    Parameters:
    results: dict with structure {method: {metric: value}}
    save_path: optional path to save the figure
    """
    methods = list(results.keys())
    floor_accuracies = [results[method]['floor_accuracy'] * 100 for method in methods]

    fig, ax = plt.subplots(figsize=(12, 7))

    # Generate colors dynamically based on number of methods
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(methods)))

    bars = ax.bar(methods, floor_accuracies,
                  color=colors,
                  linewidth=2, edgecolor='black')

    ax.set_xlabel('Feature Selection Method', fontsize=16, fontweight='bold')
    ax.set_ylabel('Floor Accuracy (%)', fontsize=16, fontweight='bold')
    ax.set_xticklabels(methods, fontsize=14, fontweight='bold')
    ax.tick_params(axis='y', labelsize=14)
    ax.grid(axis='y', alpha=0.3, linewidth=1.5)
    ax.set_ylim(0, 100)
    
    # Make tick marks thicker
    ax.tick_params(width=2, length=6)
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.2f}%',
               ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_error_range_comparison(results, save_path=None):
    """
    Compare error ranges (min to max) across methods

    Parameters:
    results: dict with structure {method: {metric: value}}
    save_path: optional path to save the figure
    """
    methods = list(results.keys())
    min_errors = [results[method]['real_min_m'] for method in methods]
    max_errors = [results[method]['real_max_m'] for method in methods]

    x = np.arange(len(methods))

    fig, ax = plt.subplots(figsize=(12, 7))

    # Generate colors dynamically based on number of methods
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(methods)))

    # Plot error ranges as vertical lines with markers
    for i, method in enumerate(methods):
        ax.plot([i, i], [min_errors[i], max_errors[i]], 'o-',
                linewidth=4, markersize=10, label=method, color=colors[i])

    ax.set_xlabel('Feature Selection Method', fontsize=16, fontweight='bold')
    ax.set_ylabel('Error (meters)', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=14, fontweight='bold')
    ax.tick_params(axis='y', labelsize=14)
    ax.grid(axis='y', alpha=0.3, linewidth=1.5)

    # Make tick marks thicker
    ax.tick_params(width=2, length=6)
    for spine in ax.spines.values():
        spine.set_linewidth(2)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color=colors[i], linewidth=4,
                             markersize=10, label=method) for i, method in enumerate(methods)]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=13,
              frameon=True, shadow=True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_radar_chart(results, save_path=None):
    """
    Create radar chart comparing methods across normalized metrics

    Parameters:
    results: dict with structure {method: {metric: value}}
    save_path: optional path to save the figure
    """
    methods = list(results.keys())
    # Select metrics (excluding duration and mean)
    metrics = ['median_3d_error', 'real_min_m', 'real_max_m', 'floor_accuracy']
    metric_labels = ['Median 3D\nError', 'Min\nError', 'Max\nError', 'Floor\nAccuracy']
    
    # Extract and normalize data
    # For errors: lower is better, so we invert them (1 - normalized)
    # For accuracy: higher is better, keep as is
    data_normalized = {}
    for method in methods:
        values = []
        for metric in metrics:
            val = results[method][metric]
            # Get min and max for normalization
            all_vals = [results[m][metric] for m in methods]
            min_val, max_val = min(all_vals), max(all_vals)
            
            if metric == 'floor_accuracy':
                # Higher is better
                normalized = (val - min_val) / (max_val - min_val) if max_val != min_val else 0.5
            else:
                # Lower is better (errors)
                normalized = 1 - ((val - min_val) / (max_val - min_val)) if max_val != min_val else 0.5
            
            values.append(normalized)
        data_normalized[method] = values
    
    # Number of variables
    num_vars = len(metrics)
    
    # Compute angle for each axis
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]
    
    # Initialize plot
    fig, ax = plt.subplots(figsize=(11, 11), subplot_kw=dict(polar=True))

    # Generate colors dynamically based on number of methods
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(methods)))

    # Plot data
    for idx, method in enumerate(methods):
        values = data_normalized[method]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=4, label=method,
                color=colors[idx], markersize=10)
        ax.fill(angles, values, alpha=0.2, color=colors[idx])

    # Fix axis to go in the right order
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=15, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0.25', '0.5', '0.75', '1.0'], fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=14,
              frameon=True, shadow=True)
    ax.grid(True, linewidth=1.5)
    
    # Make radial grid lines thicker
    ax.tick_params(width=2, length=6)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_combined_error_and_accuracy(results, save_path=None):
    """
    Create subplot with error comparison and floor accuracy side by side

    Parameters:
    results: dict with structure {method: {metric: value}}
    save_path: optional path to save the figure
    """
    methods = list(results.keys())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # Left plot: 3D Error comparison (median only)
    median_errors = [results[method]['median_3d_error'] for method in methods]

    x = np.arange(len(methods))

    bars1 = ax1.bar(x, median_errors, width=0.6, label='Median 3D Error',
                    color='coral', linewidth=2, edgecolor='black')

    ax1.set_xlabel('Feature Selection Method', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Error (meters)', fontsize=16, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, fontsize=14, fontweight='bold')
    ax1.tick_params(axis='y', labelsize=14)
    ax1.legend(fontsize=13, frameon=True, shadow=True)
    ax1.grid(axis='y', alpha=0.3, linewidth=1.5)

    # Make tick marks thicker
    ax1.tick_params(width=2, length=6)
    for spine in ax1.spines.values():
        spine.set_linewidth(2)

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Right plot: Floor accuracy
    floor_accuracies = [results[method]['floor_accuracy'] * 100 for method in methods]

    # Generate colors dynamically based on number of methods
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(methods)))

    bars3 = ax2.bar(methods, floor_accuracies,
                    color=colors,
                    linewidth=2, edgecolor='black')

    ax2.set_xlabel('Feature Selection Method', fontsize=16, fontweight='bold')
    ax2.set_ylabel('Floor Accuracy (%)', fontsize=16, fontweight='bold')
    ax2.set_xticklabels(methods, fontsize=14, fontweight='bold')
    ax2.tick_params(axis='y', labelsize=14)
    ax2.grid(axis='y', alpha=0.3, linewidth=1.5)
    ax2.set_ylim(0, 100)
    
    # Make tick marks thicker
    ax2.tick_params(width=2, length=6)
    for spine in ax2.spines.values():
        spine.set_linewidth(2)
    
    # Add value labels
    for bar in bars3:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_building_fingerprints_3d(coords, rssi_data=None, selected_aps=None,
                                   color_by='floor', figsize=(14, 10),
                                   denormalize=False, lon_min=None, lon_max=None,
                                   lat_min=None, lat_max=None, floor_height=3.0,
                                   title='Building WiFi Fingerprints - 3D View',
                                   save_path=None, show_grid=True, alpha=0.6,
                                   marker_size=50, elev=20, azim=45):
    """
    Plot building fingerprints in 3D space (Longitude, Latitude, Floor).

    Parameters:
    -----------
    coords : numpy.ndarray or pandas.DataFrame
        Coordinate data with shape (n_samples, 3) containing [LON, LAT, FLOOR].
        Can be normalized or denormalized coordinates.

    rssi_data : pandas.DataFrame, optional
        RSSI data for the fingerprints. If provided with selected_aps,
        can color points by average signal strength of selected APs.

    selected_aps : list, optional
        List of selected AP column names. Used for coloring by signal strength.

    color_by : str, optional
        What to color the points by:
        - 'floor': Color by floor level (default)
        - 'rssi': Color by average RSSI of selected APs (requires rssi_data and selected_aps)
        - 'density': Color by spatial density of measurements

    figsize : tuple, optional
        Figure size (width, height). Default: (14, 10)

    denormalize : bool, optional
        If True, denormalize coordinates using provided min/max values. Default: False

    lon_min, lon_max, lat_min, lat_max : float, optional
        Min/max values for denormalization (required if denormalize=True)

    floor_height : float, optional
        Height of each floor in meters for Z-axis scaling. Default: 3.0

    title : str, optional
        Plot title. Default: 'Building WiFi Fingerprints - 3D View'

    save_path : str, optional
        Path to save the figure. If None, figure is not saved.

    show_grid : bool, optional
        Show grid lines. Default: True

    alpha : float, optional
        Transparency of points (0-1). Default: 0.6

    marker_size : int, optional
        Size of scatter plot markers. Default: 50

    elev : float, optional
        Elevation viewing angle in degrees. Default: 20

    azim : float, optional
        Azimuth viewing angle in degrees. Default: 45

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    ax : matplotlib.axes._axes.Axes3D
        The 3D axes object
    )
    """

    # Convert to numpy array if DataFrame
    if isinstance(coords, pd.DataFrame):
        coords_array = coords.values
    else:
        coords_array = coords.copy()

    # Extract coordinates
    lon = coords_array[:, 0]
    lat = coords_array[:, 1]
    floor = coords_array[:, 2]

    # Denormalize if requested
    if denormalize:
        if lon_min is None or lon_max is None or lat_min is None or lat_max is None:
            raise ValueError("Must provide lon_min, lon_max, lat_min, lat_max for denormalization")
        lon = lon * (lon_max - lon_min) + lon_min
        lat = lat * (lat_max - lat_min) + lat_min

    # Calculate Z-axis (floor height in meters)
    z = floor * floor_height

    # Determine colors
    if color_by == 'floor':
        colors = floor
        cmap = plt.cm.viridis
        cbar_label = 'Floor Level'
    elif color_by == 'rssi':
        if rssi_data is None or selected_aps is None:
            raise ValueError("Must provide rssi_data and selected_aps for color_by='rssi'")
        # Calculate average RSSI for selected APs
        colors = rssi_data[selected_aps].mean(axis=1).values
        cmap = plt.cm.plasma
        cbar_label = 'Average RSSI (Selected APs)'
    elif color_by == 'density':
        # Calculate local density using KDE or nearest neighbors
        from scipy.spatial import cKDTree
        tree = cKDTree(coords_array)
        # Count neighbors within radius
        radius = 0.05 if not denormalize else 5.0  # Adjust based on coordinate scale
        colors = np.array([len(tree.query_ball_point(point, radius)) for point in coords_array])
        cmap = plt.cm.hot_r
        cbar_label = 'Measurement Density'
    else:
        raise ValueError(f"Unknown color_by option: {color_by}. Choose 'floor', 'rssi', or 'density'")

    # Create 3D plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot
    scatter = ax.scatter(lon, lat, z, c=colors, cmap=cmap,
                        s=marker_size, alpha=alpha, edgecolors='k', linewidth=0.5)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label(cbar_label, fontsize=14, fontweight='bold')
    cbar.ax.tick_params(labelsize=12)

    # Labels
    ax.set_xlabel('Longitude' + (' (m)' if denormalize else ' (normalized)'),
                  fontsize=16, fontweight='bold')
    ax.set_ylabel('Latitude' + (' (m)' if denormalize else ' (normalized)'),
                  fontsize=16, fontweight='bold')
    ax.set_zlabel('Height (m)', fontsize=16, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=14)

    # Grid
    if show_grid:
        ax.grid(True, alpha=0.3)

    # Set viewing angle
    ax.view_init(elev=elev, azim=azim)

    # Add floor level markers on Z-axis
    unique_floors = np.unique(floor)
    for fl in unique_floors:
        z_pos = fl * floor_height
        ax.text(ax.get_xlim()[0], ax.get_ylim()[0], z_pos,
               f'Floor {int(fl)}', fontsize=10, color='red')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    plt.show()

    return fig, ax


def plot_building_comparison_3d(coords_true, coords_pred,
                                 denormalize=False, lon_min=None, lon_max=None,
                                 lat_min=None, lat_max=None, floor_height=3.0,
                                 max_points=500, figsize=(16, 10),
                                 title='True vs Predicted Locations - 3D View',
                                 save_path=None):
    """
    Plot true vs predicted locations in 3D with error vectors.

    Parameters:
    -----------
    coords_true : numpy.ndarray
        True coordinates with shape (n_samples, 3) containing [LON, LAT, FLOOR]

    coords_pred : numpy.ndarray
        Predicted coordinates with shape (n_samples, 3)

    denormalize : bool, optional
        If True, denormalize coordinates. Default: False

    lon_min, lon_max, lat_min, lat_max : float, optional
        Min/max values for denormalization (required if denormalize=True)

    floor_height : float, optional
        Height of each floor in meters. Default: 3.0

    max_points : int, optional
        Maximum number of points to plot (for performance). Default: 500

    figsize : tuple, optional
        Figure size. Default: (16, 10)

    title : str, optional
        Plot title

    save_path : str, optional
        Path to save the figure

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    ax : matplotlib.axes._axes.Axes3D
        The 3D axes object
    """

    # Sample points if too many
    n_samples = coords_true.shape[0]
    if n_samples > max_points:
        indices = np.random.choice(n_samples, max_points, replace=False)
        coords_true = coords_true[indices]
        coords_pred = coords_pred[indices]
        print(f"Plotting {max_points} random samples out of {n_samples} total points")

    # Extract coordinates
    lon_true = coords_true[:, 0]
    lat_true = coords_true[:, 1]
    floor_true = coords_true[:, 2]

    lon_pred = coords_pred[:, 0]
    lat_pred = coords_pred[:, 1]
    floor_pred = coords_pred[:, 2]

    # Denormalize if requested
    if denormalize:
        if lon_min is None or lon_max is None or lat_min is None or lat_max is None:
            raise ValueError("Must provide lon_min, lon_max, lat_min, lat_max for denormalization")
        lon_true = lon_true * (lon_max - lon_min) + lon_min
        lat_true = lat_true * (lat_max - lat_min) + lat_min
        lon_pred = lon_pred * (lon_max - lon_min) + lon_min
        lat_pred = lat_pred * (lat_max - lat_min) + lat_min

    # Calculate Z-axis
    z_true = floor_true * floor_height
    z_pred = floor_pred * floor_height

    # Calculate 3D errors
    errors_3d = np.sqrt((lon_true - lon_pred)**2 +
                        (lat_true - lat_pred)**2 +
                        (z_true - z_pred)**2)

    # Create 3D plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # Plot true locations (green)
    ax.scatter(lon_true, lat_true, z_true,
              c='green', s=80, alpha=0.6, label='True Location',
              marker='o', edgecolors='k', linewidth=0.5)

    # Plot predicted locations (red) colored by error magnitude
    scatter = ax.scatter(lon_pred, lat_pred, z_pred,
                        c=errors_3d, cmap='Reds', s=80, alpha=0.6,
                        label='Predicted Location', marker='^',
                        edgecolors='k', linewidth=0.5)

    # Draw error vectors (lines from true to predicted)
    for i in range(len(lon_true)):
        ax.plot([lon_true[i], lon_pred[i]],
               [lat_true[i], lat_pred[i]],
               [z_true[i], z_pred[i]],
               'b-', alpha=0.3, linewidth=1)

    # Add colorbar for error magnitude
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label('3D Error' + (' (m)' if denormalize else ' (normalized)'),
                   fontsize=14, fontweight='bold')
    cbar.ax.tick_params(labelsize=12)

    # Labels
    ax.set_xlabel('Longitude' + (' (m)' if denormalize else ' (normalized)'),
                  fontsize=16, fontweight='bold')
    ax.set_ylabel('Latitude' + (' (m)' if denormalize else ' (normalized)'),
                  fontsize=16, fontweight='bold')
    ax.set_zlabel('Height (m)', fontsize=16, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=14)

    # Legend
    ax.legend(loc='upper left', fontsize=13)

    # Grid
    ax.grid(True, alpha=0.3)

    # Statistics text box
    stats_text = f"Mean Error: {np.mean(errors_3d):.2f}\n"
    stats_text += f"Median Error: {np.median(errors_3d):.2f}\n"
    stats_text += f"90th Percentile: {np.percentile(errors_3d, 90):.2f}"
    ax.text2D(0.02, 0.98, stats_text, transform=ax.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    plt.show()

    return fig, ax


# ============================================================================
# ANNEALING-SPECIFIC VISUALIZATION FUNCTIONS
# ============================================================================

def plot_tts_comparison(results_dict, save_path=None):
    """
    Plot Time-to-Solution (TTS) comparison between OpenJij and D-Wave SA.

    Parameters:
    -----------
    results_dict : dict
        Dictionary with keys 'openjij' and 'dwave_sa', each containing
        a DataFrame with columns: num_sweeps, tts, avg_time
    save_path : str, optional
        Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    colors = {'openjij': 'tab:blue', 'dwave_sa': 'tab:orange'}
    markers = {'openjij': 'o', 'dwave_sa': 's'}
    labels = {'openjij': 'OpenJij SQA', 'dwave_sa': 'D-Wave SA'}

    for solver, df in results_dict.items():
        # Filter out infinite TTS values
        valid_mask = df['tts'] != np.inf
        df_valid = df[valid_mask]

        if len(df_valid) > 0:
            ax.plot(df_valid['num_sweeps'], df_valid['tts'],
                   marker=markers[solver], linewidth=3, markersize=10,
                   label=labels[solver], color=colors[solver])

    ax.set_xlabel('Number of Sweeps', fontsize=16, fontweight='bold')
    ax.set_ylabel('Time-to-Solution (seconds)', fontsize=16, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.legend(fontsize=14, frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, linewidth=1.5)
    ax.set_xscale('log')
    ax.set_yscale('log')

    # Make tick marks thicker
    ax.tick_params(width=2, length=6)
    for spine in ax.spines.values():
        spine.set_linewidth(2)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_success_probability(results_dict, save_path=None):
    """
    Plot success probability vs. num_sweeps for both solvers.

    Parameters:
    -----------
    results_dict : dict
        Dictionary with keys 'openjij' and 'dwave_sa', each containing
        a DataFrame with columns: num_sweeps, success_prob
    save_path : str, optional
        Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    colors = {'openjij': 'tab:blue', 'dwave_sa': 'tab:orange'}
    markers = {'openjij': 'o', 'dwave_sa': 's'}
    labels = {'openjij': 'OpenJij SQA', 'dwave_sa': 'D-Wave SA'}

    for solver, df in results_dict.items():
        ax.plot(df['num_sweeps'], df['success_prob'] * 100,
               marker=markers[solver], linewidth=3, markersize=10,
               label=labels[solver], color=colors[solver])

    ax.set_xlabel('Number of Sweeps', fontsize=16, fontweight='bold')
    ax.set_ylabel('Success Probability (%)', fontsize=16, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.legend(fontsize=14, frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, linewidth=1.5)
    ax.set_ylim(0, 100)

    # Make tick marks thicker
    ax.tick_params(width=2, length=6)
    for spine in ax.spines.values():
        spine.set_linewidth(2)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_energy_convergence(convergence_df, solver_name='Solver', save_path=None):
    """
    Plot energy convergence showing how solution quality improves with num_sweeps.

    Parameters:
    -----------
    convergence_df : pandas.DataFrame
        DataFrame with columns: num_sweeps, best_energy, mean_energy, std_energy
    solver_name : str, optional
        Name of the solver for the legend. Default: 'Solver'
    save_path : str, optional
        Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot best energy
    ax.plot(convergence_df['num_sweeps'], convergence_df['best_energy'],
           marker='o', linewidth=3, markersize=10, label='Best Energy',
           color='green')

    # Plot mean energy with error bars (std)
    ax.errorbar(convergence_df['num_sweeps'], convergence_df['mean_energy'],
               yerr=convergence_df['std_energy'], marker='s', linewidth=3,
               markersize=10, label='Mean Energy ± Std', color='blue',
               capsize=5, capthick=2)

    # Plot 90th percentile
    ax.plot(convergence_df['num_sweeps'], convergence_df['q90_energy'],
           marker='^', linewidth=3, markersize=10, label='90th Percentile',
           color='red', linestyle='--')

    ax.set_xlabel('Number of Sweeps', fontsize=16, fontweight='bold')
    ax.set_ylabel('Energy', fontsize=16, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.legend(fontsize=14, frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, linewidth=1.5)
    ax.set_xscale('log')

    # Make tick marks thicker
    ax.tick_params(width=2, length=6)
    for spine in ax.spines.values():
        spine.set_linewidth(2)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_residual_energy(results_dict, save_path=None):
    """
    Plot residual energy comparison between solvers.

    Parameters:
    -----------
    results_dict : dict
        Dictionary with keys 'openjij' and 'dwave_sa', each containing
        a DataFrame with columns: num_sweeps, mean_residual, std_residual
    save_path : str, optional
        Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    colors = {'openjij': 'tab:blue', 'dwave_sa': 'tab:orange'}
    markers = {'openjij': 'o', 'dwave_sa': 's'}
    labels = {'openjij': 'OpenJij SQA', 'dwave_sa': 'D-Wave SA'}

    for solver, df in results_dict.items():
        ax.errorbar(df['num_sweeps'], df['mean_residual'],
                   yerr=df['std_residual'], marker=markers[solver],
                   linewidth=3, markersize=10, label=labels[solver],
                   color=colors[solver], capsize=5, capthick=2)

    ax.set_xlabel('Number of Sweeps', fontsize=16, fontweight='bold')
    ax.set_ylabel('Residual Energy (Mean ± Std)', fontsize=16, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.legend(fontsize=14, frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, linewidth=1.5)

    # Make tick marks thicker
    ax.tick_params(width=2, length=6)
    for spine in ax.spines.values():
        spine.set_linewidth(2)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_energy_distribution(energies, solver_name='Solver', bins=50, save_path=None):
    """
    Plot histogram of energy distribution across multiple annealing runs.

    Parameters:
    -----------
    energies : array-like
        Array of energy values from multiple runs
    solver_name : str, optional
        Name of the solver for the title. Default: 'Solver'
    bins : int, optional
        Number of histogram bins. Default: 50
    save_path : str, optional
        Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    # Histogram
    n, bins_edges, patches = ax.hist(energies, bins=bins, alpha=0.7,
                                     color='steelblue', edgecolor='black',
                                     linewidth=1.5)

    # Add statistical lines
    mean_energy = np.mean(energies)
    median_energy = np.median(energies)
    best_energy = np.min(energies)

    ax.axvline(best_energy, color='green', linestyle='--', linewidth=3,
              label=f'Best: {best_energy:.2f}')
    ax.axvline(mean_energy, color='red', linestyle='--', linewidth=3,
              label=f'Mean: {mean_energy:.2f}')
    ax.axvline(median_energy, color='orange', linestyle='--', linewidth=3,
              label=f'Median: {median_energy:.2f}')

    ax.set_xlabel('Energy', fontsize=16, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=16, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.legend(fontsize=14, frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, linewidth=1.5, axis='y')

    # Make tick marks thicker
    ax.tick_params(width=2, length=6)
    for spine in ax.spines.values():
        spine.set_linewidth(2)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_solver_comparison_metrics(results_dict, save_path=None):
    """
    Create comprehensive 2x2 subplot comparing key metrics between solvers.

    Parameters:
    -----------
    results_dict : dict
        Dictionary with keys 'openjij' and 'dwave_sa', each containing
        a DataFrame with benchmark results
    save_path : str, optional
        Path to save the figure
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))

    colors = {'openjij': 'tab:blue', 'dwave_sa': 'tab:orange'}
    markers = {'openjij': 'o', 'dwave_sa': 's'}
    labels = {'openjij': 'OpenJij SQA', 'dwave_sa': 'D-Wave SA'}

    # Plot 1: Success Probability
    for solver, df in results_dict.items():
        ax1.plot(df['num_sweeps'], df['success_prob'] * 100,
                marker=markers[solver], linewidth=3, markersize=8,
                label=labels[solver], color=colors[solver])

    ax1.set_xlabel('Number of Sweeps', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Success Probability (%)', fontsize=14, fontweight='bold')
    ax1.tick_params(axis='both', which='major', labelsize=12)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 100)

    # Plot 2: Average Time
    for solver, df in results_dict.items():
        ax2.plot(df['num_sweeps'], df['avg_time'],
                marker=markers[solver], linewidth=3, markersize=8,
                label=labels[solver], color=colors[solver])

    ax2.set_xlabel('Number of Sweeps', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Average Time (seconds)', fontsize=14, fontweight='bold')
    ax2.tick_params(axis='both', which='major', labelsize=12)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Best Energy Found
    for solver, df in results_dict.items():
        ax3.plot(df['num_sweeps'], df['best_energy'],
                marker=markers[solver], linewidth=3, markersize=8,
                label=labels[solver], color=colors[solver])

    ax3.set_xlabel('Number of Sweeps', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Best Energy', fontsize=14, fontweight='bold')
    ax3.tick_params(axis='both', which='major', labelsize=12)
    ax3.legend(fontsize=12)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Mean Residual Energy
    for solver, df in results_dict.items():
        ax4.errorbar(df['num_sweeps'], df['mean_residual'],
                    yerr=df['std_residual'], marker=markers[solver],
                    linewidth=3, markersize=8, label=labels[solver],
                    color=colors[solver], capsize=4)

    ax4.set_xlabel('Number of Sweeps', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Mean Residual Energy', fontsize=14, fontweight='bold')
    ax4.tick_params(axis='both', which='major', labelsize=12)
    ax4.legend(fontsize=12)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_solution_diversity(consistency_results, save_path=None):
    """
    Visualize solution diversity using bar chart of most frequent solutions.

    Parameters:
    -----------
    consistency_results : dict
        Results from analyze_solution_consistency() function containing
        'solution_frequencies' and 'energy_landscape'
    save_path : str, optional
        Path to save the figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # Get top 20 most frequent solutions
    solution_freq = consistency_results['solution_frequencies']
    top_solutions = solution_freq.most_common(20)

    # Plot 1: Frequency of top solutions
    solutions_labels = [f"Sol {i+1}" for i in range(len(top_solutions))]
    frequencies = [count for _, count in top_solutions]

    ax1.bar(solutions_labels, frequencies, color='steelblue',
           edgecolor='black', linewidth=1.5)
    ax1.set_xlabel('Solution Rank', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=16, fontweight='bold')
    ax1.tick_params(axis='both', which='major', labelsize=14)
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3, axis='y', linewidth=1.5)

    # Plot 2: Energy landscape
    energy_landscape = consistency_results['energy_landscape']
    top_solution_keys = [sol for sol, _ in top_solutions]
    energies = [energy_landscape[sol] for sol in top_solution_keys]

    ax2.bar(solutions_labels, energies, color='coral',
           edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('Solution Rank', fontsize=16, fontweight='bold')
    ax2.set_ylabel('Average Energy', fontsize=16, fontweight='bold')
    ax2.tick_params(axis='both', which='major', labelsize=14)
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3, axis='y', linewidth=1.5)

    # Add best energy line
    best_energy = min(energies)
    ax2.axhline(best_energy, color='green', linestyle='--',
               linewidth=2, label=f'Best Energy: {best_energy:.2f}')
    ax2.legend(fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_time_vs_quality_tradeoff(results_dict, save_path=None):
    """
    Plot the trade-off between computation time and solution quality.

    Parameters:
    -----------
    results_dict : dict
        Dictionary with keys 'openjij' and 'dwave_sa', each containing
        a DataFrame with columns: avg_time, best_energy, mean_residual
    save_path : str, optional
        Path to save the figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    colors = {'openjij': 'tab:blue', 'dwave_sa': 'tab:orange'}
    markers = {'openjij': 'o', 'dwave_sa': 's'}
    labels = {'openjij': 'OpenJij SQA', 'dwave_sa': 'D-Wave SA'}

    # Plot 1: Time vs Best Energy
    for solver, df in results_dict.items():
        ax1.scatter(df['avg_time'], df['best_energy'],
                   s=200, marker=markers[solver], label=labels[solver],
                   color=colors[solver], alpha=0.7, edgecolors='black',
                   linewidth=2)

    ax1.set_xlabel('Average Time (seconds)', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Best Energy Found', fontsize=16, fontweight='bold')
    ax1.tick_params(axis='both', which='major', labelsize=14)
    ax1.legend(fontsize=14, frameon=True, shadow=True)
    ax1.grid(True, alpha=0.3, linewidth=1.5)

    # Plot 2: Time vs Mean Residual Energy
    for solver, df in results_dict.items():
        ax2.scatter(df['avg_time'], df['mean_residual'],
                   s=200, marker=markers[solver], label=labels[solver],
                   color=colors[solver], alpha=0.7, edgecolors='black',
                   linewidth=2)

    ax2.set_xlabel('Average Time (seconds)', fontsize=16, fontweight='bold')
    ax2.set_ylabel('Mean Residual Energy', fontsize=16, fontweight='bold')
    ax2.tick_params(axis='both', which='major', labelsize=14)
    ax2.legend(fontsize=14, frameon=True, shadow=True)
    ax2.grid(True, alpha=0.3, linewidth=1.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()



