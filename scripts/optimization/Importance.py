import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression

def entropy_importance(rss_data, ap_columns):
    """
    Calculate importance using entropy for each AP
    
    Parameters:
    rss_data: pandas DataFrame or numpy array of shape (n_samples, n_aps)
    ap_columns: list of AP column names (required if rss_data is DataFrame)
    
    Returns:
    importance_array: numpy array of importance scores for each AP
    importance_dict: dictionary mapping AP names to importance scores
    """
    # Convert DataFrame to numpy if needed
    if isinstance(rss_data, pd.DataFrame):
        if ap_columns is None:
            ap_columns = rss_data.columns.tolist()
        rss_array = rss_data.to_numpy()
    else:
        rss_array = rss_data
        if ap_columns is None:
            ap_columns = [f"AP_{i}" for i in range(rss_array.shape[1])]
    
    n_aps = rss_array.shape[1]
    importance = np.zeros(n_aps)
    
    for ap_idx in range(n_aps):
        rss_values = rss_array[:, ap_idx]
        # Remove NaN or missing values
        rss_values = rss_values[~np.isnan(rss_values)]
        
        if len(rss_values) == 0:
            importance[ap_idx] = 0
            continue
        
        # Create histogram to estimate probability distribution
        hist, bin_edges = np.histogram(rss_values, bins='auto', density=True)
        bin_width = bin_edges[1] - bin_edges[0]
        
        # Normalize to get probabilities
        probabilities = hist * bin_width
        probabilities = probabilities[probabilities > 0]
        
        # Calculate Shannon entropy
        entropy = -np.sum(probabilities * np.log2(probabilities))
        importance[ap_idx] = entropy
    
    # Create dictionary mapping
    importance_dict = dict(zip(ap_columns, importance))
    
    return importance, importance_dict


def average_importance(rss_data, ap_columns):
    """
    Calculate importance using average RSS for each AP
    
    Parameters:
    rss_data: pandas DataFrame or numpy array of shape (n_samples, n_aps)
    ap_columns: list of AP column names (required if rss_data is DataFrame)
    
    Returns:
    importance_array: numpy array of importance scores for each AP
    importance_dict: dictionary mapping AP names to importance scores
    """
    # Convert DataFrame to numpy if needed
    if isinstance(rss_data, pd.DataFrame):
        if ap_columns is None:
            ap_columns = rss_data.columns.tolist()
        rss_array = rss_data.to_numpy()
    else:
        rss_array = rss_data
        if ap_columns is None:
            ap_columns = [f"AP_{i}" for i in range(rss_array.shape[1])]
    
    # Use nanmean to handle missing values
    importance = np.nanmean(rss_array, axis=0)
    
    # Create dictionary mapping
    importance_dict = dict(zip(ap_columns, importance))
    
    return importance, importance_dict


def median_importance(rss_data, ap_columns):
    """
    Calculate importance using median RSS for each AP
    
    Parameters:
    rss_data: pandas DataFrame or numpy array of shape (n_samples, n_aps)
    ap_columns: list of AP column names (required if rss_data is DataFrame)
    
    Returns:
    importance_array: numpy array of importance scores for each AP
    importance_dict: dictionary mapping AP names to importance scores
    """
    # Convert DataFrame to numpy if needed
    if isinstance(rss_data, pd.DataFrame):
        if ap_columns is None:
            ap_columns = rss_data.columns.tolist()
        rss_array = rss_data.to_numpy()
    else:
        rss_array = rss_data
        if ap_columns is None:
            ap_columns = [f"AP_{i}" for i in range(rss_array.shape[1])]
    
    # Use nanmedian to handle missing values
    importance = np.nanmedian(rss_array, axis=0)
    
    # Create dictionary mapping
    importance_dict = dict(zip(ap_columns, importance))
    
    return importance, importance_dict


def max_importance(rss_data, ap_columns):
    """
    Calculate importance using maximum RSS for each AP
    
    Parameters:
    rss_data: pandas DataFrame or numpy array of shape (n_samples, n_aps)
    ap_columns: list of AP column names (required if rss_data is DataFrame)
    
    Returns:
    importance_array: numpy array of importance scores for each AP
    importance_dict: dictionary mapping AP names to importance scores
    """
    # Convert DataFrame to numpy if needed
    if isinstance(rss_data, pd.DataFrame):
        if ap_columns is None:
            ap_columns = rss_data.columns.tolist()
        rss_array = rss_data.to_numpy()
    else:
        rss_array = rss_data
        if ap_columns is None:
            ap_columns = [f"AP_{i}" for i in range(rss_array.shape[1])]
    
    # Use nanmax to handle missing values
    importance = np.nanmax(rss_array, axis=0)
    
    # Create dictionary mapping
    importance_dict = dict(zip(ap_columns, importance))
    
    return importance, importance_dict



def variance_importance(rss_data, ap_columns):
    """
    Calculate importance using variance of RSS for each AP
    
    Higher variance indicates the AP has more diverse signal strength readings,
    which can be useful for distinguishing between different locations.
    
    Parameters:
    rss_data: pandas DataFrame or numpy array of shape (n_samples, n_aps)
    ap_columns: list of AP column names (required if rss_data is DataFrame)
    
    Returns:
    importance_array: numpy array of importance scores for each AP
    importance_dict: dictionary mapping AP names to importance scores
    """
    # Convert DataFrame to numpy if needed
    if isinstance(rss_data, pd.DataFrame):
        if ap_columns is None:
            ap_columns = rss_data.columns.tolist()
        rss_array = rss_data.to_numpy()
    else:
        rss_array = rss_data
        if ap_columns is None:
            ap_columns = [f"AP_{i}" for i in range(rss_array.shape[1])]
    
    # Use nanvar to handle missing values
    # ddof=1 for sample variance (unbiased estimator)
    importance = np.nanvar(rss_array, axis=0, ddof=1)
    
    # Create dictionary mapping
    importance_dict = dict(zip(ap_columns, importance))
    
    return importance, importance_dict


def mutual_information_importance(rssi_data, coordinates_3d):
    """
    Calculate AP importance using weighted mutual information for 3D targets

    Parameters:
    rssi_data: pandas DataFrame of shape (n_samples, n_aps) with AP columns
    coordinates_3d: numpy array of shape (n_samples, 3) with [lat, lon, floor]

    Returns:
    importance_array: numpy array of importance scores for each AP
    importance_dict: dictionary mapping AP names to importance scores
    """
    print("Calculating weighted mutual information importance scores...")

    importance_scores = {}
    ap_columns = rssi_data.columns.tolist()

    for i, ap in enumerate(ap_columns):
        ap_rssi = rssi_data[ap].values.reshape(-1, 1)

        if np.var(ap_rssi) == 0:
            importance_scores[ap] = 0.0
            continue

        # Calculate mutual information for each coordinate dimension
        mi_lat = mutual_info_regression(ap_rssi, coordinates_3d[:, 0], random_state=42)[0]
        mi_lon = mutual_info_regression(ap_rssi, coordinates_3d[:, 1], random_state=42)[0]
        mi_floor = mutual_info_regression(ap_rssi, coordinates_3d[:, 2], random_state=42)[0]

        # Weighted combination - give floor 3x importance for better vertical localization
        total_importance = (mi_lat + mi_lon + 3*mi_floor) / 5.0
        importance_scores[ap] = total_importance

    nonzero_importance = {ap: score for ap, score in importance_scores.items() if score > 0}
    print(f"APs with non-zero importance: {len(nonzero_importance)}/{len(importance_scores)}")

    print('Done')

    # Convert dictionary to array maintaining order of ap_columns
    importance_array = np.array([importance_scores[ap] for ap in ap_columns])

    return importance_array, importance_scores