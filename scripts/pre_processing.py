import pandas as pd
import numpy as np

def find_feature_min_max(df, columns):
    """
    Find the min and max for specified DataFrame columns.
    Returns: dict of (min, max) tuples per column
    """
    return {col: (df[col].min(), df[col].max()) for col in columns}

def normalize_rssi(df_rssi, min_rssi, max_rssi):
    """
    Normalize RSSI to [0, 1] range. NaN (non-detections) become 0.
    """
    df_norm = (df_rssi.fillna(min_rssi) - min_rssi) / (max_rssi - min_rssi)
    df_norm[df_rssi.isna()] = 0  # Set true non-detections to 0
    return df_norm

def normalize_col(series, min_val, max_val):
    """
    Normalize one pandas Series to [0, 1] range.
    """
    return (series - min_val) / (max_val - min_val)

def load_and_preprocess_data(df_train_path, df_validation_path, building_id, floor_height=3.0):
    """
    Main function to load, filter, and normalize UJIIndoorLoc datasets.
    
    Parameters:
    df_train_path: Path to training CSV
    df_validation_path: Path to validation CSV
    building_id: Building ID to filter (0, 1, or 2)
    floor_height: Height of each floor in meters (default: 4.0)
    
    Returns:
    rssi_train_norm, coords_train, rssi_val_norm, coords_val, ap_columns
    """
    print("Loading and preprocessing UJIIndoorLoc training and validation datasets...")

    # Load
    df_train = pd.read_csv(df_train_path)
    df_val = pd.read_csv(df_validation_path)

    print(f"Training samples: {len(df_train)}")
    print(f"Validation samples: {len(df_val)}")

    # Filter for building
    train = df_train[df_train['BUILDINGID'] == building_id].copy()
    val = df_val[df_val['BUILDINGID'] == building_id].copy()

    ap_columns = [col for col in df_train.columns if col.startswith('WAP')]
    lat_col, lon_col = 'LATITUDE', 'LONGITUDE'

    # Prep RSSI: replace 100 with NaN
    rssi_train = train[ap_columns].replace(100, np.nan)
    rssi_val   = val[ap_columns].replace(100, np.nan)

    # --- Normalize RSSI ---
    min_rssi = pd.concat([rssi_train, rssi_val]).min().min()
    max_rssi = pd.concat([rssi_train, rssi_val]).max().max()
    rssi_train_norm = normalize_rssi(rssi_train, min_rssi, max_rssi)
    rssi_val_norm   = normalize_rssi(rssi_val, min_rssi, max_rssi)

    # --- Normalize Lat / Long ---
    lat_min = pd.concat([train[[lat_col]], val[[lat_col]]])[lat_col].min()
    lat_max = pd.concat([train[[lat_col]], val[[lat_col]]])[lat_col].max()
    lon_min = pd.concat([train[[lon_col]], val[[lon_col]]])[lon_col].min()
    lon_max = pd.concat([train[[lon_col]], val[[lon_col]]])[lon_col].max()

    train['LAT_NORM'] = normalize_col(train[lat_col], lat_min, lat_max)
    val['LAT_NORM']   = normalize_col(val[lat_col], lat_min, lat_max)
    train['LON_NORM'] = normalize_col(train[lon_col], lon_min, lon_max)
    val['LON_NORM']   = normalize_col(val[lon_col], lon_min, lon_max)

    # Create 3D coordinates (LON_NORM, LAT_NORM, FLOOR)
    coords_train = train[['LON_NORM', 'LAT_NORM', 'FLOOR']].values
    coords_val   = val[['LON_NORM', 'LAT_NORM', 'FLOOR']].values

    print(f"Training coordinate ranges:")
    print(f"  Longitude: [{train[lon_col].min():.6f}, {train[lon_col].max():.6f}]")
    print(f"  Latitude: [{train[lat_col].min():.6f}, {train[lat_col].max():.6f}]")
    print(f"  Floors: {sorted(train['FLOOR'].unique())}")
    print(f"  Floor height: {floor_height} meters")

    # Save for denormalization and floor height globally
    global LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, FLOOR_HEIGHT
    LAT_MIN, LAT_MAX = lat_min, lat_max
    LON_MIN, LON_MAX = lon_min, lon_max
    FLOOR_HEIGHT = floor_height

    return rssi_train_norm, coords_train, rssi_val_norm, coords_val, ap_columns


def denormalize_col(series_norm, min_val, max_val):
    """
    Convert normalized series [0,1] back to original scale.
    """
    return series_norm * (max_val - min_val) + min_val


def denormalize_lat(norm_val):
    return denormalize_col(norm_val, LAT_MIN, LAT_MAX)

def denormalize_lon(norm_val):
    return denormalize_col(norm_val, LON_MIN, LON_MAX)


