"""
Utility functions for loading pre-computed importance scores and redundancy matrices.

This module provides convenient functions to load saved importance dictionaries
and redundancy matrices from CSV/Excel files, avoiding expensive re-computation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle


def load_importance_dict_from_csv(filepath):
    """
    Load importance dictionary from CSV file

    Parameters:
    -----------
    filepath : str or Path
        Path to CSV file with columns ['AP', 'Score']

    Returns:
    --------
    dict : {AP_name: score}
        Dictionary mapping AP names to importance scores
    """
    df = pd.read_csv(filepath)
    return dict(zip(df['AP'], df['Score']))


def load_redundancy_matrix_from_csv(filepath):
    """
    Load redundancy matrix from CSV file

    Parameters:
    -----------
    filepath : str or Path
        Path to CSV file with redundancy matrix

    Returns:
    --------
    DataFrame : redundancy matrix with AP names as index and columns
    """
    return pd.read_csv(filepath, index_col=0)


def load_all_importance_scores(base_dir='data/output_data/importance_scores'):
    """
    Load all importance score dictionaries from the default directory

    Parameters:
    -----------
    base_dir : str or Path, optional
        Base directory containing importance score CSV files
        Default: 'data/output_data/importance_scores'

    Returns:
    --------
    dict : Dictionary containing all importance methods
        {
            'entropy': {AP: score, ...},
            'average': {AP: score, ...},
            'median': {AP: score, ...},
            'max': {AP: score, ...},
            'variance': {AP: score, ...}
        }
    """
    importance_dir = Path(base_dir)

    importance_methods = ['entropy', 'average', 'median', 'max', 'variance', 'mutual_info']
    importance_dicts = {}

    for method in importance_methods:
        filepath = importance_dir / f'{method}_importance_dict.csv'
        if filepath.exists():
            importance_dicts[method] = load_importance_dict_from_csv(filepath)
            print(f"✓ Loaded {len(importance_dicts[method])} APs for {method} importance")
        else:
            print(f"⚠ Warning: {filepath} not found, skipping {method}")

    return importance_dicts


def load_redundancy_matrix(base_dir='data/output_data/redundancy_scores'):
    """
    Load redundancy matrix from the default directory

    Parameters:
    -----------
    base_dir : str or Path, optional
        Base directory containing redundancy matrix CSV file
        Default: 'data/output_data/redundancy_scores'

    Returns:
    --------
    DataFrame : redundancy matrix
    """
    redundancy_dir = Path(base_dir)
    filepath = redundancy_dir / 'redundancy_matrix.csv'

    if not filepath.exists():
        raise FileNotFoundError(
            f"Redundancy matrix not found at {filepath}. "
            "Please run the redundancy calculation first."
        )

    matrix = load_redundancy_matrix_from_csv(filepath)
    print(f"✓ Loaded redundancy matrix with shape: {matrix.shape}")
    return matrix


def load_all_precomputed_data(importance_dir='data/output_data/importance_scores',
                                redundancy_dir='data/output_data/redundancy_scores'):
    """
    Convenience function to load all pre-computed importance scores and redundancy matrix

    Parameters:
    -----------
    importance_dir : str or Path, optional
        Directory containing importance score CSV files
    redundancy_dir : str or Path, optional
        Directory containing redundancy matrix CSV file

    Returns:
    --------
    tuple : (importance_dicts, redundancy_matrix)
        - importance_dicts: dict of {method: {AP: score}}
        - redundancy_matrix: DataFrame

    Example:
    --------
    >>> importance_dicts, redundancy_matrix = load_all_precomputed_data()
    >>> Q, relevant_aps, offset = formulate_qubo(
    ...     importance_dicts['entropy'],
    ...     redundancy_matrix,
    ...     k=20, alpha=0.9, penalty=2.0
    ... )
    """
    print("="*60)
    print("Loading pre-computed importance scores and redundancy matrix")
    print("="*60)

    print("\nLoading importance scores...")
    importance_dicts = load_all_importance_scores(importance_dir)

    print("\nLoading redundancy matrix...")
    redundancy_matrix = load_redundancy_matrix(redundancy_dir)

    print("\n✓ All data loaded successfully!")
    print("="*60)

    return importance_dicts, redundancy_matrix


# Convenience function for saving importance dictionaries
def save_importance_dict_to_csv(importance_dict, filepath):
    """
    Save importance dictionary to CSV file

    Parameters:
    -----------
    importance_dict : dict
        Dictionary mapping AP names to importance scores
    filepath : str or Path
        Output CSV file path
    """
    df = pd.DataFrame(list(importance_dict.items()), columns=['AP', 'Score'])
    df.to_csv(filepath, index=False)
    print(f"✓ Saved importance dictionary to {filepath}")


def save_all_importance_dicts(importance_dicts, output_dir='data/output_data/importance_scores'):
    """
    Save all importance dictionaries to CSV files

    Parameters:
    -----------
    importance_dicts : dict
        Dictionary of {method_name: {AP: score}}
    output_dir : str or Path, optional
        Output directory for CSV files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for method, imp_dict in importance_dicts.items():
        filepath = output_path / f'{method}_importance_dict.csv'
        save_importance_dict_to_csv(imp_dict, filepath)

    print(f"\n✓ All importance dictionaries saved to {output_dir}")


def save_preprocessed_data(rssi_train, coords_train, rssi_val, coords_val, ap_columns,
                           building_id, output_dir='data/output_data/preprocessed_data'):
    """
    Save preprocessed data to files for later loading

    Parameters:
    -----------
    rssi_train : numpy.ndarray
        Training RSSI data (normalized)
    coords_train : numpy.ndarray
        Training coordinates (normalized)
    rssi_val : numpy.ndarray
        Validation RSSI data (normalized)
    coords_val : numpy.ndarray
        Validation coordinates (normalized)
    ap_columns : list
        List of AP column names
    building_id : int
        Building ID
    output_dir : str or Path, optional
        Output directory for saved files
        Default: 'data/output_data/preprocessed_data'

    Returns:
    --------
    dict : Paths to saved files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create a pickle file for numpy arrays (faster and preserves data types)
    data_dict = {
        'rssi_train': rssi_train,
        'coords_train': coords_train,
        'rssi_val': rssi_val,
        'coords_val': coords_val,
        'ap_columns': ap_columns,
        'building_id': building_id
    }

    pickle_path = output_path / f'preprocessed_building_{building_id}.pkl'
    with open(pickle_path, 'wb') as f:
        pickle.dump(data_dict, f)

    print(f"✓ Saved preprocessed data to pickle: {pickle_path}")

    # Also save as Excel for human readability (optional, slower)
    # Save training data
    train_df = pd.DataFrame(rssi_train, columns=ap_columns)
    train_df['LON_NORM'] = coords_train[:, 0]
    train_df['LAT_NORM'] = coords_train[:, 1]
    train_df['FLOOR'] = coords_train[:, 2]

    # Save validation data
    val_df = pd.DataFrame(rssi_val, columns=ap_columns)
    val_df['LON_NORM'] = coords_val[:, 0]
    val_df['LAT_NORM'] = coords_val[:, 1]
    val_df['FLOOR'] = coords_val[:, 2]

    excel_path = output_path / f'preprocessed_building_{building_id}.xlsx'
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        train_df.to_excel(writer, sheet_name='Training', index=False)
        val_df.to_excel(writer, sheet_name='Validation', index=False)
        pd.DataFrame({'AP': ap_columns}).to_excel(writer, sheet_name='AP_Columns', index=False)

    print(f"✓ Saved preprocessed data to Excel: {excel_path}")

    return {
        'pickle': str(pickle_path),
        'excel': str(excel_path)
    }


def load_preprocessed_data(building_id, input_dir='data/output_data/preprocessed_data',
                           use_pickle=True):
    """
    Load preprocessed data from saved files

    Parameters:
    -----------
    building_id : int
        Building ID
    input_dir : str or Path, optional
        Input directory containing saved files
        Default: 'data/output_data/preprocessed_data'
    use_pickle : bool, optional
        If True, load from pickle (faster). If False, load from Excel.
        Default: True

    Returns:
    --------
    tuple : (rssi_train, coords_train, rssi_val, coords_val, ap_columns)
        - rssi_train: Training RSSI array (n_samples, n_aps)
        - coords_train: Training coordinates (n_samples, 3) [LON_NORM, LAT_NORM, FLOOR]
        - rssi_val: Validation RSSI array
        - coords_val: Validation coordinates
        - ap_columns: List of AP names

    Example:
    --------
    >>> # Load preprocessed data for building 1
    >>> rssi_train, coords_train, rssi_val, coords_val, ap_columns = load_preprocessed_data(building_id=1)
    """
    input_path = Path(input_dir)

    if use_pickle:
        # Load from pickle (faster)
        pickle_path = input_path / f'preprocessed_building_{building_id}.pkl'

        if not pickle_path.exists():
            raise FileNotFoundError(
                f"Preprocessed data not found at {pickle_path}. "
                f"Please run load_and_preprocess_data() and save_preprocessed_data() first."
            )

        with open(pickle_path, 'rb') as f:
            data_dict = pickle.load(f)

        print(f"✓ Loaded preprocessed data from pickle: {pickle_path}")
        print(f"  Training samples: {data_dict['rssi_train'].shape[0]}")
        print(f"  Validation samples: {data_dict['rssi_val'].shape[0]}")
        print(f"  Number of APs: {len(data_dict['ap_columns'])}")

        return (
            data_dict['rssi_train'],
            data_dict['coords_train'],
            data_dict['rssi_val'],
            data_dict['coords_val'],
            data_dict['ap_columns']
        )

    else:
        # Load from Excel (slower, human-readable)
        excel_path = input_path / f'preprocessed_building_{building_id}.xlsx'

        if not excel_path.exists():
            raise FileNotFoundError(
                f"Preprocessed data not found at {excel_path}. "
                f"Please run load_and_preprocess_data() and save_preprocessed_data() first."
            )

        # Read Excel sheets
        train_df = pd.read_excel(excel_path, sheet_name='Training')
        val_df = pd.read_excel(excel_path, sheet_name='Validation')
        ap_df = pd.read_excel(excel_path, sheet_name='AP_Columns')

        ap_columns = ap_df['AP'].tolist()

        # Extract RSSI and coordinates
        rssi_train = train_df[ap_columns].values
        coords_train = train_df[['LON_NORM', 'LAT_NORM', 'FLOOR']].values

        rssi_val = val_df[ap_columns].values
        coords_val = val_df[['LON_NORM', 'LAT_NORM', 'FLOOR']].values

        print(f"✓ Loaded preprocessed data from Excel: {excel_path}")
        print(f"  Training samples: {rssi_train.shape[0]}")
        print(f"  Validation samples: {rssi_val.shape[0]}")
        print(f"  Number of APs: {len(ap_columns)}")

        return rssi_train, coords_train, rssi_val, coords_val, ap_columns
