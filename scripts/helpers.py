from pathlib import Path
import pandas as pd

def save_to_excel(df, folder_path, filename, sheet_name='Sheet1', overwrite=True):
    """
    Save a pandas DataFrame to an Excel file, updating sheets if file exists.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to save
    folder_path : str
        The path to the folder where the file should be saved
    filename : str
        The name for the Excel file (with or without .xlsx extension)
    sheet_name : str, optional (default='Sheet1')
        The name of the sheet to save/update
    overwrite : bool, optional (default=True)
        If True, overwrites/updates existing sheets
    """
    try:
        # Create folder if it doesn't exist
        folder = Path(folder_path)
        folder.mkdir(parents=True, exist_ok=True)
        
        # Ensure filename has .xlsx extension
        if not filename.lower().endswith(('.xlsx', '.xls')):
            filename += '.xlsx'
            
        # Construct full file path
        file_path = folder / filename
        
        # If file exists and overwrite is True, append/update mode
        if file_path.exists() and overwrite:
            # Load existing workbook
            with pd.ExcelWriter(file_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
                df.to_excel(writer, sheet_name=sheet_name, index=True)
            print(f"Sheet '{sheet_name}' updated in {file_path}")
            
        # If file doesn't exist or overwrite is False
        else:
            if file_path.exists() and not overwrite:
                raise ValueError(f"File {file_path} already exists and overwrite=False")
            
            # Create new file
            df.to_excel(file_path, sheet_name=sheet_name, index=True)
            print(f"New file created at {file_path}")
        
        return str(file_path)
        
    except Exception as e:
        print(f"Error saving DataFrame to Excel: {str(e)}")
        raise


def remove_selected_aps(results_dict):
    """
    Remove 'selected_aps' from results dictionary, keeping only performance metrics

    Parameters:
    -----------
    results_dict : dict
        Dictionary containing method results with 'selected_aps' and metrics

    Returns:
    --------
    cleaned_dict : dict
        Dictionary with only performance metrics (no selected_aps)

    Example:
    --------
    >>> results = {
    ...     'entropy': {
    ...         'selected_aps': ['WAP001', 'WAP002'],
    ...         'mean_3d_error': 15.5,
    ...         'floor_accuracy': 0.85
    ...     }
    ... }
    >>> cleaned = remove_selected_aps(results)
    >>> print(cleaned['entropy'])
    {'mean_3d_error': 15.5, 'floor_accuracy': 0.85}
    """
    cleaned_dict = {}

    for method, results in results_dict.items():
        cleaned_dict[method] = {k: v for k, v in results.items() if k != 'selected_aps'}

    return cleaned_dict