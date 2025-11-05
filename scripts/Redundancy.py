

def calculate_redundancy_matrix(rssi_data):
    """
    Calculate pairwise redundancy (Pearson correlation) between APs
    
    Parameters:
    rssi_data: DataFrame with RSSI values
    
    Returns:
    redundancy_matrix: Correlation matrix for all APs
    """
    print("Calculating redundancy matrix...")
    
    # Calculate correlation matrix for all APs
    redundancy_matrix = rssi_data.corr().abs()
    
    print('Done')
    
    return redundancy_matrix

