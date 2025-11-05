def create_3d_coordinates(building_data, floor_height, fit_scaler=True):
    """
    Create normalized 3D coordinates (lat, lon, floor_z)

    Parameters:
    building_data: DataFrame with LATITUDE, LONGITUDE, FLOOR columns
    floor_height: Height difference between floors
    fit_scaler: Whether to fit the scaler (True for training data)
    """
    global scaler_coords

    # Convert floor to Z-height
    floor_z = building_data['FLOOR'] * floor_height

    # Combine coordinates
    coordinates = np.column_stack([
        building_data['LATITUDE'].values,
        building_data['LONGITUDE'].values,
        floor_z.values
    ])

    # Normalize coordinates
    if fit_scaler:
        normalized_coords = scaler_coords.fit_transform(coordinates)
    else:
        normalized_coords = scaler_coords.transform(coordinates)

    return normalized_coords

