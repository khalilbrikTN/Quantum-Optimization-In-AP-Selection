import numpy as np
import pandas as pd
from scripts.data.pre_processing import denormalize_col

def evaluate_and_print_results(solver, coords_train, coords_val,
                               predictions_openjij, selected_aps_openjij,
                               predictions_SA, selected_aps_SA, total_aps):
    """
    Evaluate and print results for a specific solver

    Parameters:
    solver: str - Either 'openjij' or 'SA'
    coords_train: Training coordinates
    coords_val: Validation coordinates
    predictions_openjij: OpenJij predictions dictionary
    selected_aps_openjij: OpenJij selected APs list
    predictions_SA: SA predictions dictionary
    selected_aps_SA: SA selected APs list
    total_aps: Total number of APs

    Returns:
    results: Dictionary containing evaluation results
    """
    if solver not in ['openjij', 'SA']:
        print(f"Error: Unknown solver '{solver}'. Please use 'openjij' or 'SA'.")
        return None

    print(f"Results {solver}")

    if solver == 'openjij':
        results = evaluate_comprehensive_performance(
            coords_train, coords_val, predictions_openjij, selected_aps_openjij, total_aps
        )
    else:  # solver == 'SA'
        results = evaluate_comprehensive_performance(
            coords_train, coords_val, predictions_SA, selected_aps_SA, total_aps
        )

    print(f"Results created for: {list(results.keys())}")
    for key, value in results.items():
        print(f"{key}: has {len(value.get('real_errors', []))} error values")

    return results

def calculate_3d_distance_error(y_true, y_pred):
    """
    Calculate 3D Euclidean distance error

    Parameters:
    y_true: True coordinates
    y_pred: Predicted coordinates

    Returns:
    distance_errors: Array of individual distance errors
    metrics: Dictionary of error metrics
    """
    distance_errors = np.sqrt(np.sum((y_true - y_pred)**2, axis=1))

    metrics = {
        'mean_error': np.mean(distance_errors),
        'median_error': np.median(distance_errors),
        'std_error': np.std(distance_errors),
        'percentile_90': np.percentile(distance_errors, 90),
        'percentile_95': np.percentile(distance_errors, 95),
        'min_error': np.min(distance_errors),
        'max_error': np.max(distance_errors)
    }

    return distance_errors, metrics


def calculate_comprehensive_metrics(y_true, y_pred, 
                                   lon_min, lon_max, 
                                   lat_min, lat_max, 
                                   floor_height):
    """
    Calculate both normalized and corrected real-world metrics
    """
    # Normalized metrics
    norm_errors = np.sqrt(np.sum((y_true - y_pred)**2, axis=1))

    # Corrected real-world metrics
    real_errors, floor_accuracy = denormalize_and_calculate_real_distance(
        y_true, y_pred, lon_min, lon_max, lat_min, lat_max, floor_height
    )

    metrics = {
        # Normalized metrics
        'norm_mean': np.mean(norm_errors),
        'norm_median': np.median(norm_errors),
        'norm_90th': np.percentile(norm_errors, 90),
        'norm_95th': np.percentile(norm_errors, 95),

        # Real-world metrics (corrected)
        'real_mean_m': np.mean(real_errors),
        'real_median_m': np.median(real_errors),
        'real_90th_m': np.percentile(real_errors, 90),
        'real_95th_m': np.percentile(real_errors, 95),
        'floor_accuracy': floor_accuracy,

        # Additional stats
        'real_min_m': np.min(real_errors),
        'real_max_m': np.max(real_errors)
    }

    return norm_errors, real_errors, metrics



def evaluate_comprehensive_performance(coords_train, coords_val, predictions, selected_aps, total_aps):
    """
    Re-evaluation with corrected distance calculations
    """
    print("\n" + "="*80)
    print("CORRECTED PERFORMANCE EVALUATION")
    print("="*80)

    results = {}

    # Decision Tree Results
    if 'dt_train' in predictions and 'dt_val' in predictions:
        print("\nDECISION TREE RESULTS:")
        print("-" * 50)

        # Training metrics
        dt_norm_train, dt_real_train, dt_train_metrics = calculate_comprehensive_metrics(
            coords_train, predictions['dt_train']
        )

        # Validation metrics
        dt_norm_val, dt_real_val, dt_val_metrics = calculate_comprehensive_metrics(
            coords_val, predictions['dt_val']
        )

        # Display results
        print(f"Training - Normalized: {dt_train_metrics['norm_mean']:.4f}, Real: {dt_train_metrics['real_mean_m']:.2f}m")
        print(f"Validation - Normalized: {dt_val_metrics['norm_mean']:.4f}, Real: {dt_val_metrics['real_mean_m']:.2f}m")
        print(f"Floor Accuracy: {dt_val_metrics['floor_accuracy']:.1%}")
        print(f"90th Percentile: {dt_val_metrics['real_90th_m']:.2f}m")

        overfitting_ratio = dt_val_metrics['norm_mean'] / dt_train_metrics['norm_mean']
        print(f"Overfitting Ratio: {overfitting_ratio:.2f}")

        results['decision_tree'] = {
            'val_metrics': dt_val_metrics,
            'train_metrics': dt_train_metrics,
            'overfitting_ratio': overfitting_ratio,
            'real_errors': dt_real_val
        }

    # Random Forest Results
    if 'rf_train' in predictions and 'rf_val' in predictions:
        print("\nRANDOM FOREST RESULTS:")
        print("-" * 50)

        # Training metrics
        rf_norm_train, rf_real_train, rf_train_metrics = calculate_comprehensive_metrics(
            coords_train, predictions['rf_train']
        )

        # Validation metrics
        rf_norm_val, rf_real_val, rf_val_metrics = calculate_comprehensive_metrics(
            coords_val, predictions['rf_val']
        )

        # Display results
        print(f"Training - Normalized: {rf_train_metrics['norm_mean']:.4f}, Real: {rf_train_metrics['real_mean_m']:.2f}m")
        print(f"Validation - Normalized: {rf_val_metrics['norm_mean']:.4f}, Real: {rf_val_metrics['real_mean_m']:.2f}m")
        print(f"Floor Accuracy: {rf_val_metrics['floor_accuracy']:.1%}")
        print(f"90th Percentile: {rf_val_metrics['real_90th_m']:.2f}m")

        overfitting_ratio = rf_val_metrics['norm_mean'] / rf_train_metrics['norm_mean']
        print(f"Overfitting Ratio: {overfitting_ratio:.2f}")

        results['random_forest'] = {
            'val_metrics': rf_val_metrics,
            'train_metrics': rf_train_metrics,
            'overfitting_ratio': overfitting_ratio,
            'real_errors': rf_real_val
        }

    print(f"\nSelected APs: {len(selected_aps)}/{total_aps} ({(1-len(selected_aps)/total_aps)*100:.1f}% reduction)")

    return results


def denormalize_and_calculate_real_distance(y_true_norm, y_pred_norm, 
                                            lon_min, lon_max, 
                                            lat_min, lat_max, 
                                            floor_height):
    """
    Corrected real-world distance calculation using manual denormalization
    """
    # Denormalize longitude and latitude
    lon_true = denormalize_col(y_true_norm[:, 0], lon_min, lon_max)
    lat_true = denormalize_col(y_true_norm[:, 1], lat_min, lat_max)
    
    lon_pred = denormalize_col(y_pred_norm[:, 0], lon_min, lon_max)
    lat_pred = denormalize_col(y_pred_norm[:, 1], lat_min, lat_max)
    
    # Floor is NOT normalized
    floor_true = y_true_norm[:, 2]
    floor_pred = y_pred_norm[:, 2]
    
    # Calculate 2D horizontal distance (UTM coordinates in meters)
    lon_diff_m = lon_true - lon_pred
    lat_diff_m = lat_true - lat_pred
    horizontal_distance = np.sqrt(lat_diff_m**2 + lon_diff_m**2)
    
    # Vertical distance using floor height
    floor_diff = floor_true - floor_pred
    vertical_distance_m = np.abs(floor_diff) * floor_height
    
    # 3D Euclidean distance
    real_distance_errors = np.sqrt(horizontal_distance**2 + vertical_distance_m**2)
    
    # Floor accuracy
    floor_predicted_rounded = np.round(floor_pred).astype(int)
    floor_true_rounded = np.round(floor_true).astype(int)
    floor_accuracy = np.mean(floor_predicted_rounded == floor_true_rounded)
    
    return real_distance_errors, floor_accuracy





def save_master_summary(output_folder, all_results, timestamp, alpha, penalty):
    """
    Save master summary file with all results.

    Parameters:
    -----------
    output_folder : str
        Path to output folder
    all_results : dict
        Complete results dictionary
    timestamp : str
        Experiment timestamp
    alpha : float
        Alpha parameter used
    penalty : float
        Penalty parameter used
    """
    filename = f"Complete_Experiment_Results_{timestamp}.xlsx"
    filepath = os.path.join(output_folder, filename)

    try:
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:

            # Sheet 1: Summary Table
            summary_rows = []
            budgets = sorted([k for k in all_results.keys() if isinstance(k, int)])

            for k in budgets:
                if 'error' in all_results[k]:
                    continue

                for solver in ['openjij', 'SA']:
                    if 'random_forest' in all_results[k][solver]:
                        rf_metrics = all_results[k][solver]['random_forest']['val_metrics']
                        summary_rows.append({
                            'Budget': k,
                            'Solver': solver,
                            'Mean_Error_m': rf_metrics.get('real_mean_m', None),
                            'Median_Error_m': rf_metrics.get('real_median_m', None),
                            '90th_Percentile_m': rf_metrics.get('real_90th_m', None),
                            '95th_Percentile_m': rf_metrics.get('real_95th_m', None),
                            'Floor_Accuracy': rf_metrics.get('floor_accuracy', None),
                            'Computation_Time_s': all_results[k][solver].get('duration', None),
                            'Num_APs_Selected': all_results[k][solver].get('num_selected', None)
                        })

            df_summary = pd.DataFrame(summary_rows)
            df_summary.to_excel(writer, sheet_name='Summary_Table', index=False)

            # Sheet 2: CDF Data
            cdf_rows = []

            for k in budgets:
                if 'error' in all_results[k]:
                    continue

                for solver in ['openjij', 'SA']:
                    if 'random_forest' in all_results[k][solver]:
                        errors = all_results[k][solver]['random_forest'].get('real_errors', [])
                        if len(errors) > 0:
                            # Sort errors
                            sorted_errors = np.sort(errors)
                            n_samples = len(sorted_errors)

                            # Calculate cumulative percentages
                            cumulative_pct = np.arange(1, n_samples + 1) / n_samples * 100

                            # Add to CDF data
                            for error, pct in zip(sorted_errors, cumulative_pct):
                                cdf_rows.append({
                                    'Budget': k,
                                    'Solver': solver,
                                    'Error_Meters': error,
                                    'Cumulative_Percentage': pct
                                })

            df_cdf = pd.DataFrame(cdf_rows)
            df_cdf.to_excel(writer, sheet_name='CDF_Data', index=False)

            # Sheet 3: Timing Comparison
            timing_rows = []

            for k in budgets:
                if 'error' in all_results[k]:
                    continue

                openjij_time = all_results[k]['openjij'].get('duration', None)
                sa_time = all_results[k]['SA'].get('duration', None)

                if openjij_time is not None and sa_time is not None:
                    timing_rows.append({
                        'Budget': k,
                        'OpenJij_Time_s': openjij_time,
                        'SA_Time_s': sa_time,
                        'Time_Difference_s': sa_time - openjij_time,
                        'Faster_Solver': 'OpenJij' if openjij_time < sa_time else 'SA'
                    })

            df_timing = pd.DataFrame(timing_rows)
            df_timing.to_excel(writer, sheet_name='Timing_Comparison', index=False)

            # Sheet 4: Experiment Parameters
            params_data = {
                'Parameter': ['Alpha', 'Penalty', 'Experiment_Timestamp', 'Total_Budgets_Tested'],
                'Value': [alpha, penalty, timestamp, len(budgets)]
            }
            df_params = pd.DataFrame(params_data)
            df_params.to_excel(writer, sheet_name='Experiment_Parameters', index=False)

        print(f"\n✓ Master summary saved: {filename}")

    except Exception as e:
        print(f"\n✗ Failed to save master summary: {e}")

def save_individual_budget_results(output_folder, budget, solver, result_data, timestamp):
    """
    Save individual budget results to Excel file.

    Parameters:
    -----------
    output_folder : str
        Path to output folder
    budget : int
        Budget value (k)
    solver : str
        'openjij' or 'SA'
    result_data : dict
        Results dictionary for this budget and solver
    timestamp : str
        Experiment timestamp
    """
    filename = f"Results_k{budget}_{solver}.xlsx"
    filepath = os.path.join(output_folder, filename)

    try:
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:

            # Sheet 1: Metadata
            metadata = {
                'Parameter': ['Budget_k', 'Solver', 'Num_APs_Selected', 'Total_APs_Available',
                             'Computation_Time_Seconds', 'Alpha', 'Penalty', 'Experiment_Timestamp'],
                'Value': [
                    budget,
                    solver,
                    result_data.get('num_selected', 'N/A'),
                    result_data.get('total_aps', 'N/A'),
                    result_data.get('duration', 'N/A'),
                    result_data.get('alpha', 'N/A'),
                    result_data.get('penalty', 'N/A'),
                    timestamp
                ]
            }
            df_metadata = pd.DataFrame(metadata)
            df_metadata.to_excel(writer, sheet_name='Metadata', index=False)

            # Sheet 2: Random Forest Metrics (if available)
            if 'random_forest' in result_data and 'val_metrics' in result_data['random_forest']:
                rf_metrics = result_data['random_forest']['val_metrics']
                metrics_data = {
                    'Metric': [
                        'Mean_Error_m',
                        'Median_Error_m',
                        '90th_Percentile_m',
                        '95th_Percentile_m',
                        'Min_Error_m',
                        'Max_Error_m',
                        'Floor_Accuracy',
                        'Normalized_Mean',
                        'Normalized_Median',
                        'Normalized_90th',
                        'Normalized_95th',
                        'Overfitting_Ratio'
                    ],
                    'Value': [
                        rf_metrics.get('real_mean_m', 'N/A'),
                        rf_metrics.get('real_median_m', 'N/A'),
                        rf_metrics.get('real_90th_m', 'N/A'),
                        rf_metrics.get('real_95th_m', 'N/A'),
                        rf_metrics.get('real_min_m', 'N/A'),
                        rf_metrics.get('real_max_m', 'N/A'),
                        rf_metrics.get('floor_accuracy', 'N/A'),
                        rf_metrics.get('norm_mean', 'N/A'),
                        rf_metrics.get('norm_median', 'N/A'),
                        rf_metrics.get('norm_90th', 'N/A'),
                        rf_metrics.get('norm_95th', 'N/A'),
                        result_data['random_forest'].get('overfitting_ratio', 'N/A')
                    ]
                }
                df_metrics = pd.DataFrame(metrics_data)
                df_metrics.to_excel(writer, sheet_name='Random_Forest_Metrics', index=False)

            # Sheet 3: Selected APs
            if 'selected_aps' in result_data and result_data['selected_aps']:
                df_aps = pd.DataFrame({'Selected_AP': result_data['selected_aps']})
                df_aps.to_excel(writer, sheet_name='Selected_APs', index=False)

            # Sheet 4: Validation Errors
            if 'random_forest' in result_data and 'real_errors' in result_data['random_forest']:
                errors = result_data['random_forest']['real_errors']
                df_errors = pd.DataFrame({'Error_Meters': errors})
                df_errors.to_excel(writer, sheet_name='Validation_Errors', index=False)

        print(f"  ✓ Saved: {filename}")

    except Exception as e:
        print(f"  ✗ Failed to save {filename}: {e}")


def extract_metrics_summary(all_results, metric_name='real_mean_m', model='random_forest'):
    """
    Extract a specific metric across all budgets for easy comparison.

    Parameters:
    -----------
    all_results : dict
        Results from run_multi_budget_experiment
    metric_name : str
        Metric to extract (e.g., 'real_mean_m', 'floor_accuracy', 'real_90th_m')
    model : str
        Model type ('decision_tree' or 'random_forest')

    Returns:
    --------
    summary : dict
        {
            'budgets': [10, 20, 30, ...],
            'openjij': [value1, value2, ...],
            'SA': [value1, value2, ...]
        }

    Example:
    --------
    >>> summary = extract_metrics_summary(all_results, 'real_mean_m', 'random_forest')
    >>> plt.plot(summary['budgets'], summary['openjij'], label='OpenJij')
    >>> plt.plot(summary['budgets'], summary['SA'], label='SA')
    >>> plt.xlabel('Number of APs')
    >>> plt.ylabel('Mean Error (m)')
    >>> plt.legend()
    >>> plt.show()
    """
    budgets = sorted([k for k in all_results.keys() if isinstance(k, int)])

    openjij_values = []
    sa_values = []

    for k in budgets:
        # Extract OpenJij metric
        try:
            oj_val = all_results[k]['openjij'][model]['val_metrics'][metric_name]
            openjij_values.append(oj_val)
        except (KeyError, TypeError):
            openjij_values.append(None)

        # Extract SA metric
        try:
            sa_val = all_results[k]['SA'][model]['val_metrics'][metric_name]
            sa_values.append(sa_val)
        except (KeyError, TypeError):
            sa_values.append(None)

    return {
        'budgets': budgets,
        'openjij': openjij_values,
        'SA': sa_values
    }


def print_comparison_table(all_results):
    """
    Print a formatted comparison table of all budgets.

    Parameters:
    -----------
    all_results : dict
        Results from run_multi_budget_experiment
    """
    budgets = sorted([k for k in all_results.keys() if isinstance(k, int)])

    print("\n" + "="*100)
    print("COMPREHENSIVE COMPARISON TABLE - RANDOM FOREST RESULTS")
    print("="*100)
    print(f"{'Budget':<10} {'Solver':<12} {'Mean Error':<15} {'90th Pct':<15} "
          f"{'Floor Acc':<15} {'Time (s)':<10}")
    print("-"*100)

    for k in budgets:
        if 'error' in all_results[k]:
            print(f"{k:<10} {'BOTH':<12} {'FAILED':<15}")
            continue

        # OpenJij row
        if 'random_forest' in all_results[k]['openjij']:
            oj_rf = all_results[k]['openjij']['random_forest']['val_metrics']
            oj_time = all_results[k]['openjij']['duration']
            print(f"{k:<10} {'OpenJij':<12} "
                  f"{oj_rf['real_mean_m']:<15.2f} "
                  f"{oj_rf['real_90th_m']:<15.2f} "
                  f"{oj_rf['floor_accuracy']:<15.1%} "
                  f"{oj_time:<10.2f}")

        # SA row
        if 'random_forest' in all_results[k]['SA']:
            sa_rf = all_results[k]['SA']['random_forest']['val_metrics']
            sa_time = all_results[k]['SA']['duration']
            print(f"{'':<10} {'SA':<12} "
                  f"{sa_rf['real_mean_m']:<15.2f} "
                  f"{sa_rf['real_90th_m']:<15.2f} "
                  f"{sa_rf['floor_accuracy']:<15.1%} "
                  f"{sa_time:<10.2f}")

        print("-"*100)

    print("="*100)
