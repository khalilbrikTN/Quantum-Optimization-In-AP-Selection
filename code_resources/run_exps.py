
from datetime import datetime
import os


from pre_processing import *
from Importance import *
from Redundancy import *
from QUBO import *
from ML_post_processing import *
from Visualizations_Functions import *


def run_multi_budget_experiment(df_train_path, df_validation_path,
                                building_id, floor_height,
                                budget_list, alpha, penalty,
                                num_reads=1000, num_sweeps=1000,
                                output_folder=None):
    """
    Run the complete AP selection pipeline across multiple budgets with Excel export.

    Parameters:
    -----------
    df_train_path : str
        Path to training CSV file
    df_validation_path : str
        Path to validation CSV file
    building_id : int
        ID of building of interest
    floor_height : float
        Height of every floor in meters
    budget_list : list of int
        List of AP budgets (k values) to test
    alpha : float
        Weight for importance vs redundancy (0-1)
    penalty : float
        Penalty for constraint violation
    num_reads : int, optional
        Number of annealing runs (default: 1000)
    num_sweeps : int, optional
        Number of annealing sweeps (default: 1000)
    output_folder : str, optional
        Path to output folder for Excel files (default: None, no export)

    Returns:
    --------
    all_results : dict
        Complete results dictionary
    """

    print("="*80)
    print("MULTI-BUDGET AP SELECTION EXPERIMENT")
    print("="*80)
    print(f"Budgets to test: {budget_list}")
    print(f"Building ID: {building_id}")
    print(f"Alpha: {alpha}, Penalty: {penalty}")
    print(f"Annealing params: {num_reads} reads, {num_sweeps} sweeps")

    # Setup output folder if provided
    if output_folder:
        print(f"Output folder: {output_folder}")
        try:
            os.makedirs(output_folder, exist_ok=True)
            print("✓ Output folder ready")
        except Exception as e:
            print(f"⚠ Warning: Could not create output folder: {e}")
            print("  Continuing without file export...")
            output_folder = None

    print("="*80 + "\n")

    # Generate timestamp for this experiment
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Initialize results storage
    all_results = {}

    # STEP 1: Load and preprocess data
    print("\n" + "█"*80)
    print("STEP 1: LOADING AND PREPROCESSING DATA (ONE TIME)")
    print("█"*80)

    rssi_train, coords_train, rssi_val, coords_val, ap_columns = load_and_preprocess_data(
        df_train_path, df_validation_path, building_id, floor_height
    )
    total_aps = len(ap_columns)

    # STEP 2: Calculate importance scores
    print("\n" + "█"*80)
    print("STEP 2: CALCULATING IMPORTANCE SCORES (ONE TIME)")
    print("█"*80)

    _, importance_scores = mutual_information_importance(rssi_train, coords_train)

    # STEP 3: Calculate redundancy matrix
    print("\n" + "█"*80)
    print("STEP 3: CALCULATING REDUNDANCY MATRIX (ONE TIME)")
    print("█"*80)

    redundancy_matrix = calculate_redundancy_matrix(rssi_train, importance_scores)

    # Loop through each budget
    for budget_idx, k in enumerate(budget_list, 1):
        print("\n\n" + "╔"+"═"*78+"╗")
        print(f"║  BUDGET {budget_idx}/{len(budget_list)}: k = {k} APs" + " "*(78-len(f"  BUDGET {budget_idx}/{len(budget_list)}: k = {k} APs")) + "║")
        print("╚"+"═"*78+"╝")

        # Initialize storage for this budget
        all_results[k] = {
            'openjij': {},
            'SA': {},
            'budget': k,
            'total_aps': total_aps
        }

        try:
            # STEP 4: Formulate QUBO
            print("\n" + "─"*80)
            print(f"STEP 4: FORMULATING QUBO (k={k})")
            print("─"*80)

            Q, relevant_aps, offset = formulate_qubo(
                importance_scores, redundancy_matrix, k, alpha, penalty
            )

            # STEP 5.1: Solve with OpenJij
            print("\n" + "─"*80)
            print(f"STEP 5.1: SOLVING WITH OPENJIJ (k={k})")
            print("─"*80)

            selected_indices_openjij, duration_openjij = solve_qubo_with_openjij(
                Q, num_reads=num_reads, num_sweeps=num_sweeps
            )
            selected_aps_openjij = [relevant_aps[i] for i in selected_indices_openjij]

            # Store OpenJij metadata
            all_results[k]['openjij']['selected_aps'] = selected_aps_openjij
            all_results[k]['openjij']['num_selected'] = len(selected_aps_openjij)
            all_results[k]['openjij']['duration'] = duration_openjij
            all_results[k]['openjij']['total_aps'] = total_aps
            all_results[k]['openjij']['alpha'] = alpha
            all_results[k]['openjij']['penalty'] = penalty

            # STEP 5.2: Solve with SA
            print("\n" + "─"*80)
            print(f"STEP 5.2: SOLVING WITH SA (k={k})")
            print("─"*80)

            selected_indices_SA, duration_SA = solve_qubo_with_SA(
                Q, num_reads=num_reads, num_sweeps=num_sweeps
            )
            selected_aps_SA = [relevant_aps[i] for i in selected_indices_SA]

            # Store SA metadata
            all_results[k]['SA']['selected_aps'] = selected_aps_SA
            all_results[k]['SA']['num_selected'] = len(selected_aps_SA)
            all_results[k]['SA']['duration'] = duration_SA
            all_results[k]['SA']['total_aps'] = total_aps
            all_results[k]['SA']['alpha'] = alpha
            all_results[k]['SA']['penalty'] = penalty

            # STEP 6.1: Train ML with OpenJij APs
            print("\n" + "─"*80)
            print(f"STEP 6.1: TRAINING ML MODELS WITH OPENJIJ APs (k={k})")
            print("─"*80)

            if selected_aps_openjij:
                models_openjij, predictions_openjij = train_regressor(
                    rssi_train, coords_train, rssi_val, coords_val,
                    selected_aps_openjij, model_type='random_forest'
                )
                print("✓ Model training completed (OpenJij)")
            else:
                print("⚠ WARNING: No APs selected by OpenJij, skipping training")
                predictions_openjij = {}

            # STEP 6.2: Train ML with SA APs
            print("\n" + "─"*80)
            print(f"STEP 6.2: TRAINING ML MODELS WITH SA APs (k={k})")
            print("─"*80)

            if selected_aps_SA:
                models_SA, predictions_SA = train_regressor(
                    rssi_train, coords_train, rssi_val, coords_val,
                    selected_aps_SA, model_type='random_forest'
                )
                print("✓ Model training completed (SA)")
            else:
                print("⚠ WARNING: No APs selected by SA, skipping training")
                predictions_SA = {}

            # STEP 7.1: Evaluate OpenJij results
            print("\n" + "─"*80)
            print(f"STEP 7.1: EVALUATING OPENJIJ RESULTS (k={k})")
            print("─"*80)

            results_openjij = evaluate_and_print_results(
                'openjij', coords_train, coords_val,
                predictions_openjij, selected_aps_openjij,
                predictions_SA, selected_aps_SA,
                total_aps
            )

            # Store OpenJij results
            if results_openjij:
                all_results[k]['openjij'].update(results_openjij)

            # STEP 7.2: Evaluate SA results
            print("\n" + "─"*80)
            print(f"STEP 7.2: EVALUATING SA RESULTS (k={k})")
            print("─"*80)

            results_SA = evaluate_and_print_results(
                'SA', coords_train, coords_val,
                predictions_openjij, selected_aps_openjij,
                predictions_SA, selected_aps_SA,
                total_aps
            )

            # Store SA results
            if results_SA:
                all_results[k]['SA'].update(results_SA)

            # STEP 8: Save individual results to Excel (if output folder specified)
            if output_folder:
                print("\n" + "─"*80)
                print(f"STEP 8: SAVING RESULTS TO EXCEL (k={k})")
                print("─"*80)

                save_individual_budget_results(
                    output_folder, k, 'openjij',
                    all_results[k]['openjij'], timestamp
                )
                save_individual_budget_results(
                    output_folder, k, 'SA',
                    all_results[k]['SA'], timestamp
                )

            print(f"\n✓ Budget k={k} completed successfully!")

        except Exception as e:
            print(f"\n✗ ERROR processing budget k={k}: {e}")
            import traceback
            traceback.print_exc()
            all_results[k]['error'] = str(e)
            continue

    # Save master summary file
    if output_folder:
        print("\n\n" + "█"*80)
        print("SAVING MASTER SUMMARY FILE")
        print("█"*80)
        save_master_summary(output_folder, all_results, timestamp, alpha, penalty)

    # Print final summary
    print("\n\n" + "╔"+"═"*78+"╗")
    print("║" + " "*25 + "EXPERIMENT COMPLETE" + " "*34 + "║")
    print("╚"+"═"*78+"╝")

    print("\nSUMMARY OF ALL BUDGETS:")
    print("-"*80)

    for k in budget_list:
        if k in all_results and 'error' not in all_results[k]:
            print(f"\nBudget k={k}:")

            # OpenJij summary
            if 'random_forest' in all_results[k]['openjij']:
                oj_rf = all_results[k]['openjij']['random_forest']['val_metrics']
                print(f"  OpenJij RF: {oj_rf['real_mean_m']:.2f}m, "
                      f"Floor Acc: {oj_rf['floor_accuracy']:.1%}, "
                      f"Time: {all_results[k]['openjij']['duration']:.2f}s")

            # SA summary
            if 'random_forest' in all_results[k]['SA']:
                sa_rf = all_results[k]['SA']['random_forest']['val_metrics']
                print(f"  SA RF:      {sa_rf['real_mean_m']:.2f}m, "
                      f"Floor Acc: {sa_rf['floor_accuracy']:.1%}, "
                      f"Time: {all_results[k]['SA']['duration']:.2f}s")
        else:
            print(f"\nBudget k={k}: ✗ FAILED")

    print("\n" + "="*80)
    print(f"Total budgets tested: {len(budget_list)}")
    print(f"Successful runs: {sum(1 for k in budget_list if k in all_results and 'error' not in all_results[k])}")
    if output_folder:
        print(f"Results saved to: {output_folder}")
    print("="*80)

    return all_results