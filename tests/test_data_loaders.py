"""Test script to verify data_loaders.py works with absolute paths"""
import sys
import os
from pathlib import Path

# Simulate running from different directories
print("=" * 70)
print("Testing data_loaders.py with absolute path resolution")
print("=" * 70)

# Test 1: From project root
print("\n[TEST 1] Running from project root...")
project_root = Path(__file__).parent
os.chdir(project_root)
print(f"  Current directory: {os.getcwd()}")

try:
    from scripts.data.data_loaders import (
        get_project_root,
        resolve_data_path,
        load_preprocessed_data
    )

    detected_root = get_project_root()
    print(f"  Detected project root: {detected_root}")
    print(f"  Match: {detected_root == project_root}")

    # Test path resolution
    test_path = resolve_data_path('data/output_data/preprocessed_data')
    print(f"  Resolved path: {test_path}")
    print(f"  Path exists: {test_path.exists()}")

    print("  SUCCESS: Project root detected correctly")
except Exception as e:
    print(f"  FAILED: {e}")

# Test 2: From notebooks directory
print("\n[TEST 2] Running from notebooks/01_experiments/...")
notebooks_dir = project_root / 'notebooks' / '01_experiments'
if notebooks_dir.exists():
    os.chdir(notebooks_dir)
    print(f"  Current directory: {os.getcwd()}")

    try:
        # Re-import to test from new directory
        import importlib
        import scripts.data.data_loaders as dl
        importlib.reload(dl)

        detected_root = dl.get_project_root()
        print(f"  Detected project root: {detected_root}")
        print(f"  Match: {detected_root == project_root}")

        # Test path resolution
        test_path = dl.resolve_data_path('data/output_data/preprocessed_data')
        print(f"  Resolved path: {test_path}")
        print(f"  Path exists: {test_path.exists()}")

        print("  SUCCESS: Project root detected correctly from notebooks dir")
    except Exception as e:
        print(f"  FAILED: {e}")
else:
    print("  SKIPPED: notebooks/01_experiments/ directory not found")

# Test 3: Load actual data
print("\n[TEST 3] Loading preprocessed data for building 1...")
os.chdir(project_root)

try:
    rssi_train, coords_train, rssi_val, coords_val, ap_columns = load_preprocessed_data(
        building_id=1,
        use_pickle=True
    )

    print(f"  SUCCESS: Data loaded!")
    print(f"    Training samples: {rssi_train.shape[0]}")
    print(f"    Validation samples: {rssi_val.shape[0]}")
    print(f"    Number of APs: {len(ap_columns)}")

except Exception as e:
    print(f"  FAILED: {e}")

# Test 4: Load importance scores
print("\n[TEST 4] Loading importance scores...")
try:
    from scripts.data.data_loaders import load_all_precomputed_data

    importance_dicts, redundancy_matrix = load_all_precomputed_data()

    print(f"  SUCCESS: Precomputed data loaded!")
    print(f"    Importance methods: {list(importance_dicts.keys())}")
    print(f"    Redundancy matrix shape: {redundancy_matrix.shape}")

except Exception as e:
    print(f"  FAILED: {e}")

print("\n" + "=" * 70)
print("TESTING COMPLETE")
print("=" * 70)
