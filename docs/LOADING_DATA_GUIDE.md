# Guide: Loading Pre-computed Importance Scores and Redundancy Matrix

## Overview

This guide explains how to save and load pre-computed importance scores and redundancy matrices to avoid expensive re-computation.

## Benefits

- **Save Time**: Importance calculation and redundancy matrix computation can take several minutes
- **Reproducibility**: Use the same importance scores across multiple experiments
- **Flexibility**: Run QUBO formulations with different parameters without recalculating features

---

## Workflow

### 1. First-Time Setup (Compute and Save)

Run these cells **once** to compute and save the data:

```python
# Cell 6: Load and preprocess data
rssi_train, coords_train, rssi_val, coords_val, ap_columns = load_and_preprocess_data(...)

# Cell 8: Calculate importance scores (all methods)
importance_entropy_dict = entropy_importance(rssi_train, ap_columns)
importance_average_dict = average_importance(rssi_train, ap_columns)
# ... etc for all methods

# Cell 10: Save importance scores
save_all_importance_dicts(importance_dicts, 'data/output_data/importance_scores')

# Cell 13: Calculate redundancy matrix
redundancy_matrix = calculate_redundancy_matrix(rssi_train)

# Cell 14: Save redundancy matrix
redundancy_matrix.to_csv('data/output_data/redundancy_scores/redundancy_matrix.csv')
```

**Output Files:**
```
data/output_data/
├── importance_scores/
│   ├── entropy_importance_dict.csv
│   ├── average_importance_dict.csv
│   ├── median_importance_dict.csv
│   ├── max_importance_dict.csv
│   └── variance_importance_dict.csv
└── redundancy_scores/
    └── redundancy_matrix.csv
```

---

### 2. Subsequent Runs (Load from Files)

**Skip cells 8-14** and run this instead:

```python
# Load all pre-computed data at once
from data_loaders import load_all_precomputed_data

importance_dicts_loaded, redundancy_matrix_loaded = load_all_precomputed_data()
```

**Output:**
```
============================================================
Loading pre-computed importance scores and redundancy matrix
============================================================

Loading importance scores...
✓ Loaded 520 APs for entropy importance
✓ Loaded 520 APs for average importance
✓ Loaded 520 APs for median importance
✓ Loaded 520 APs for max importance
✓ Loaded 520 APs for variance importance

Loading redundancy matrix...
✓ Loaded redundancy matrix with shape: (520, 520)

✓ All data loaded successfully!
============================================================
```

---

### 3. Use Loaded Data in QUBO Formulation

```python
# Access individual importance methods
for label in ['entropy', 'average', 'max', 'variance']:
    imp_dict = importance_dicts_loaded[label]  # ← Load from file

    # Formulate QUBO
    Q, relevant_aps, offset = formulate_qubo(
        imp_dict,                    # ← From loaded data
        redundancy_matrix_loaded,    # ← From loaded data
        k=20,
        alpha=0.9,
        penalty=2.0
    )

    # Continue with QUBO solving...
```

---

## API Reference

### `data_loaders.py` Module

#### Main Functions

**`load_all_precomputed_data()`**
```python
importance_dicts, redundancy_matrix = load_all_precomputed_data(
    importance_dir='data/output_data/importance_scores',
    redundancy_dir='data/output_data/redundancy_scores'
)
```
- Returns: `(dict, DataFrame)`
- Loads all importance scores and redundancy matrix at once

**`load_all_importance_scores()`**
```python
importance_dicts = load_all_importance_scores(
    base_dir='data/output_data/importance_scores'
)
```
- Returns: `dict` with keys: `['entropy', 'average', 'median', 'max', 'variance']`
- Each value is a dictionary: `{AP_name: score}`

**`load_redundancy_matrix()`**
```python
redundancy_matrix = load_redundancy_matrix(
    base_dir='data/output_data/redundancy_scores'
)
```
- Returns: `DataFrame` with AP names as index and columns

**`save_all_importance_dicts()`**
```python
save_all_importance_dicts(
    importance_dicts,
    output_dir='data/output_data/importance_scores'
)
```
- Saves all importance dictionaries to CSV files

---

## File Formats

### Importance Dictionary CSV
```csv
AP,Score
WAP001,0.523
WAP002,0.0
WAP003,1.245
...
```

### Redundancy Matrix CSV
```csv
,WAP001,WAP002,WAP003,...
WAP001,1.0,0.234,0.156,...
WAP002,0.234,1.0,0.678,...
WAP003,0.156,0.678,1.0,...
...
```

---

## Example: Complete Workflow

### Initial Run (Compute Everything)

```python
# 1. Import modules
from data_loaders import save_all_importance_dicts, load_all_precomputed_data
from pre_processing import load_and_preprocess_data
from Importance import entropy_importance, average_importance, max_importance, variance_importance
from Redundancy import calculate_redundancy_matrix

# 2. Load and preprocess data
rssi_train, coords_train, rssi_val, coords_val, ap_columns = load_and_preprocess_data(
    'data/input_data/TrainingData.csv',
    'data/input_data/ValidationData.csv',
    building_id=1
)

# 3. Calculate importance scores
_, importance_entropy = entropy_importance(rssi_train, ap_columns)
_, importance_average = average_importance(rssi_train, ap_columns)
_, importance_max = max_importance(rssi_train, ap_columns)
_, importance_variance = variance_importance(rssi_train, ap_columns)

# 4. Save importance scores
importance_dicts = {
    'entropy': importance_entropy,
    'average': importance_average,
    'max': importance_max,
    'variance': importance_variance
}
save_all_importance_dicts(importance_dicts)

# 5. Calculate and save redundancy matrix
redundancy_matrix = calculate_redundancy_matrix(rssi_train)
redundancy_matrix.to_csv('data/output_data/redundancy_scores/redundancy_matrix.csv')

print("✓ All data computed and saved!")
```

### Subsequent Runs (Load and Use)

```python
# 1. Import modules
from data_loaders import load_all_precomputed_data
from QUBO import formulate_qubo, solve_qubo_with_openjij

# 2. Load pre-computed data (fast!)
importance_dicts, redundancy_matrix = load_all_precomputed_data()

# 3. Run experiments with different parameters
for k in [10, 20, 30, 40]:
    for alpha in [0.7, 0.8, 0.9]:
        Q, relevant_aps, offset = formulate_qubo(
            importance_dicts['max'],  # ← Loaded from file
            redundancy_matrix,         # ← Loaded from file
            k=k,
            alpha=alpha,
            penalty=2.0
        )

        selected_indices, duration = solve_qubo_with_openjij(Q)
        # Continue processing...
```

---

## Troubleshooting

### FileNotFoundError
**Problem:** `FileNotFoundError: redundancy_matrix.csv not found`

**Solution:** Run cells 8-14 to compute and save the data first.

### Empty Dictionaries
**Problem:** Loaded importance dictionaries are empty

**Solution:** Check that CSV files exist in `data/output_data/importance_scores/`

### Different AP Counts
**Problem:** Loaded data has different number of APs than expected

**Solution:** Re-run preprocessing with the same building_id used during computation

---

## Performance Comparison

| Operation | Without Loading | With Loading | Time Saved |
|-----------|----------------|--------------|------------|
| Importance calculation | ~30 seconds | ~0.5 seconds | **98%** |
| Redundancy matrix | ~45 seconds | ~1 second | **98%** |
| **Total preprocessing** | **~75 seconds** | **~1.5 seconds** | **98%** |

**Note:** Times are approximate and depend on dataset size and hardware.

---

## Best Practices

1. **Version Control**: Save different versions with timestamps
   ```python
   timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
   save_all_importance_dicts(dicts, f'data/output_data/importance_scores_{timestamp}')
   ```

2. **Verify Loaded Data**: Always check shapes and counts
   ```python
   print(f"Loaded {len(importance_dicts['entropy'])} APs")
   print(f"Matrix shape: {redundancy_matrix.shape}")
   ```

3. **Document Parameters**: Save metadata with your data
   ```python
   metadata = {
       'building_id': 1,
       'floor_height': 3.0,
       'date': datetime.now().isoformat()
   }
   pd.Series(metadata).to_csv('data/output_data/metadata.csv')
   ```

---

## Questions?

See the notebook `RUNNER.ipynb` for working examples, or check the module docstrings:
```python
from data_loaders import load_all_precomputed_data
help(load_all_precomputed_data)
```
