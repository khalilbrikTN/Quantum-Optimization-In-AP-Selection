# Notebook Organization Guide

## Overview

The codebase has been reorganized to separate data processing/analysis from visualization. This improves clarity and allows you to generate visualizations independently of running experiments.

---

## Notebooks

### 1. `RUNNER.ipynb` - Main Experiment Notebook

**Purpose:** Run WiFi localization experiments and generate results

**Contents:**
- Data loading and preprocessing
- Importance score calculation (or loading from files)
- Redundancy matrix calculation (or loading from files)
- QUBO formulation and solving
- Machine learning model training
- Performance evaluation
- **Results saving** (both cleaned results and full results with APs)

**Output Files:**
```
data/output_data/
├── importance_scores/
│   ├── *_importance_dict.csv (for loading)
│   └── *.xlsx (for analysis)
├── redundancy_scores/
│   └── redundancy_matrix.csv
└── results/
    ├── cleaned_results_k{k}_alpha{alpha}_penalty{penalty}.xlsx
    └── results_with_aps.pkl (for visualizations)
```

**Key Features:**
- ✅ Load pre-computed importance scores and redundancy matrix from files
- ✅ Save all results for reproducibility
- ✅ No visualization code (keeps notebook clean and fast)

---

### 2. `produce_visualizations.ipynb` - Visualization Notebook

**Purpose:** Generate all visualizations from saved results

**Contents:**
- Load saved results from RUNNER.ipynb
- Generate 2D comparison plots (5 types)
- Generate 3D building fingerprint plots
- Generate 3D true vs predicted comparison plots
- Save all plots to files

**Input Files:**
```
data/output_data/results/
├── cleaned_results_k{k}_alpha{alpha}_penalty{penalty}.xlsx
└── results_with_aps.pkl
```

**Output Files:**
```
data/output_data/
├── visualizations_2d/
│   ├── error_comparison.png
│   ├── floor_accuracy_comparison.png
│   ├── error_range_comparison.png
│   ├── radar_chart.png
│   └── combined_error_accuracy.png
└── visualizations_3d/
    ├── 3d_comparison_entropy.png
    ├── 3d_comparison_average.png
    ├── 3d_comparison_max.png
    └── 3d_comparison_variance.png
```

**Key Features:**
- ✅ Automatically loads most recent results
- ✅ Generates all 2D comparison plots
- ✅ Generates 3D fingerprint visualizations
- ✅ Generates 3D prediction error visualizations
- ✅ Saves all plots to high-resolution PNG files
- ✅ Can be run independently of RUNNER.ipynb

---

## Workflow

### Complete Workflow (First Time)

```
1. Run RUNNER.ipynb
   ├─ Cells 1-6: Load and preprocess data
   ├─ Cells 8-14: Calculate importance scores and redundancy (OPTIONAL - can skip if already computed)
   ├─ Cell 18: Load pre-computed data from files
   ├─ Cell 20: Run QUBO experiments
   ├─ Cell 24: Save results
   └─ Output: cleaned_results_*.xlsx and results_with_aps.pkl

2. Run produce_visualizations.ipynb
   ├─ Automatically loads saved results
   ├─ Generates all 2D plots
   ├─ Generates all 3D plots
   └─ Saves plots to PNG files
```

### Quick Workflow (Subsequent Runs)

```
1. Run RUNNER.ipynb (skip cells 8-14)
   ├─ Cell 18: Load pre-computed importance/redundancy
   ├─ Cell 20: Run QUBO with new parameters (k, alpha, penalty)
   └─ Cell 24: Save results

2. Run produce_visualizations.ipynb
   ├─ Loads new results automatically
   └─ Generates updated visualizations
```

---

## Benefits of Separation

### 1. **Speed**
- RUNNER.ipynb is faster without visualization rendering
- Regenerate visualizations without re-running experiments

### 2. **Clarity**
- RUNNER.ipynb focuses on data processing and analysis
- produce_visualizations.ipynb focuses only on visualization
- Easier to find and modify specific plots

### 3. **Flexibility**
- Change visualization parameters without re-running experiments
- Generate visualizations for multiple experiment results
- Easy to create custom visualizations

### 4. **Reproducibility**
- Results are saved to files (Excel + pickle)
- Visualizations can be regenerated anytime
- Easy to share results without code

### 5. **Presentation**
- All plots saved to high-resolution PNG files
- Ready for papers, presentations, thesis
- Consistent styling across all plots

---

## File Dependencies

### RUNNER.ipynb Dependencies
```
Input:
├── data/input_data/TrainingData.csv
├── data/input_data/ValidationData.csv
└── (optional) data/output_data/importance_scores/*.csv
                data/output_data/redundancy_scores/redundancy_matrix.csv

Output:
├── data/output_data/results/cleaned_results_*.xlsx
└── data/output_data/results/results_with_aps.pkl
```

### produce_visualizations.ipynb Dependencies
```
Input:
├── data/output_data/results/cleaned_results_*.xlsx
├── data/output_data/results/results_with_aps.pkl
├── data/input_data/TrainingData.csv (for 3D plots)
└── data/input_data/ValidationData.csv (for 3D plots)

Output:
├── data/output_data/visualizations_2d/*.png
└── data/output_data/visualizations_3d/*.png
```

---

## Visualization Types

### 2D Plots (5 types)

1. **Error Comparison** - Bar chart comparing median 3D error across methods
2. **Floor Accuracy Comparison** - Bar chart comparing floor prediction accuracy
3. **Error Range Comparison** - Min/max error ranges for each method
4. **Radar Chart** - Multi-metric comparison on normalized scale
5. **Combined Error and Accuracy** - Side-by-side error and accuracy comparison

### 3D Plots (Multiple types)

**Building Fingerprints:**
- Colored by floor level
- Colored by RSSI strength of selected APs
- Colored by measurement density
- Real-world coordinates vs normalized coordinates

**True vs Predicted Comparisons:**
- Green circles = true locations
- Red triangles = predicted locations
- Blue lines = error vectors
- Color intensity = error magnitude
- Statistics box with mean/median/90th percentile errors

---

## Tips

### For Running Experiments

1. **First time:** Run cells 8-14 to compute importance scores and redundancy matrix
2. **Subsequent runs:** Skip cells 8-14, use cell 18 to load from files
3. **Parameter tuning:** Change k, alpha, penalty in cell 17, then run from cell 20
4. **Always run cell 24** to save results for visualization notebook

### For Generating Visualizations

1. **Run after RUNNER.ipynb** to ensure latest results are available
2. **Automatic loading:** Notebook finds the most recent results file
3. **Selective plotting:** Comment out cells for plots you don't need
4. **Custom plots:** Add new cells with your own visualization code
5. **High-res outputs:** All plots saved as PNG for publication quality

### For Presentations/Papers

1. Run produce_visualizations.ipynb with `save_path` parameters
2. Find plots in `data/output_data/visualizations_2d/` and `visualizations_3d/`
3. Use PNG files directly in LaTeX, PowerPoint, Word, etc.
4. All plots use consistent styling (colors, fonts, sizes)

---

## Troubleshooting

### "No results found" error in produce_visualizations.ipynb
**Solution:** Run RUNNER.ipynb first and ensure cell 24 executes successfully

### "results_with_aps.pkl not found" warning
**Solution:** This file is needed for 3D visualizations with selected APs. Ensure RUNNER.ipynb cell 24 completes.

### Plots look different from RUNNER.ipynb
**Solution:** This is normal - visualizations now load from saved results. Re-run RUNNER.ipynb if you need updated results.

### Want to visualize old experiment results
**Solution:** The notebook automatically uses the most recent file. To use older results, manually specify the filename in the loading cell.

---

## Summary

| Aspect | RUNNER.ipynb | produce_visualizations.ipynb |
|--------|--------------|------------------------------|
| **Purpose** | Run experiments | Generate plots |
| **Input** | Raw data | Saved results |
| **Output** | Results files | Plot files |
| **Runtime** | Slower (experiments) | Faster (plotting only) |
| **Frequency** | When parameters change | Anytime |
| **Dependencies** | Training data | Results from RUNNER |

**Best Practice:** Run RUNNER.ipynb once with your desired parameters, then use produce_visualizations.ipynb to generate and customize all plots.
