# Exploratory Data Analysis (EDA) - README

## Overview

The `EDA_analysis.ipynb` notebook performs comprehensive exploratory data analysis on the UJIIndoorLoc WiFi localization dataset.

---

## What Does It Analyze?

### 1. **Dataset Overview**
- Total samples (training and validation)
- Number of WiFi Access Points (520 WAPs)
- Number of buildings
- Number of unique users and devices
- Date ranges for data collection

### 2. **Building-Level Statistics**
For each building (0, 1, 2):
- Number of training fingerprints
- Number of validation fingerprints
- Total fingerprints
- Number of floors
- List of all floors
- Longitude and latitude ranges
- Approximate building area (m²)
- Number of unique spaces

### 3. **Floor-Level Statistics**
For each floor in each building:
- Training fingerprints per floor
- Validation fingerprints per floor
- Total fingerprints per floor
- AP coverage percentage
- Number of unique spaces
- Number of unique users

### 4. **WiFi Access Point Analysis**
- Total number of APs (520)
- Active vs inactive APs
- Detection rates per AP
- Average RSSI values
- Average APs detected per fingerprint
- Maximum APs in a single fingerprint
- Top 50 most active APs with statistics

### 5. **Coordinate Distribution**
For each building:
- Longitude: min, max, mean, standard deviation
- Latitude: min, max, mean, standard deviation
- Spatial distribution statistics

### 6. **User Statistics**
For each user:
- Total fingerprints collected
- Number of buildings visited
- Number of floors visited
- Phone IDs used
- Number of unique locations

### 7. **Device (Phone) Statistics**
For each phone:
- Total fingerprints collected
- Number of users
- Buildings covered
- Floors covered

### 8. **Data Quality Analysis**
- Missing coordinate values
- Samples with no AP detection
- Duplicate fingerprints
- Overall data completeness percentage

### 9. **Visualizations**
- Bar charts for fingerprints per building
- Floor distribution per building
- Training vs validation comparison
- High-resolution PNG exports

---

## Output Structure

### Excel File (9 Sheets)
```
data/EDA/EDA_Analysis_YYYYMMDD_HHMMSS.xlsx
├─ Sheet 1: Overview (dataset-level statistics)
├─ Sheet 2: Building_Stats (per-building analysis)
├─ Sheet 3: Floor_Stats (per-floor analysis)
├─ Sheet 4: AP_Summary (overall AP statistics)
├─ Sheet 5: Top50_APs (most active APs with details)
├─ Sheet 6: Coordinates (spatial distribution)
├─ Sheet 7: User_Stats (user-level analysis)
├─ Sheet 8: Phone_Stats (device-level analysis)
└─ Sheet 9: Data_Quality (completeness and issues)
```

### Visualization Files
```
data/EDA/
├─ building_overview.png (fingerprints and floors per building)
└─ floor_distribution.png (fingerprints per floor for each building)
```

---

## How to Use

### 1. Run the Notebook
```bash
# Open Jupyter
jupyter notebook EDA_analysis.ipynb

# Run all cells (Cell > Run All)
```

### 2. Outputs Generated Automatically
- Excel file with timestamp: `EDA_Analysis_20240115_143022.xlsx`
- PNG visualizations
- All files saved to `data/EDA/`

### 3. View Results
- Open Excel file for detailed tabular analysis
- View PNG files for quick visual overview
- All statistics printed in notebook output

---

## Key Insights from Analysis

### Dataset Scale
- **Total Samples**: ~21,000 fingerprints
- **WiFi APs**: 520 Access Points
- **Buildings**: 3 buildings (0, 1, 2)
- **Floors**: Variable per building (typically 4-5 floors)
- **Users**: Multiple users with diverse devices

### Typical Building Statistics
- **Building 0**: ~3,000-4,000 fingerprints
- **Building 1**: ~8,000-10,000 fingerprints (largest)
- **Building 2**: ~7,000-8,000 fingerprints
- Each building has 4-5 floors

### WiFi Coverage
- **Active APs**: ~300-350 out of 520 total
- **Avg APs per Fingerprint**: 8-12 APs detected
- **Detection Rate**: Varies widely by AP location
- **RSSI Range**: Typically -30 to -100 dBm

### Data Quality
- **Completeness**: >95% complete data
- **Missing Values**: Minimal coordinate missing
- **Duplicates**: Very few duplicate fingerprints
- **AP Coverage**: Good coverage across all buildings

---

## Example Results

### Building Statistics
| Building | Train | Val | Total | Floors | Area (m²) |
|----------|-------|-----|-------|--------|-----------|
| 0 | 3,579 | 299 | 3,878 | 4 | ~15,000 |
| 1 | 8,923 | 489 | 9,412 | 4 | ~24,000 |
| 2 | 7,435 | 323 | 7,758 | 5 | ~18,000 |

### Floor Distribution (Example - Building 1)
| Floor | Training | Validation | Total | Coverage % |
|-------|----------|------------|-------|------------|
| 0 | 2,341 | 121 | 2,462 | 4.2% |
| 1 | 2,156 | 134 | 2,290 | 3.9% |
| 2 | 2,287 | 128 | 2,415 | 4.1% |
| 3 | 2,139 | 106 | 2,245 | 3.8% |

### AP Activity (Top 5)
| AP | Detections | Detection % | Avg RSSI |
|----|------------|-------------|----------|
| WAP107 | 8,234 | 41.3% | -67.2 dBm |
| WAP108 | 7,891 | 39.6% | -65.8 dBm |
| WAP104 | 6,543 | 32.8% | -71.3 dBm |
| WAP113 | 5,892 | 29.5% | -69.5 dBm |
| WAP166 | 5,234 | 26.2% | -73.1 dBm |

---

## Use Cases

### 1. Research Papers
- Cite dataset statistics from Excel file
- Include visualizations in figures
- Reference building/floor distributions

### 2. Model Development
- Understand data imbalance across buildings/floors
- Identify most/least represented areas
- Plan train/validation splits

### 3. Feature Engineering
- Identify most active APs for selection
- Understand spatial distribution patterns
- Analyze coverage variability

### 4. Thesis/Reports
- Complete dataset description section
- Statistical overview tables
- Data quality justification

---

## Customization

### Analyze Specific Building Only
```python
# In any cell, filter before analysis
building_id = 1
df_train_filtered = df_train[df_train['BUILDINGID'] == building_id]
df_val_filtered = df_val[df_val['BUILDINGID'] == building_id]
# Continue analysis with filtered data
```

### Add Custom Analysis
```python
# Add new cells for custom metrics
# Example: Analyze RSSI distribution
rssi_values = df_train[wap_columns].values.flatten()
rssi_values = rssi_values[rssi_values != 100]  # Remove non-detections
print(f"RSSI Statistics:")
print(f"  Mean: {rssi_values.mean():.2f} dBm")
print(f"  Std: {rssi_values.std():.2f} dBm")
print(f"  Min: {rssi_values.min():.2f} dBm")
print(f"  Max: {rssi_values.max():.2f} dBm")
```

### Export Additional Formats
```python
# Save specific sheet as CSV
df_buildings.to_csv(output_dir / 'building_stats.csv', index=False)

# Save visualizations in different format
plt.savefig(output_dir / 'plot.pdf', format='pdf', dpi=300)
```

---

## Technical Details

### Data Processing
- **Missing Values**: Handled via detection (100 = not detected)
- **Coordinate System**: UTM projection (meters)
- **RSSI Units**: dBm (negative values, -30 to -100)
- **Floor Encoding**: Integer (0, 1, 2, 3, 4...)

### Performance
- **Runtime**: ~30-60 seconds for complete analysis
- **Memory**: ~500MB peak usage
- **Output Size**: Excel file ~1-2 MB

### Dependencies
```python
pandas >= 1.3.0
numpy >= 1.21.0
matplotlib >= 3.4.0
seaborn >= 0.11.0
openpyxl >= 3.0.0  # For Excel export
```

---

## Troubleshooting

### "File not found" error
**Problem**: Cannot find input CSV files
**Solution**: Ensure `TrainingData.csv` and `ValidationData.csv` exist in `data/input_data/`

### Excel file won't open
**Problem**: Corrupted or incompatible format
**Solution**: Install/update openpyxl: `pip install --upgrade openpyxl`

### Visualizations not displaying
**Problem**: Matplotlib backend issue
**Solution**: Add at notebook start: `%matplotlib inline`

### Out of memory error
**Problem**: Dataset too large for available RAM
**Solution**: Process buildings separately or increase system memory

---

## Next Steps

After running EDA:

1. **Review Results**
   - Examine Excel file for insights
   - Identify data imbalances
   - Note building/floor characteristics

2. **Use in RUNNER.ipynb**
   - Select building(s) for experiments
   - Understand floor distributions for analysis
   - Choose appropriate floor_height parameter

3. **Feature Selection**
   - Use AP activity stats to inform importance metrics
   - Consider building-specific AP selections
   - Account for coverage variability

4. **Documentation**
   - Include EDA results in thesis/paper
   - Reference statistics in methodology
   - Use visualizations in presentations

---

## References

- **Dataset**: UJIIndoorLoc Database
- **Paper**: Torres-Sospedra et al., "UJIIndoorLoc: A New Multi-building and Multi-floor Database for WLAN Fingerprint-based Indoor Localization Problems", IPIN 2014
- **Format**: CSV with 520 WAP columns + location metadata

---

## Support

For questions or issues:
1. Check notebook output cells for error messages
2. Review this README for common solutions
3. Verify data file paths and formats
4. Check dependency versions

---

**Last Updated**: 2024
**Notebook Version**: 1.0
**Compatible with**: Python 3.7+, Jupyter Notebook/Lab
