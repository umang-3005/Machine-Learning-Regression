# Machine Learning Challenge Winter 2025 - Dataset ID: 37

**Course:** Machine Learning Project Work W25/26  
**Institution:** Ostbayerische Technische Hochschule Amberg-Weiden  
**Supervisor:** Prof. Dr. Patrick Levi  
**Author:** Umang Dholakiya

## 📋 Project Overview

This project addresses a two-part tabular regression challenge with distinct objectives and deployment constraints:

- **Part 1:** Build a high-performance ML model to predict the continuous target `target01` from 273 features
- **Part 2:** Reverse-engineer a rule-based system for `target02` that must run on edge devices without ML inference capabilities

**Dataset Specifications:**
- Training samples: 10,000
- Features: 273 (feat_1 through feat_273)
- Two regression targets: `target01` and `target02`

## 🗂️ Repository Structure

```
├── Part1.py                      # Main pipeline for target01 prediction
├── Part2.py                      # Rule discovery and validation for target02
├── parameter_experiment.py       # Hyperparameter tuning experiments
├── framework_37.py              # Deployment-ready rule engine for target02
├── requirements.txt             # Python dependencies
├── EVAL_target01_37.csv        # Final predictions for target01
├── Report_ML_W26.pdf           # Complete technical report
├── pra_mal_w25.pdf             # Project assignment specification
└── problem_37/                 # Dataset directory (not included)
    ├── dataset_37.csv          # Training features
    ├── target_37.csv           # Training targets
    └── EVAL_37.csv             # Evaluation features
```

## 🎯 Part 1: Predicting target01

### Approach

A robust regression pipeline combining feature selection with gradient boosting:

1. **Data Preprocessing:** Median imputation for missing values
2. **Feature Selection:** ExtraTreesRegressor-based selection (273 → 136 features)
3. **Model:** HistGradientBoostingRegressor with optimized hyperparameters
4. **Validation:** 5-fold cross-validation with out-of-fold predictions

### Pipeline Architecture

```python
Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('selector', SelectFromModel(
        estimator=ExtraTreesRegressor(n_estimators=400),
        threshold='median'
    )),
    ('reg', HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_leaf_nodes=31,
        min_samples_leaf=10,
        max_iter=900,
        random_state=42
    ))
])
```

### Performance Metrics

| Metric | With Feature Selection | Without Selection | Improvement |
|--------|------------------------|-------------------|-------------|
| MAE    | 0.070452 ± 0.004699   | 0.074450 ± 0.005234 | **-5.37%** |
| RMSE   | 0.088561 ± 0.005073   | 0.093234 ± 0.005229 | **-5.01%** |
| R²     | 0.859576 ± 0.017345   | 0.844368 ± 0.019170 | **+1.80%** |

### Running Part 1

```bash
python Part1.py
```

**Outputs:**
- `EVAL_target01_37.csv` - Predictions for evaluation set
- `target01_histogram.png` - Target distribution visualization
- `domain_shift_roc.png` - Domain shift analysis

## 🧩 Part 2: Rule-Based Prediction for target02

### Problem Statement

Predict `target02` using only simple conditional rules and arithmetic operations (no ML at runtime) for edge device deployment.

### Discovered Rule System

**Key Feature:** `feat_132` (gating feature)  
**Predictor Features:** `feat_108`, `feat_116`, `feat_255`  
**Thresholds:** 0.2, 0.5, 0.7

### Rule Formulas

```python
if feat_132 ≤ 0.2:
    target02 = 1.35×feat_108 + 1.75×feat_116 - 0.75×feat_255

elif feat_132 ≤ 0.5:
    target02 = 0.35×feat_108 - 0.45×feat_116 + 0.55×feat_255

elif feat_132 ≤ 0.7:
    target02 = 0.15×feat_108 + 0.85×feat_116 - 1.95×feat_255

else:  # feat_132 > 0.7
    target02 = 1.85×feat_108 - 1.75×feat_116 - 0.75×feat_255
```

### Validation Results

| Region | Condition | Samples | R² | Max Error |
|--------|-----------|---------|----|-----------| 
| 1 | ≤ 0.2 | 1,999 | 1.0 | ~10⁻¹⁵ |
| 2 | (0.2, 0.5] | 2,951 | 1.0 | ~10⁻¹⁵ |
| 3 | (0.5, 0.7] | 2,011 | 1.0 | ~10⁻¹⁵ |
| 4 | > 0.7 | 3,039 | 1.0 | ~10⁻¹⁵ |

**Note:** Errors at 10⁻¹⁵ scale indicate floating-point precision limits, confirming perfect reconstruction.

### Running Part 2

```bash
# Rule discovery and validation
python Part2.py

# Deploy rules (edge device compatible)
python framework_37.py --eval_file_path problem_37/EVAL_37.csv
```

**Outputs:**
- `rules_target02_37.json` - Extracted rule parameters
- `part2_decision_tree.png` - Tree visualization
- `part2_coefficient_contributions.png` - Region-wise coefficients
- `part2_feature_distribution.png` - Split feature analysis
- `part2_actual_vs_predicted.png` - Reconstruction validation

## 🛠️ Installation

### Requirements

```bash
pip install -r requirements.txt
```

**Dependencies:**
- numpy >= 1.21.0
- pandas >= 1.3.0
- scikit-learn >= 1.0.0
- matplotlib >= 3.4.0

### Python Version

Tested with Python 3.8+

## 📊 Methodology Highlights

### Part 1: Machine Learning Pipeline

1. **Feature Selection Rationale:** Reduces overfitting by eliminating 50% of low-importance features using median-threshold ExtraTreesRegressor
2. **Hyperparameter Tuning:** GridSearchCV over 243 combinations (5-fold CV)
3. **Validation Strategy:** Out-of-fold predictions to prevent leakage
4. **Domain Shift Analysis:** ROC-based classifier to detect train/eval distribution differences

### Part 2: Rule Discovery Process

1. **Tree Exploration:** DecisionTreeRegressor to identify splitting feature and thresholds
2. **Threshold Simplification:** Rounding to 0.1 precision for deployable boundaries
3. **Per-Region Regression:** LinearRegression fitted independently in each region
4. **Validation:** Numerical parity check (reconstruction error ~10⁻¹⁵)

## 📈 Visualizations

The project generates comprehensive visualizations for analysis:

- **Part 1:** Target distribution, domain shift ROC curve
- **Part 2:** Decision tree structure, coefficient contributions, feature distributions, actual vs predicted comparisons

## 🔬 Technical Details

### Hyperparameter Search Space

```python
param_grid = {
    'reg__learning_rate': [0.03, 0.05, 0.08],
    'reg__max_leaf_nodes': [31, 63, 127],
    'reg__min_samples_leaf': [10, 20, 50],
    'reg__max_iter': [300, 600, 900],
    'reg__l2_regularization': [0.0, 0.1, 1.0]
}
# Total: 243 combinations
```

### Reproducibility

All experiments use fixed random seed: `RANDOM_SEED = 42`

## ⚠️ AI Tool Disclosure

Portions of code were edited with assistance from OpenAI ChatGPT (GPT-5.2 Thinking) for:
- Debugging assistance
- Code refactoring suggestions
- Framework interface compliance verification

All logic was independently reviewed and tested by the author.

## 📄 Documentation

Complete technical documentation available in `Report_ML_W26.pdf` covering:
- Detailed methodology for both parts
- Mathematical formulations
- Validation strategies
- Performance analysis
- Scientific references

## 🏆 Key Achievements

✅ **Part 1:** Feature selection improved MAE by 5.37% and RMSE by 5.01%  
✅ **Part 2:** Achieved perfect reconstruction (R² = 1.0) with only 4 simple rules  
✅ **Edge Deployment:** Zero ML dependencies at runtime  
✅ **Reproducibility:** Complete pipeline with fixed random seeds

## 📧 Contact

**Author:** Umang Dholakiya  
**Institution:** OTH Amberg-Weiden  
**Course:** Machine Learning W25/26

## 📝 License

This project is submitted as coursework for Machine Learning Winter 2025 at OTH Amberg-Weiden.

---

**Project Completion Date:** January 2026  
**Dataset ID:** 37
