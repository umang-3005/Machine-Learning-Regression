Here’s the full content formatted and ready to save as **`README.md`** in your repo:

```markdown
# Machine Learning Regression Challenge 

## 📋 Project Overview

This project addresses a two-part tabular regression challenge with distinct objectives and deployment constraints:

- **Part 1:** Build a high-performance ML model to predict the continuous target `target01` from 273 features
- **Part 2:** Reverse-engineer a rule-based system for `target02` that must run on edge devices without ML inference capabilities

**Dataset Specifications:**
- Training samples: 10,000
- Features: 273 (feat_1 through feat_273)
- Two regression targets: `target01` and `target02`

## 🗂️ Repository Structure

ML_37/
├── Part1.py                      # Main pipeline for target01 prediction
├── Part2.py                      # Rule discovery and validation for target02
├── parameter_experiment.py       # Hyperparameter tuning experiments
├── framework_37.py               # Deployment-ready rule engine for target02
├── requirements.txt              # Python dependencies
├── EVAL_target01_37.csv          # Final predictions for target01
├── Report_ML_W26.pdf             # Complete technical report
├── pra_mal_w25.pdf               # Project assignment specification
└── problem_37/                   # Dataset directory (not included)
    ├── dataset_37.csv            # Training features
    ├── target_37.csv             # Training targets
    └── EVAL_37.csv               # Evaluation features

---

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
````

### Performance Metrics

| Metric | With Feature Selection | Without Selection   | Improvement |
| ------ | ---------------------- | ------------------- | ----------- |
| MAE    | 0.070452 ± 0.004699    | 0.074450 ± 0.005234 | -5.37%      |
| RMSE   | 0.088561 ± 0.005073    | 0.093234 ± 0.005229 | -5.01%      |
| R²     | 0.859576 ± 0.017345    | 0.844368 ± 0.019170 | +1.80%      |

### Running Part 1

```bash
python Part1.py
```

**Outputs:**

* `EVAL_target01_37.csv` - Predictions for evaluation set
* `target01_histogram.png` - Target distribution visualization
* `domain_shift_roc.png` - Domain shift analysis

---

## 🧩 Part 2: Rule-Based Prediction for target02

### Problem Statement

Predict `target02` using only simple conditional rules and arithmetic operations (no ML at runtime) for edge device deployment.

### Discovered Rule System

* **Gating feature:** `feat_132`
* **Predictor features:** `feat_108`, `feat_116`, `feat_255`
* **Thresholds:** 0.2, 0.5, 0.7

### Rule Formulas

```python
if feat_132 <= 0.2:
    target02 = 1.35 * feat_108 + 1.75 * feat_116 - 0.75 * feat_255
elif feat_132 <= 0.5:
    target02 = 0.35 * feat_108 - 0.45 * feat_116 + 0.55 * feat_255
elif feat_132 <= 0.7:
    target02 = 0.15 * feat_108 + 0.85 * feat_116 - 1.95 * feat_255
else:  # feat_132 > 0.7
    target02 = 1.85 * feat_108 - 1.75 * feat_116 - 0.75 * feat_255
```

### Validation Results

| Region | Condition  | Samples | R²  | Max Error |
| ------ | ---------- | ------- | --- | --------- |
| 1      | ≤ 0.2      | 1,999   | 1.0 | ~10⁻¹⁵    |
| 2      | (0.2, 0.5] | 2,951   | 1.0 | ~10⁻¹⁵    |
| 3      | (0.5, 0.7] | 2,011   | 1.0 | ~10⁻¹⁵    |
| 4      | > 0.7      | 3,039   | 1.0 | ~10⁻¹⁵    |

### Running Part 2

```bash
python Part2.py
python framework_37.py --eval_file_path problem_37/EVAL_37.csv
```

**Outputs:**

* `rules_target02_37.json` - Extracted rule parameters
* `part2_decision_tree.png` - Tree visualization
* `part2_coefficient_contributions.png` - Region-wise coefficients
* `part2_feature_distribution.png` - Split feature analysis
* `part2_actual_vs_predicted.png` - Reconstruction validation

---

## 🛠️ Installation

```bash
pip install -r requirements.txt
```

**Dependencies:**

* numpy >= 1.21.0
* pandas >= 1.3.0
* scikit-learn >= 1.0.0
* matplotlib >= 3.4.0

**Python Version:** 3.8+

---

## 📊 Methodology Highlights

### Part 1: Machine Learning Pipeline

1. **Data loading and sanity checks** – Verify shapes, columns, missing values, duplicates.
2. **Preprocessing and feature selection** – Defensive median imputation, ExtraTrees-based selection inside the pipeline.
3. **Model choice:** HistGradientBoostingRegressor with tuned hyperparameters.
4. **Hyperparameter tuning:** 243 combinations via `parameter_experiment.py` using 5-fold CV.
5. **Cross-validation:** Out-of-fold predictions for unbiased estimates.
6. **Baseline comparison:** Pipeline without feature selection to justify improvement.

### Part 2: Rule-Based Prediction

* Only three features influence `target02` with four piecewise-linear regimes.
* Implementation in `framework_37.py` is edge-device compatible: **no ML dependencies at runtime**.

---

## 📈 Visualizations

* **Part 1:** Target distribution, domain shift ROC curve
* **Part 2:** Decision tree, coefficient contributions, feature distributions, actual vs predicted comparisons

---

## 🔬 Technical Details

* Hyperparameter Search Space:

```python
param_grid = {
    'reg__learning_rate': [0.03, 0.05, 0.08],
    'reg__max_leaf_nodes': [31, 63, 127],
    'reg__min_samples_leaf': [10, 20, 50],
    'reg__max_iter': [300, 600, 900],
    'reg__l2_regularization': [0.0, 0.1, 1.0]
}
```

* Reproducibility: `RANDOM_SEED = 42`

---

## ⚠️ AI Tool Disclosure

Portions of code were assisted by OpenAI ChatGPT (GPT-5.2 Thinking) for:

* Debugging
* Refactoring suggestions
* Interface compliance

All logic was **independently reviewed and tested** by the author.

---

## 📄 Documentation

Complete technical report available in `Report_ML_W26.pdf`.

---

## 🏆 Key Achievements

* **Part 1:** Feature selection improved MAE by 5.37% and RMSE by 5.01%
* **Part 2:** Perfect reconstruction (R² = 1.0) with 4 simple rules
* **Edge Deployment:** Zero ML dependencies at runtime
* **Reproducibility:** Fixed random seeds across pipeline

```

---

If you want, I can also **draft a plan for file-by-file commits** so that later, when you update **comments in each file**, each file can have its **own descriptive commit message**, keeping your Git history clean.  

Do you want me to do that next?
```
