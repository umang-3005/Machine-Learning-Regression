"""
Experiment 1.1: Regression Pipeline with Feature Selection
============================================================
This script performs regression analysis using HistGradientBoostingRegressor
with ExtraTrees-based feature selection on dataset_37. 

Pipeline Steps:
1. Load datasets (train, target, eval)
2. Data preprocessing and sanity checks
3. Target variable analysis
4. Model training with feature selection
5. Cross-validation evaluation
6. Generate predictions and save submission
7. Domain shift analysis (ROC curve)
8. Baseline comparison (without feature selection)
"""

# AI TOOL DISCLOSURE:
# Portions of this file were edited with assistance from OpenAI ChatGPT (GPT-5.2 Thinking),
# used for debugging, refactoring suggestions, and ensuring compliance with the provided framework interface.
# All logic was reviewed and tested locally by the me.


# IMPORTS
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, cross_validate, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_curve, auc as compute_auc
from sklearn.linear_model import LogisticRegression

# CONFIGURATION
ID = 37
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# File paths - using problem_37 folder, save outputs to main ML directory
BASE_DIR = os.path.dirname(__file__)
MAIN_DIR = os.path.dirname(BASE_DIR)  # Main ML directory
DATASET_DIR = os.path.join(MAIN_DIR, "problem_37")
RESULTS_DIR = MAIN_DIR  # Save all outputs to main ML directory

X_PATH = os.path.join(DATASET_DIR, f"dataset_{ID}.csv")
Y_PATH = os.path.join(DATASET_DIR, f"target_{ID}.csv")
EVAL_PATH = os.path.join(DATASET_DIR, f"EVAL_{ID}.csv")

# DATA LOADING
print("=" * 80)
print("LOADING DATASETS")
print("=" * 80)

X = pd.read_csv(X_PATH)
y_df = pd.read_csv(Y_PATH)
X_eval = pd.read_csv(EVAL_PATH)

# Extract target variable
y = y_df["target01"].to_numpy(dtype=float)

print(f"X shape     : {X.shape}")
print(f"y shape     : {y.shape}")
print(f"X_eval shape: {X_eval.shape}")
print(f"Target file columns: {list(y_df.columns)}")

# DATA PREPROCESSING - SANITY CHECKS
print("\n" + "=" * 80)
print("DATA PREPROCESSING - SANITY CHECKS")
print("=" * 80)

# Validate data consistency
assert len(X) == len(y), "Row count mismatch between X and y."
assert list(X.columns) == list(X_eval.columns), "Train/EVAL columns mismatch (names or order)."

print(f"Any NaNs in X?         {X.isna().any().any()}")
print(f"Any NaNs in X_eval? {X_eval.isna().any().any()}")
print(f"Any NaNs in y?        {np.isnan(y).any()}")
print(f"Duplicate rows in X:   {X.duplicated().sum()}")
print(f"Constant columns in X: {(X.nunique(dropna=False) <= 1).sum()}")
print("\n✅ Sanity checks passed.")

# TARGET VARIABLE STATISTICS AND HISTOGRAM
print("\n" + "=" * 80)
print("TARGET VARIABLE STATISTICS")
print("=" * 80)

target_series = pd.Series(y)
print(target_series.describe())

# Plot histogram of target variable
plt.figure(figsize=(10, 6))
plt.hist(y, bins=40, edgecolor='black', alpha=0.7)
plt.title("target01 distribution (train)", fontsize=14, fontweight='bold')
plt.xlabel("target01", fontsize=12)
plt.ylabel("count", fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()
hist_path = "target01_histogram.png"
plt.savefig(hist_path, dpi=150)
plt.show()
print(f"Histogram saved: {hist_path}")

# MODEL PIPELINE DEFINITION
print("\n" + "=" * 80)
print("MODEL PIPELINE DEFINITION")
print("=" * 80)

# Feature selector using ExtraTreesRegressor
selector_model = ExtraTreesRegressor(
    n_estimators=400,
    random_state=RANDOM_SEED,
    n_jobs=-1
)

# Full pipeline with imputation, feature selection, and regression
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("selector", SelectFromModel(
        estimator=selector_model,
        threshold="median"  # Keeps approximately 50% of features
    )),
    ("reg", HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_leaf_nodes=31,
        min_samples_leaf=10,
        max_iter=900,
        early_stopping=False,
        random_state=RANDOM_SEED
    ))
])

print("Pipeline components:")
print(model)

# FEATURE SELECTION ANALYSIS
print("\n" + "=" * 80)
print("FEATURE SELECTION ANALYSIS")
print("=" * 80)

# Prepare data for feature selection analysis
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)

# Fit the ExtraTreesRegressor for feature importance
selector_model.fit(X_imputed, y)

# Extract feature importances
importances = selector_model.feature_importances_
feature_names = X.columns.tolist()

# Determine selected features (importance > median)
median_importance = np.median(importances)
selected_mask = importances > median_importance
selected_indices = np.where(selected_mask)[0]
selected_features = [feature_names[i] for i in selected_indices]
discarded_features = [feat for feat in feature_names if feat not in selected_features]

# Summary counts
print(f"Total features:      {len(feature_names)}")
print(f"Selected features:  {len(selected_features)} (threshold = median importance)")
print(f"Discarded features: {len(discarded_features)}")

# Importance statistics
print(f"\nIMPORTANCE STATISTICS:")
print(f"  Median (threshold): {median_importance:.6f}")
print(f"  Mean:                {np.mean(importances):.6f}")
print(f"  Min:                {np.min(importances):.6f}")
print(f"  Max:                {np.max(importances):.6f}")
print(f"  Std:                {np.std(importances):.6f}")

# Top 10 most important features
sorted_indices = np.argsort(importances)[::-1]
print(f"\nTOP 10 MOST IMPORTANT FEATURES:")
for rank, idx in enumerate(sorted_indices[:10], start=1):
    status = "✓" if selected_mask[idx] else "✗"
    print(f"  {rank: 2d}. {feature_names[idx]: <12} (importance: {importances[idx]:.6f}) {status}")

# List of all selected features (compact format)
print(f"\nSELECTED FEATURES ({len(selected_features)}):")
# Print in rows of 8 features each for readability
for i in range(0, len(selected_features), 8):
    chunk = selected_features[i:i+8]
    print(f"  {', '.join(chunk)}")

# CROSS-VALIDATION EVALUATION
print("\n" + "=" * 80)
print("CROSS-VALIDATION EVALUATION (WITH FEATURE SELECTION)")
print("=" * 80)

# Define cross-validation strategy
cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

# Define scoring metrics
scoring = {
    "mae": "neg_mean_absolute_error",
    "rmse": "neg_root_mean_squared_error",
    "r2": "r2"
}

# Perform cross-validation
cv_out = cross_validate(
    model, X, y,
    cv=cv,
    scoring=scoring,
    n_jobs=-1,
    return_train_score=False
)

# Extract and display results
mae = -cv_out["test_mae"]
rmse = -cv_out["test_rmse"]
r2 = cv_out["test_r2"]

print(f"CV MAE :   {mae.mean():.6f} ± {mae.std():.6f}")
print(f"CV RMSE: {rmse.mean():.6f} ± {rmse.std():.6f}")
print(f"CV R2  :  {r2.mean():.6f} ± {r2.std():.6f}")

# TRAIN FINAL MODEL AND GENERATE PREDICTIONS
print("\n" + "=" * 80)
print("TRAINING FINAL MODEL AND GENERATING PREDICTIONS")
print("=" * 80)

# Train on full training data
final_model = model.fit(X, y)
eval_pred = final_model.predict(X_eval)

# Save submission file to main ML directory
sub_path = f"EVAL_target01_{ID}.csv"
pd.DataFrame({"target01": eval_pred}).to_csv(sub_path, index=False)
print(f"Saved submission:  {sub_path}")
print(pd.read_csv(sub_path).head())

# STATISTICS COMPARISON:  TRAIN TRUE vs EVAL PRED
print("\n" + "=" * 80)
print("STATISTICS COMPARISON:  TRAIN TRUE vs EVAL PRED")
print("=" * 80)


def display_quantile_stats(array, name):
    """Display descriptive statistics and quantiles for an array."""
    series = pd.Series(array)
    print(f"\n{name}")
    print(series.describe())
    quantiles = series.quantile([0.01, 0.05, 0.5, 0.95, 0.99]).to_dict()
    print(f"Quantiles (q01, q05, q50, q95, q99): {quantiles}")


# Display statistics for comparison
display_quantile_stats(y, "Train TRUE y (target01)")
display_quantile_stats(eval_pred, "EVAL PRED")

# OUT-OF-FOLD PREDICTIONS AND METRICS
print("\n" + "=" * 80)
print("OUT-OF-FOLD PREDICTIONS AND METRICS")
print("=" * 80)

oof = np.empty(len(X), dtype=float)

for fold, (tr_idx, va_idx) in enumerate(cv.split(X), start=1):
    fold_model = clone(model)
    fold_model.fit(X.iloc[tr_idx], y[tr_idx])
    oof[va_idx] = fold_model.predict(X.iloc[va_idx])

display_quantile_stats(oof, "Train OOF PRED")

print("\nOOF Metrics:")
print(f"  MAE :   {mean_absolute_error(y, oof):.6f}")
print(f"  RMSE:  {np.sqrt(mean_squared_error(y, oof)):.6f}")
print(f"  R2  :  {r2_score(y, oof):.6f}")

# DOMAIN SHIFT ANALYSIS (ROC CURVE)
print("\n" + "=" * 80)
print("DOMAIN SHIFT ANALYSIS")
print("=" * 80)

# Combine train and eval data
X_all = pd.concat([X, X_eval], axis=0, ignore_index=True)
domain_y = np.r_[np.zeros(len(X)), np.ones(len(X_eval))]  # 0 = train, 1 = eval

# Domain classifier
domain_clf = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("lr", LogisticRegression(max_iter=2000))
])

# Cross-validated AUC score
domain_auc_cv = cross_val_score(domain_clf, X_all, domain_y, cv=5, scoring="roc_auc").mean()
print(f"Domain AUC (CV 5-fold): {domain_auc_cv:.8f}")

# Fit on full data for ROC curve
domain_clf.fit(X_all, domain_y)
y_proba = domain_clf.predict_proba(X_all)[:, 1]

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(domain_y, y_proba)
roc_auc = compute_auc(fpr, tpr)

# Plot ROC curve
fig, ax = plt.subplots(figsize=(7, 7))
ax.plot(fpr, tpr, color='darkorange', lw=2.5, label=f'ROC Curve (AUC = {roc_auc:.4f})')
ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier (AUC = 0.5000)')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate (EVAL classified as Train)', fontsize=12)
ax.set_ylabel('True Positive Rate (EVAL correctly identified)', fontsize=12)
ax.set_title('ROC Curve:  Domain Shift Detection', fontsize=14, fontweight='bold')
ax.legend(loc="lower right", fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
roc_path = "domain_shift_roc.png"
plt.savefig(roc_path, dpi=150)
plt.show()

print(f"ROC curve saved: {roc_path}")

# BASELINE COMPARISON (WITHOUT FEATURE SELECTION)
print("\n" + "=" * 80)
print("BASELINE COMPARISON (WITHOUT FEATURE SELECTION)")
print("=" * 80)

# Baseline pipeline without feature selection
baseline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("reg", HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_leaf_nodes=31,
        min_samples_leaf=10,
        max_iter=900,
        early_stopping=False,
        random_state=RANDOM_SEED
    ))
])

# Cross-validate baseline
base_out = cross_validate(
    baseline, X, y,
    cv=cv,
    scoring=scoring,
    n_jobs=-1,
    return_train_score=False
)

base_mae = -base_out["test_mae"]
base_rmse = -base_out["test_rmse"]
base_r2 = base_out["test_r2"]

print("BASELINE (no feature selector):")
print(f"  CV MAE :  {base_mae.mean():.6f} ± {base_mae.std():.6f}")
print(f"  CV RMSE: {base_rmse.mean():.6f} ± {base_rmse.std():.6f}")
print(f"  CV R2  :  {base_r2.mean():.6f} ± {base_r2.std():.6f}")

# FINAL COMPARISON SUMMARY
print("\n" + "=" * 80)
print("FINAL COMPARISON SUMMARY")
print("=" * 80)

# Calculate improvements
mae_improvement = ((base_mae.mean() - mae.mean()) / base_mae.mean() * 100)
rmse_improvement = ((base_rmse.mean() - rmse.mean()) / base_rmse.mean() * 100)
r2_improvement = ((r2.mean() - base_r2.mean()) / base_r2.mean() * 100)

print(f"\n{'Metric':<10} {'With Selection':<22} {'Without Selection':<22} {'Improvement':<12}")
print("-" * 70)
print(f"{'MAE':<10} {mae.mean():.6f} ± {mae.std():.6f}    {base_mae.mean():.6f} ± {base_mae.std():.6f}    {mae_improvement:+.2f}%")
print(f"{'RMSE':<10} {rmse.mean():.6f} ± {rmse.std():.6f}    {base_rmse.mean():.6f} ± {base_rmse.std():.6f}    {rmse_improvement:+.2f}%")
print(f"{'R2':<10} {r2.mean():.6f} ± {r2.std():.6f}    {base_r2.mean():.6f} ± {base_r2.std():.6f}    {r2_improvement:+.2f}%")

print("\n" + "=" * 80)
print("EXPERIMENT COMPLETED SUCCESSFULLY")
print("=" * 80)


