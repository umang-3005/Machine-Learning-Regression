# Hyperparameter Tuning Experiment for target01 Prediction
# This script performs GridSearchCV to find optimal hyperparameters for
# HistGradientBoostingRegressor. Results are saved to a leaderboard CSV.

# AI TOOL DISCLOSURE:
# Portions of this file were edited with assistance from OpenAI ChatGPT (GPT-5.2 Thinking),
# used for debugging, refactoring suggestions, and ensuring compliance with the provided framework interface.
# All logic was reviewed and tested locally by the me.
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.base import clone

# Unique identifier for the dataset
ID = 37
# Random seed for reproducibility
RANDOM_SEED = 42

# Step 1: Load Data
X = pd.read_csv(f"problem_37/dataset_{ID}.csv")       # Training features
y = pd.read_csv(f"problem_37/target_{ID}.csv")["target01"]  # Target values
X_eval = pd.read_csv(f"problem_37/EVAL_{ID}.csv")     # Evaluation features (unlabeled)

# Verify data alignment
assert len(X) == len(y), "Mismatch: X and y row counts differ."
assert list(X.columns) == list(X_eval.columns), "Train/EVAL feature columns do not match (names/order)."

print("X:", X.shape, "| y:", y.shape, "| X_eval:", X_eval.shape)


# Step 2: Base Model Definition
# Pipeline with median imputation and HistGradientBoostingRegressor
# Hyperparameters will be tuned via GridSearchCV
base_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),  # Handle missing values
    ("reg", HistGradientBoostingRegressor(
        random_state=RANDOM_SEED,
        early_stopping=False    # Disabled for reproducibility
    ))
])

print(base_model)  # Display pipeline structure



# Step 3: Define Hyperparameter Grid
# Grid of hyperparameters to search over
param_grid = {
    "reg__learning_rate": [0.03, 0.05, 0.08],      # Step size for gradient descent
    "reg__max_leaf_nodes": [31, 63, 127],          # Tree complexity
    "reg__min_samples_leaf": [10, 20, 50],         # Regularization via min samples
    "reg__max_iter": [300, 600, 900],              # Number of boosting iterations
    "reg__l2_regularization": [0.0, 0.1, 1.0],     # L2 regularization strength
}

# Calculate total number of parameter combinations
n = 1
for k, v in param_grid.items():
    n *= len(v)
print("Total parameter combinations:", n)

cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

scoring = {
    "rmse": "neg_root_mean_squared_error",
    "mae": "neg_mean_absolute_error",
    "r2": "r2"
}

# Step 4: Run GridSearchCV
grid = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    scoring=scoring,
    refit="rmse",          # Select best model based on RMSE
    cv=cv,
    n_jobs=-1,             # Use all available CPU cores
    verbose=2,             # Print progress during fitting
    return_train_score=False
)

# Fit all parameter combinations and find the best
grid.fit(X, y)

print("\nBest params:", grid.best_params_)
print("Best CV RMSE:", -grid.best_score_)


# Step 5: Process and Save Results
results = pd.DataFrame(grid.cv_results_)

# Convert negative metrics to positive for readability
# (sklearn uses neg_* scorers for optimization purposes)
results["mean_rmse"] = -results["mean_test_rmse"]
results["mean_mae"]  = -results["mean_test_mae"]
results["mean_r2"]   = results["mean_test_r2"]

# Keep a clean table
keep_cols = [
    "rank_test_rmse",
    "mean_rmse", "std_test_rmse",
    "mean_mae",  "std_test_mae",
    "mean_r2",   "std_test_r2",
    "params"
]
leaderboard = results[keep_cols].sort_values("rank_test_rmse")

print(leaderboard.head(15).to_string(index=False))

# Save for reporting to main ML directory
leaderboard_path = f"gridsearch_target01_{ID}_leaderboard.csv"
leaderboard.to_csv(leaderboard_path, index=False)
print("\nSaved:", leaderboard_path)


# Step 6: Out-of-Fold Validation with Best Model
# Use best parameters to generate OOF predictions for honest evaluation
best_model = grid.best_estimator_
kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

n_train = len(X)
n_eval = len(X_eval)

# Arrays to store OOF and EVAL predictions
oof_pred = np.empty(n_train, dtype=float)                         # One pred per training sample
eval_fold_preds = np.zeros((kf.get_n_splits(), n_eval), dtype=float)  # 5 predictions per EVAL sample

fold_rows = []

for fold, (tr_idx, va_idx) in enumerate(kf.split(X), start=1):
    m = clone(best_model)
    X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
    X_va, y_va = X.iloc[va_idx], y.iloc[va_idx]

    m.fit(X_tr, y_tr)

    va_pred = m.predict(X_va)
    oof_pred[va_idx] = va_pred

    eval_fold_preds[fold - 1, :] = m.predict(X_eval)

    fold_mae  = mean_absolute_error(y_va, va_pred)
    fold_rmse = root_mean_squared_error(y_va, va_pred)
    fold_r2   = r2_score(y_va, va_pred)

    fold_rows.append({"fold": fold, "mae": fold_mae, "rmse": fold_rmse, "r2": fold_r2})
    print(f"Fold {fold}: MAE={fold_mae:.6f} | RMSE={fold_rmse:.6f} | R2={fold_r2:.6f}")

# OOF metrics (main estimate)
oof_mae  = mean_absolute_error(y, oof_pred)
oof_rmse = root_mean_squared_error(y, oof_pred)
oof_r2   = r2_score(y, oof_pred)

print("\n=== TUNED MODEL: OUT-OF-FOLD METRICS ===")
print(f"OOF MAE :  {oof_mae:.6f}")
print(f"OOF RMSE:  {oof_rmse:.6f}")
print(f"OOF R2  :  {oof_r2:.6f}")

fold_df = pd.DataFrame(fold_rows)
print("\n=== FOLD METRICS (mean ± std) ===")
print(f"MAE :  {fold_df['mae'].mean():.6f} ± {fold_df['mae'].std():.6f}")
print(f"RMSE:  {fold_df['rmse'].mean():.6f} ± {fold_df['rmse'].std():.6f}")
print(f"R2  :  {fold_df['r2'].mean():.6f} ± {fold_df['r2'].std():.6f}")

# Eval stability
eval_mean = eval_fold_preds.mean(axis=0)
eval_std  = eval_fold_preds.std(axis=0)

print("\n=== EVAL PREDICTION STABILITY (std across folds per row) ===")
print(pd.Series(eval_std).describe(percentiles=[0.5, 0.9, 0.95, 0.99]))

# Save averaged submission to main ML directory
sub_path = f"EVAL_target01_{ID}.csv"
pd.DataFrame({"target01": eval_mean}).to_csv(sub_path, index=False)
print("\nSaved tuned averaged submission:", sub_path)
print("Preview:\n", pd.read_csv(sub_path).head())

