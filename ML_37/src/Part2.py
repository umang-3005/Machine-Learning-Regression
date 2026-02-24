import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from io import StringIO
from sklearn.tree import DecisionTreeRegressor, _tree, export_text
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# AI TOOL DISCLOSURE:
# Portions of this file were edited with assistance from OpenAI ChatGPT (GPT-5.2 Thinking),
# used for debugging, refactoring suggestions, and ensuring compliance with the provided framework interface.
# All logic was reviewed and tested locally by the me.

ID = 37
X_PATH = f"problem_37/dataset_{ID}.csv"
Y_PATH = f"problem_37/target_{ID}.csv"

# These are the features for final rule uses
SPLIT_FEAT_IDX = 132
LIN_FEAT_IDXS = [108, 116, 255]


def extract_thresholds_for_feature(dt_model, feat_idx: int):
    """Return raw thresholds used by the tree for a given feature index."""
    tree = dt_model.tree_
    thr = []
    for node_id in range(tree.node_count):
        if tree.feature[node_id] == feat_idx:
            thr.append(float(tree.threshold[node_id]))
    return thr


def simplify_thresholds(thresholds):
    """
    In case: tree gives ~0.7 and ~0.71; we group them by rounding to 1 decimal.
    Returns sorted unique thresholds, e.g. [0.2, 0.5, 0.7]
    """
    if not thresholds:
        return []
    simplified = sorted(set([round(t, 1) for t in thresholds]))
    return simplified


def fit_piecewise_linear(X_np, y, split_idx: int, split_points, lin_idxs):
    """
    Fit LinearRegression in each region defined by split_points.
    Regions:
      (-inf, t1], (t1, t2], (t2, t3], (t3, +inf)
    Returns list of region dicts with coefs and intercept.
    """
    fsplit = X_np[:, split_idx]
    X_lin = X_np[:, lin_idxs]

    # build masks for regions
    t1, t2, t3 = split_points
    masks = [
        fsplit <= t1,
        (fsplit > t1) & (fsplit <= t2),
        (fsplit > t2) & (fsplit <= t3),
        fsplit > t3,
    ]

    regions = []
    for k, mask in enumerate(masks, start=1):
        Xk = X_lin[mask]
        yk = y[mask]
        lr = LinearRegression().fit(Xk, yk)
        yk_pred = lr.predict(Xk)

        regions.append({
            "region": k,
            "n": int(mask.sum()),
            "coef": lr.coef_.astype(float).tolist(),      # aligned with lin_idxs
            "intercept": float(lr.intercept_),
            "r2_in_region": float(r2_score(yk, yk_pred)),
        })

    return regions


def predict_piecewise(X_np, split_idx: int, split_points, lin_idxs, regions):
    """Apply the learned piecewise linear rule to X_np."""
    fsplit = X_np[:, split_idx]
    X_lin = X_np[:, lin_idxs]
    t1, t2, t3 = split_points

    preds = np.empty(X_np.shape[0], dtype=float)

    masks = [
        fsplit <= t1,
        (fsplit > t1) & (fsplit <= t2),
        (fsplit > t2) & (fsplit <= t3),
        fsplit > t3,
    ]

    for i, mask in enumerate(masks):
        coef = np.array(regions[i]["coef"], dtype=float)
        intercept = float(regions[i]["intercept"])
        preds[mask] = X_lin[mask] @ coef + intercept

    return preds


def plot_decision_tree(dt_model, X_columns, results_dir: str):
    """Plot decision tree visualization for target02 with depth 3."""
    # results_dir points to main ML directory, no need to create
    
    from sklearn import tree
    
    fig, ax = plt.subplots(figsize=(20, 12))
    
    # Limit tree plot to depth 3 for readability
    tree.plot_tree(dt_model, 
                   feature_names=X_columns,
                   filled=True,
                   rounded=True,
                   fontsize=9,
                   ax=ax,
                   proportion=True,
                   max_depth=3)
    
    fig.suptitle('Decision Tree for target02 (First 3 Levels)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    out_path = os.path.join(results_dir, 'part2_decision_tree.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()


def plot_coefficient_contributions(regions, lin_feat_idxs, results_dir: str):
    """Plot coefficient contributions for each region (4-panel layout)."""
    # results_dir points to main ML directory, no need to create
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Part 2: Coefficient Contributions by Region', fontsize=14, fontweight='bold')
    axes = axes.flatten()
    
    feat_names = [f"feat_{i}" for i in lin_feat_idxs]
    colors_pos = '#2ecc71'  # Green for positive
    colors_neg = '#e74c3c'  # Red for negative
    
    for idx, region in enumerate(regions):
        ax = axes[idx]
        coefs = region["coef"]
        
        # Create bars with different colors for positive/negative
        bar_colors = [colors_pos if c >= 0 else colors_neg for c in coefs]
        bars = ax.bar(feat_names, coefs, color=bar_colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar, coef in zip(bars, coefs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + (0.1 if height > 0 else -0.15),
                    f'{coef:.2f}', ha='center', va='bottom' if height > 0 else 'top', 
                    fontsize=10, fontweight='bold')
        
        ax.axhline(0, color='black', linestyle='-', linewidth=0.8)
        ax.set_ylabel('Coefficient Value', fontweight='bold')
        ax.set_title(f'Region {region["region"]} (feat_132 ≤ {["0.2", "0.5", "0.7", ">0.7"][idx]})\nR²={region["r2_in_region"]:.4f}, n={region["n"]}', 
                     fontweight='bold')
        ax.grid(alpha=0.3, axis='y')
        ax.set_ylim([min(coefs) - 0.5, max(coefs) + 0.5])
    
    plt.tight_layout()
    out_path = os.path.join(results_dir, 'part2_coefficient_contributions.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()


def plot_feature_distribution_and_regions(X_np, split_idx, split_points, results_dir: str):
    """Plot distribution of split feature with threshold lines and region distribution."""
    # results_dir points to main ML directory, no need to create
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Part 2: Feature {split_idx} Distribution & Region Split', fontsize=14, fontweight='bold')
    
    fsplit = X_np[:, split_idx]
    
    # Plot 1: Distribution with threshold lines
    ax = axes[0]
    ax.hist(fsplit, bins=60, color='steelblue', alpha=0.7, edgecolor='black')
    
    colors_thresh = ['red', 'orange', 'purple']
    for t, color in zip(split_points, colors_thresh):
        ax.axvline(t, color=color, linestyle='--', linewidth=2.5, label=f'Threshold: {t}')
    
    ax.set_xlabel(f'Feature {split_idx}', fontweight='bold', fontsize=11)
    ax.set_ylabel('Frequency', fontweight='bold', fontsize=11)
    ax.set_title(f'Distribution of feat_{split_idx} with Simplified Thresholds')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, axis='y')
    
    # Plot 2: Sample distribution across regions
    ax = axes[1]
    t1, t2, t3 = split_points
    
    masks = [
        fsplit <= t1,
        (fsplit > t1) & (fsplit <= t2),
        (fsplit > t2) & (fsplit <= t3),
        fsplit > t3,
    ]
    
    # Create threshold-based labels for x-axis
    threshold_labels = [f'≤{t1}', f'{t1}-{t2}', f'{t2}-{t3}', f'>{t3}']
    region_counts = [mask.sum() for mask in masks]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    bars = ax.bar(range(len(threshold_labels)), region_counts, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.set_xlabel('feat_132 Threshold Ranges', fontweight='bold', fontsize=11)
    ax.set_ylabel('Sample Count', fontweight='bold', fontsize=11)
    ax.set_title('Sample Distribution Across Regions')
    ax.set_xticks(range(len(threshold_labels)))
    ax.set_xticklabels(threshold_labels, fontsize=10, fontweight='bold')
    
    for bar, count in zip(bars, region_counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 50,
                f'{int(count)}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    out_path = os.path.join(results_dir, 'part2_feature_distribution.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()


def plot_actual_vs_predicted_histogram(y, y_pred, results_dir: str):
    """Plot histogram comparison of actual vs predicted values."""
    # results_dir points to main ML directory, no need to create
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Part 2: Actual vs Predicted Distribution', fontsize=14, fontweight='bold')
    
    # Plot 1: Histogram comparison
    ax = axes[0]
    ax.hist(y, bins=50, alpha=0.6, label='Actual target02', color='steelblue', edgecolor='black')
    ax.hist(y_pred, bins=50, alpha=0.6, label='Predicted (simple rules)', color='orange', edgecolor='black')
    ax.set_xlabel('target02 value', fontweight='bold', fontsize=11)
    ax.set_ylabel('Frequency', fontweight='bold', fontsize=11)
    ax.set_title('Histogram: Actual vs Predicted')
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(alpha=0.3, axis='y')
    
    # Plot 2: Scatter plot with perfect fit line
    ax = axes[1]
    ax.scatter(y, y_pred, alpha=0.4, s=15, color='steelblue', edgecolor='none')
    min_val, max_val = y.min(), y.max()
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2.5, label='Perfect fit')
    ax.set_xlabel('Actual target02', fontweight='bold', fontsize=11)
    ax.set_ylabel('Predicted target02', fontweight='bold', fontsize=11)
    ax.set_title(f'Actual vs Predicted (R² = {r2_score(y, y_pred):.4f})')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    out_path = os.path.join(results_dir, 'part2_actual_vs_predicted.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()


def main():
    # Save all outputs to main ML directory (current working directory)
    results_dir = "."
    
    # 1) Load data
    X = pd.read_csv(X_PATH)
    y = pd.read_csv(Y_PATH)["target02"].to_numpy(dtype=float)
    X_np = X.to_numpy(dtype=float)

    print(f"Loaded X: {X_np.shape}, y: {y.shape}")

    # 2) Fit tree (just to confirm structure)
    dt = DecisionTreeRegressor(max_depth=8, min_samples_leaf=50, random_state=37)
    dt.fit(X_np, y)

    used_idx = sorted(set([f for f in dt.tree_.feature if f != _tree.TREE_UNDEFINED]))
    used_names = [X.columns[i] for i in used_idx]
    print(f"Tree used features: {used_names} (idx={used_idx})")

    # 3) Extract and simplify thresholds for feat_132
    raw_thr = extract_thresholds_for_feature(dt, SPLIT_FEAT_IDX)
    simp_thr = simplify_thresholds(raw_thr)

    # Expect 3 split points => 4 regions
    if len(simp_thr) != 3:
        raise RuntimeError(f"Expected 3 simplified thresholds, got {simp_thr}")

    print(f"Raw thresholds on feat_{SPLIT_FEAT_IDX}: {sorted(set(round(t, 4) for t in raw_thr))}")
    print(f"Simplified split points: {simp_thr}")

    # 4) Fit linear regression inside each region
    regions = fit_piecewise_linear(X_np, y, SPLIT_FEAT_IDX, simp_thr, LIN_FEAT_IDXS)

    # Print rules cleanly
    feat_names = [f"feat_{i}" for i in LIN_FEAT_IDXS]
    for r in regions:
        c = r["coef"]
        b = r["intercept"]
        print(f"\nRegion {r['region']} (n={r['n']}), R²={r['r2_in_region']:.12f}")
        print(f"  target02 = {c[0]:.2f}*{feat_names[0]} + {c[1]:.2f}*{feat_names[1]} + {c[2]:.2f}*{feat_names[2]} + {b:.6g}")

    # 5) Verify on full training data (should be ~perfect if target02 is generated by this rule)
    y_pred = predict_piecewise(X_np, SPLIT_FEAT_IDX, simp_thr, LIN_FEAT_IDXS, regions)
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    rmse = math.sqrt(mean_squared_error(y, y_pred))

    print("\n=== Full-dataset verification ===")
    print(f"R²   = {r2:.12f}")
    print(f"MAE  = {mae:.12e}")
    print(f"RMSE = {rmse:.12e}")
    print(f"Max abs err = {np.max(np.abs(y - y_pred)):.12e}")

    # 6) Generate visualizations
    print("\n=== Generating visualizations ===")
    plot_decision_tree(dt, X.columns, results_dir)
    plot_coefficient_contributions(regions, LIN_FEAT_IDXS, results_dir)
    plot_feature_distribution_and_regions(X_np, SPLIT_FEAT_IDX, simp_thr, results_dir)
    plot_actual_vs_predicted_histogram(y, y_pred, results_dir)

    # 7) Export rules (so it's obvious they came from code)
    out = {
        "id": ID,
        "split_feature_idx": SPLIT_FEAT_IDX,
        "split_points": simp_thr,
        "linear_feature_idxs": LIN_FEAT_IDXS,
        "regions": regions,
    }

    out_path = f"rules_target02_{ID}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()