"""
Cost-Optimized Fraud Detection System
======================================
Maximizing Business ROI Through Cost-Sensitive ML

This script trains three fraud detection models and identifies the optimal
decision threshold that minimizes total business cost (false positives + false negatives).

Author: Anusha Sadu
Dataset: Kaggle Credit Card Fraud Detection (284,807 transactions)
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Cost Matrix: Business impact of each error type
FP_COST = 25      # False Positive: Cost of blocking a legitimate transaction
FN_COST = 500     # False Negative: Cost of missing a fraudulent transaction

# Train-test split ratio
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Threshold range to test
THRESHOLD_STEP = 0.01

# Output files
OUTPUT_JSON = 'results_summary.json'
OUTPUT_CSV = 'creditcard.csv'

# ============================================================================
# STEP 1: LOAD & EXPLORE DATA
# ============================================================================

print("=" * 70)
print("STEP 1: Loading and exploring dataset...")
print("=" * 70)

try:
    df = pd.read_csv(OUTPUT_CSV)
    print(f"✓ Dataset loaded: {df.shape[0]:,} transactions, {df.shape[1]} features")
    print(f"✓ Fraud rate: {df['Class'].sum() / len(df) * 100:.2f}%")
    print(f"✓ Legitimate: {(df['Class'] == 0).sum():,} | Fraud: {(df['Class'] == 1).sum():,}")
except FileNotFoundError:
    print("✗ Error: creditcard.csv not found. Please ensure it's in the project directory.")
    exit(1)

# ============================================================================
# STEP 2: PREPROCESSING
# ============================================================================

print("\n" + "=" * 70)
print("STEP 2: Preprocessing...")
print("=" * 70)

# Separate features and target
X = df.drop('Class', axis=1)
y = df['Class']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
print(f"✓ Train set: {X_train.shape[0]:,} samples")
print(f"✓ Test set: {X_test.shape[0]:,} samples")

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(f"✓ Features standardized using StandardScaler")

# ============================================================================
# STEP 3: TRAIN THREE MODELS
# ============================================================================

print("\n" + "=" * 70)
print("STEP 3: Training three Logistic Regression models...")
print("=" * 70)

# Model 1: Baseline (threshold 0.5, no class weights)
print("\n[Model 1] Baseline Logistic Regression (threshold 0.5)")
baseline_model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
baseline_model.fit(X_train_scaled, y_train)
y_proba_baseline = baseline_model.predict_proba(X_test_scaled)[:, 1]

# Calculate cost for baseline
y_pred_baseline = (y_proba_baseline >= 0.5).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_baseline).ravel()
baseline_cost = fp * FP_COST + fn * FN_COST
print(f"  - TP: {tp}, FN: {fn}, FP: {fp}, TN: {tn}")
print(f"  - Total Cost: ${baseline_cost:,}")

# Model 2: Balanced (class weights to handle imbalance)
print("\n[Model 2] Balanced Logistic Regression (class_weight='balanced')")
balanced_model = LogisticRegression(
    class_weight='balanced', max_iter=1000, random_state=RANDOM_STATE
)
balanced_model.fit(X_train_scaled, y_train)
y_proba_balanced = balanced_model.predict_proba(X_test_scaled)[:, 1]

# Find best threshold for balanced model
best_threshold_balanced = 0.5
best_cost_balanced = float('inf')
for threshold in np.arange(0.01, 1.0, THRESHOLD_STEP):
    y_pred_t = (y_proba_balanced >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_t).ravel()
    total_cost = fp * FP_COST + fn * FN_COST
    if total_cost < best_cost_balanced:
        best_cost_balanced = total_cost
        best_threshold_balanced = threshold

y_pred_balanced = (y_proba_balanced >= best_threshold_balanced).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_balanced).ravel()
print(f"  - Best Threshold: {best_threshold_balanced:.2f}")
print(f"  - TP: {tp}, FN: {fn}, FP: {fp}, TN: {tn}")
print(f"  - Total Cost: ${best_cost_balanced:,}")

# Model 3: Cost-Optimized (standard LR, but with optimal threshold)
print("\n[Model 3] Cost-Optimized Logistic Regression (optimal threshold)")
cost_optimized_model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
cost_optimized_model.fit(X_train_scaled, y_train)
y_proba_optimized = cost_optimized_model.predict_proba(X_test_scaled)[:, 1]

# Find optimal threshold
best_threshold_optimized = 0.5
best_cost_optimized = float('inf')
for threshold in np.arange(0.01, 1.0, THRESHOLD_STEP):
    y_pred_t = (y_proba_optimized >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_t).ravel()
    total_cost = fp * FP_COST + fn * FN_COST
    if total_cost < best_cost_optimized:
        best_cost_optimized = total_cost
        best_threshold_optimized = threshold

y_pred_optimized = (y_proba_optimized >= best_threshold_optimized).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_optimized).ravel()
print(f"  - Best Threshold: {best_threshold_optimized:.2f}")
print(f"  - TP: {tp}, FN: {fn}, FP: {fp}, TN: {tn}")
print(f"  - Total Cost: ${best_cost_optimized:,}")

# ============================================================================
# STEP 4: COMPARISON & COST SAVINGS
# ============================================================================

print("\n" + "=" * 70)
print("STEP 4: Model Comparison")
print("=" * 70)

cost_reduction = ((baseline_cost - best_cost_optimized) / baseline_cost) * 100
print(f"\n✓ Baseline Cost: ${baseline_cost:,}")
print(f"✓ Optimized Cost: ${best_cost_optimized:,}")
print(f"✓ COST SAVINGS: ${baseline_cost - best_cost_optimized:,} ({cost_reduction:.1f}%)")

# ============================================================================
# STEP 5: GENERATE RESULTS SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("STEP 5: Generating results summary...")
print("=" * 70)

# Extract metrics for all three models
def get_metrics(y_pred, model_name):
    """Extract confusion matrix and metrics"""
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return {
        'accuracy': round(accuracy, 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1_score': round(f1, 4),
        'fraud_caught': int(tp),
        'fraud_missed': int(fn),
        'false_alarms': int(fp),
        'legitimate_blocked': int(fp),
        'legitimate_allowed': int(tn)
    }

results_summary = {
    'baseline': {
        'model_type': 'Logistic Regression (no class weights)',
        'threshold': 0.5,
        'total_cost': int(baseline_cost),
        'cost_breakdown': {'FP_cost': int(fp * FP_COST), 'FN_cost': int(fn * FN_COST)},
        'metrics': get_metrics(y_pred_baseline, 'Baseline')
    },
    'balanced': {
        'model_type': 'Logistic Regression (class_weight=balanced)',
        'threshold': round(best_threshold_balanced, 2),
        'total_cost': int(best_cost_balanced),
        'metrics': get_metrics(y_pred_balanced, 'Balanced')
    },
    'optimized': {
        'model_type': 'Logistic Regression (cost-optimized threshold)',
        'threshold': round(best_threshold_optimized, 2),
        'total_cost': int(best_cost_optimized),
        'cost_breakdown': {'FP_cost': int(fp * FP_COST), 'FN_cost': int(fn * FN_COST)},
        'metrics': get_metrics(y_pred_optimized, 'Optimized')
    },
    'cost_matrix': {
        'false_positive_cost': FP_COST,
        'false_negative_cost': FN_COST
    },
    'savings': {
        'dollar_amount': int(baseline_cost - best_cost_optimized),
        'percentage': round(cost_reduction, 1)
    },
    'dataset': {
        'total_transactions': len(df),
        'fraud_cases': int(df['Class'].sum()),
        'fraud_rate': round(df['Class'].sum() / len(df) * 100, 2)
    }
}

# Save results to JSON
with open(OUTPUT_JSON, 'w') as f:
    json.dump(results_summary, f, indent=2)
print(f"✓ Results saved to {OUTPUT_JSON}")

# ============================================================================
# STEP 6: GENERATE VISUALIZATIONS
# ============================================================================

print("\n" + "=" * 70)
print("STEP 6: Generating visualizations...")
print("=" * 70)

# Visualization 1: Cost vs Threshold
print("\n[Viz 1] Cost vs Threshold curve...")
thresholds = np.arange(0.01, 1.0, THRESHOLD_STEP)
costs = []

for threshold in thresholds:
    y_pred_t = (y_proba_optimized >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_t).ravel()
    total_cost = fp * FP_COST + fn * FN_COST
    costs.append(total_cost)

plt.figure(figsize=(10, 6))
plt.plot(thresholds, costs, linewidth=2.5, color='#2E86AB', label='Total Cost')
plt.scatter([best_threshold_optimized], [best_cost_optimized], 
           color='red', s=200, zorder=5, label=f'Optimal (${best_cost_optimized:,})')
plt.axhline(y=baseline_cost, color='gray', linestyle='--', linewidth=1.5, 
           label=f'Baseline Cost (${baseline_cost:,})')
plt.xlabel('Decision Threshold', fontsize=12, fontweight='bold')
plt.ylabel('Total Business Cost ($)', fontsize=12, fontweight='bold')
plt.title('Cost vs Decision Threshold\nFinding the Optimal Fraud Detection Threshold', 
         fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('cost_vs_threshold.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: cost_vs_threshold.png")
plt.close()

# Visualization 2: ROC Comparison
print("\n[Viz 2] ROC curve comparison...")
fpr_baseline, tpr_baseline, _ = roc_curve(y_test, y_proba_baseline)
fpr_balanced, tpr_balanced, _ = roc_curve(y_test, y_proba_balanced)
fpr_optimized, tpr_optimized, _ = roc_curve(y_test, y_proba_optimized)

roc_auc_baseline = auc(fpr_baseline, tpr_baseline)
roc_auc_balanced = auc(fpr_balanced, tpr_balanced)
roc_auc_optimized = auc(fpr_optimized, tpr_optimized)

plt.figure(figsize=(10, 8))
plt.plot(fpr_baseline, tpr_baseline, label=f'Baseline (AUC = {roc_auc_baseline:.3f})', 
        linewidth=2, color='#FF6B6B')
plt.plot(fpr_balanced, tpr_balanced, label=f'Balanced (AUC = {roc_auc_balanced:.3f})', 
        linewidth=2, color='#4ECDC4')
plt.plot(fpr_optimized, tpr_optimized, label=f'Optimized (AUC = {roc_auc_optimized:.3f})', 
        linewidth=2.5, color='#2E86AB')
plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
plt.title('ROC Curve Comparison\nAll Three Models', fontsize=14, fontweight='bold')
plt.legend(fontsize=11, loc='lower right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('roc_comparison.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: roc_comparison.png")
plt.close()

# Visualization 3: Sensitivity Analysis (Cost matrix variations)
print("\n[Viz 3] Sensitivity analysis heatmap...")
fp_costs = [10, 25, 50, 100]
fn_costs = [250, 500, 1000, 2000]
sensitivity_matrix = np.zeros((len(fn_costs), len(fp_costs)))

for i, fn_c in enumerate(fn_costs):
    for j, fp_c in enumerate(fp_costs):
        best_cost_temp = float('inf')
        for threshold in np.arange(0.01, 1.0, THRESHOLD_STEP):
            y_pred_t = (y_proba_optimized >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred_t).ravel()
            total_cost = fp * fp_c + fn * fn_c
            if total_cost < best_cost_temp:
                best_cost_temp = total_cost
        sensitivity_matrix[i, j] = best_cost_temp

plt.figure(figsize=(10, 7))
sns.heatmap(sensitivity_matrix, annot=True, fmt='.0f', cmap='RdYlGn_r', 
           xticklabels=[f'${c}' for c in fp_costs],
           yticklabels=[f'${c}' for c in fn_costs],
           cbar_kws={'label': 'Minimum Total Cost ($)'})
plt.xlabel('False Positive Cost', fontsize=12, fontweight='bold')
plt.ylabel('False Negative Cost', fontsize=12, fontweight='bold')
plt.title('Sensitivity Analysis: Cost Matrix Variations\nOptimal Cost Under Different Scenarios', 
         fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('sensitivity_heatmap.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: sensitivity_heatmap.png")
plt.close()

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("EXECUTION COMPLETE ✓")
print("=" * 70)
print(f"\n📊 Key Results:")
print(f"   • Baseline Cost: ${baseline_cost:,}")
print(f"   • Optimized Cost: ${best_cost_optimized:,}")
print(f"   • Cost Reduction: {cost_reduction:.1f}% (${baseline_cost - best_cost_optimized:,})")
print(f"   • Optimal Threshold: {best_threshold_optimized:.2f}")
print(f"\n📁 Output Files Generated:")
print(f"   ✓ {OUTPUT_JSON} (results & metrics)")
print(f"   ✓ cost_vs_threshold.png")
print(f"   ✓ roc_comparison.png")
print(f"   ✓ sensitivity_heatmap.png")
print("\n" + "=" * 70)
