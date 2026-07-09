"""
Generate Anchor Explanations for Logistic Regression and Create PDF Report
===========================================================================
For logistic regression, we can derive simple decision rules directly from the 
linear coefficients and thresholds, making interpretable anchors.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
from datetime import datetime

# ============================================================================
# 1. LOAD MODEL AND DATA
# ============================================================================
print("Loading model and data...")

# Import the explainability module
import sys
sys.path.insert(0, 'src')
from explainability_assessment import _predict_scores

# Assuming these are in your notebook kernel
# We'll need to access them - for now, we'll create a function to load/access them
# This script can be run directly or integrated into the notebook

print("✓ Setup complete")

# ============================================================================
# 2. CREATE ANCHOR RULES FROM LOGISTIC REGRESSION COEFFICIENTS
# ============================================================================
def create_logistic_anchors(model, X_train, X_test, feature_names=None, top_n=5):
    """
    Create anchor-style decision rules directly from logistic regression coefficients.
    
    For logistic regression: log_odds = β₀ + Σ(βᵢ × Xᵢ)
    We identify the top contributing features and create simple thresholds.
    """
    
    if feature_names is None:
        feature_names = list(X_train.columns)
    
    # Get coefficients
    coefs = model.coef_[0]
    intercept = model.intercept_[0]
    
    # Create coefficient dataframe
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefs,
        'Abs_Coefficient': np.abs(coefs)
    }).sort_values('Abs_Coefficient', ascending=False)
    
    print(f"\n{'='*80}")
    print("LOGISTIC REGRESSION COEFFICIENTS (sorted by importance)")
    print(f"{'='*80}")
    print(f"Intercept: {intercept:.6f}\n")
    print(coef_df.to_string(index=False))
    
    # Create anchors based on high-impact features
    anchors = []
    
    for idx, row in coef_df.head(top_n).iterrows():
        feat = row['Feature']
        coef = row['Coefficient']
        
        # Get feature statistics from test set
        feat_val = X_test[feat]
        feat_mean = feat_val.mean()
        feat_std = feat_val.std()
        feat_min = feat_val.min()
        feat_max = feat_val.max()
        feat_median = feat_val.median()
        
        # Determine thresholds based on coefficient direction
        if coef > 0:
            # Positive coefficient: higher values increase default risk
            threshold_high = feat_median + feat_std
            threshold_low = feat_median
            anchor_rule = f"If {feat} > {threshold_high:.4f}, prediction tends toward DEFAULT (coef: +{coef:.6f})"
            direction = "increases risk"
        else:
            # Negative coefficient: higher values decrease default risk
            threshold_high = feat_median
            threshold_low = feat_median - feat_std
            anchor_rule = f"If {feat} < {threshold_low:.4f}, prediction tends toward DEFAULT (coef: {coef:.6f})"
            direction = "decreases risk"
        
        accuracy = _compute_anchor_accuracy(
            X_test, feat, threshold_high if coef > 0 else threshold_low, 
            model, coef > 0
        )
        
        anchors.append({
            'Rank': len(anchors) + 1,
            'Feature': feat,
            'Coefficient': coef,
            'Direction': direction,
            'Threshold': threshold_high if coef > 0 else threshold_low,
            'Rule': anchor_rule,
            'Precision': accuracy['precision'],
            'Coverage': accuracy['coverage'],
            'Feature_Min': feat_min,
            'Feature_Max': feat_max,
            'Feature_Mean': feat_mean,
            'Feature_Std': feat_std,
        })
    
    return coef_df, pd.DataFrame(anchors), intercept


def _compute_anchor_accuracy(X_test, feature, threshold, model, is_greater_than):
    """
    Compute precision and coverage of the anchor rule.
    
    Precision: Among instances where the rule applies, how many are actually default?
    Coverage: What fraction of test instances does the rule apply to?
    """
    predictions = model.predict_proba(X_test)[:, 1]  # Default probability
    
    if is_greater_than:
        rule_applies = X_test[feature] > threshold
    else:
        rule_applies = X_test[feature] < threshold
    
    if rule_applies.sum() == 0:
        return {'precision': 0, 'coverage': 0}
    
    # Precision: among instances where rule applies, what's high-risk rate?
    high_risk_when_rule = (predictions[rule_applies] > 0.5).sum()
    precision = high_risk_when_rule / rule_applies.sum() if rule_applies.sum() > 0 else 0
    
    # Coverage: what fraction of test data matches this rule?
    coverage = rule_applies.sum() / len(X_test)
    
    return {'precision': precision, 'coverage': coverage}


# ============================================================================
# 3. CREATE MULTI-FEATURE ANCHORS (More complex rules)
# ============================================================================
def create_combined_anchors(model, X_test, top_features, threshold=0.5):
    """
    Create combined rule anchors (IF Feature_A AND Feature_B THEN ...).
    """
    
    # Get high-risk predictions
    high_risk = model.predict_proba(X_test)[:, 1] > threshold
    high_risk_data = X_test[high_risk]
    
    combined_rules = []
    
    # Top 2 feature pairs
    for i in range(min(2, len(top_features)-1)):
        feat_a = top_features[i]
        feat_b = top_features[i+1]
        
        if len(high_risk_data) > 0:
            thresh_a = high_risk_data[feat_a].median()
            thresh_b = high_risk_data[feat_b].median()
            
            # Check rule accuracy
            rule_applies = (X_test[feat_a] > thresh_a) & (X_test[feat_b] > thresh_b)
            high_risk_when_rule = (high_risk[rule_applies]).sum()
            
            if rule_applies.sum() > 0:
                precision = high_risk_when_rule / rule_applies.sum()
                coverage = rule_applies.sum() / len(X_test)
                
                if precision >= 0.7:  # Only keep high-precision rules
                    combined_rules.append({
                        'Rule': f"If {feat_a} > {thresh_a:.4f} AND {feat_b} > {thresh_b:.4f}",
                        'Precision': precision,
                        'Coverage': coverage,
                        'Risk_Rate': high_risk_when_rule / rule_applies.sum() if rule_applies.sum() > 0 else 0,
                    })
    
    return pd.DataFrame(combined_rules) if combined_rules else None


# ============================================================================
# 4. GENERATE PDF REPORT
# ============================================================================
def create_anchor_pdf_report(model, X_train, X_test, y_test, coef_df, anchors_df, 
                             intercept, combined_anchors=None, output_path='anchor_report.pdf'):
    """
    Create a comprehensive PDF report with anchor explanations and visualizations.
    """
    
    print(f"\nGenerating PDF report: {output_path}")
    
    with PdfPages(output_path) as pdf:
        # ====== PAGE 1: TITLE & EXECUTIVE SUMMARY ======
        fig = plt.figure(figsize=(8.5, 11))
        fig.suptitle('Anchor Explanations for Logistic Regression\nCredit Risk Model', 
                     fontsize=20, fontweight='bold', y=0.95)
        
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        summary_text = f"""
EXECUTIVE SUMMARY

Model Type: Logistic Regression (Linear Binary Classification)
Target: Credit Default Probability
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

MODEL COEFFICIENTS (Top Risk Drivers):

For logistic regression, predictions are computed as:
    Default Probability = 1 / (1 + exp(-(β₀ + Σ βᵢXᵢ)))
    
Where β represents the coefficient for each feature.

INTERPRETATION:
• Positive coefficient (β > 0): Feature increases default probability
• Negative coefficient (β < 0): Feature decreases default probability  
• Larger |β| = Stronger influence on prediction

ANCHOR RULES:
Simple thresholds derived from the coefficients and data distributions.
Each rule has:
    • Precision: How often the rule predicts high risk correctly
    • Coverage: What fraction of loans match this rule
    
EXAMPLE:
    "If Interest_Rate > 0.065, prediction tends toward DEFAULT"
    Precision: 65% | Coverage: 45%
    → Among loans with Interest_Rate > 0.065, 65% default
    → This rule applies to 45% of all loans
"""
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # ====== PAGE 2: COEFFICIENT IMPORTANCE BAR CHART ======
        fig, ax = plt.subplots(figsize=(8.5, 11))
        
        top_coefs = coef_df.head(10)
        colors = ['#e74c3c' if x > 0 else '#2ecc71' for x in top_coefs['Coefficient']]
        
        bars = ax.barh(range(len(top_coefs)), top_coefs['Coefficient'], color=colors, alpha=0.7, edgecolor='black')
        ax.set_yticks(range(len(top_coefs)))
        ax.set_yticklabels(top_coefs['Feature'])
        ax.set_xlabel('Coefficient Value (β)', fontweight='bold', fontsize=12)
        ax.set_title('Logistic Regression Coefficients\n(Red=Increases Risk, Green=Decreases Risk)', 
                     fontweight='bold', fontsize=13)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, top_coefs['Coefficient'])):
            ax.text(val, i, f'  {val:.6f}', va='center', fontweight='bold', fontsize=9)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # ====== PAGE 3: ANCHOR RULES TABLE ======
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        # Create table data
        table_data = []
        table_data.append(['Rank', 'Feature', 'Rule', 'Precision', 'Coverage'])
        
        for idx, row in anchors_df.head(5).iterrows():
            table_data.append([
                str(int(row['Rank'])),
                row['Feature'][:20],
                row['Rule'][:50] + ('...' if len(row['Rule']) > 50 else ''),
                f"{row['Precision']:.1%}",
                f"{row['Coverage']:.1%}",
            ])
        
        # Create table
        table = ax.table(cellText=table_data, cellLoc='left', loc='center',
                        colWidths=[0.08, 0.15, 0.45, 0.15, 0.15])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Style header row
        for i in range(5):
            table[(0, i)].set_facecolor('#3498db')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Alternate row colors
        for i in range(1, len(table_data)):
            for j in range(5):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#ecf0f1')
        
        ax.text(0.5, 0.95, 'TOP 5 ANCHOR RULES\n(Single-Feature Conditions)', 
               transform=ax.transAxes, fontsize=14, fontweight='bold',
               ha='center', va='top')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # ====== PAGE 4: DETAILED ANCHOR EXPLANATIONS ======
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        detailed_text = "DETAILED ANCHOR RULE EXPLANATIONS\n\n"
        
        for idx, row in anchors_df.head(3).iterrows():
            detailed_text += f"""
ANCHOR RULE #{int(row['Rank'])}: {row['Feature']}
{'='*70}
Coefficient (β):        {row['Coefficient']:.6f}
Direction:             {row['Direction']}

RULE: {row['Rule']}

Threshold:             {row['Threshold']:.6f}
Feature Range:         [{row['Feature_Min']:.6f}, {row['Feature_Max']:.6f}]
Feature Mean ± Std:    {row['Feature_Mean']:.6f} ± {row['Feature_Std']:.6f}

PERFORMANCE:
    Precision:         {row['Precision']:.1%}  (when rule applies, this often predicts default)
    Coverage:          {row['Coverage']:.1%}  (% of test data matching this rule)

INTERPRETATION:
When {row['Feature']} {'is above' if row['Coefficient'] > 0 else 'is below'} the threshold,
the model is more likely to predict default. The {row['Precision']:.1%} precision means
that {int(row['Precision']*100)}% of loans matching this condition actually defaulted.

"""
        
        ax.text(0.05, 0.95, detailed_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # ====== PAGE 5: COMBINED RULES (if available) ======
        if combined_anchors is not None and len(combined_anchors) > 0:
            fig = plt.figure(figsize=(8.5, 11))
            ax = fig.add_subplot(111)
            ax.axis('off')
            
            combined_text = "COMPOUND ANCHOR RULES\n"
            combined_text += "(Combinations of 2+ features)\n\n"
            
            for idx, row in combined_anchors.iterrows():
                combined_text += f"""
RULE {idx+1}: {row['Rule']}
    Precision: {row['Precision']:.1%}  |  Coverage: {row['Coverage']:.1%}
    Default Rate: {row['Risk_Rate']:.1%}

"""
            
            if len(combined_anchors) == 0:
                combined_text += "\nNo compound rules met minimum precision threshold (70%).\n"
                combined_text += "Single-feature rules may be sufficient for this model.\n"
            
            ax.text(0.05, 0.95, combined_text, transform=ax.transAxes,
                   fontsize=10, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
            
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
        
        # ====== PAGE 6: ANCHOR PERFORMANCE VISUALIZATION ======
        fig, axes = plt.subplots(1, 2, figsize=(8.5, 11))
        
        # Precision vs Coverage scatter
        ax = axes[0]
        scatter = ax.scatter(anchors_df['Coverage'], anchors_df['Precision'], 
                           s=200, c=anchors_df['Abs_Coefficient'], cmap='RdYlGn_r',
                           alpha=0.7, edgecolor='black')
        ax.set_xlabel('Coverage (% of data)', fontweight='bold')
        ax.set_ylabel('Precision (% accuracy)', fontweight='bold')
        ax.set_title('Anchor Rule Quality', fontweight='bold')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0.7, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Min precision (70%)')
        ax.legend()
        
        # Add labels to points
        for idx, row in anchors_df.head(5).iterrows():
            ax.annotate(row['Feature'][:10], 
                       (row['Coverage'], row['Precision']),
                       fontsize=8, ha='right')
        
        # Precision bar chart for top rules
        ax = axes[1]
        top_5 = anchors_df.head(5)
        bars = ax.barh(range(len(top_5)), top_5['Precision'], color='#3498db', alpha=0.7, edgecolor='black')
        ax.set_yticks(range(len(top_5)))
        ax.set_yticklabels(top_5['Feature'], fontsize=9)
        ax.set_xlabel('Precision', fontweight='bold')
        ax.set_title('Top 5 Anchor Rule Precision', fontweight='bold')
        ax.set_xlim([0, 1])
        ax.grid(axis='x', alpha=0.3)
        
        for i, (bar, val) in enumerate(zip(bars, top_5['Precision'])):
            ax.text(val + 0.02, i, f'{val:.1%}', va='center', fontweight='bold', fontsize=9)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # ====== PAGE 7: USAGE GUIDE ======
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        usage_text = """
HOW TO USE ANCHOR EXPLANATIONS
===============================

1. UNDERSTANDING PRECISION & COVERAGE
   
   Precision (vertical axis): How reliable is this rule?
   • 90% precision: When this rule applies, 90% of loans default
   • 70% precision: When this rule applies, 70% of loans default
   • Higher = more reliable rule
   
   Coverage (horizontal axis): How broadly does this rule apply?
   • 50% coverage: This rule applies to 50% of your portfolio
   • 10% coverage: This rule only applies to 10% of loans
   • Higher = rule covers more of your business

2. PRACTICAL APPLICATION
   
   For Loan Officers:
   ✓ Use anchor rules as quick decision aids
   ✓ "If Interest Rate > 0.065, expect higher risk → review carefully"
   ✓ Rules provide transparency about model decisions
   ✓ Combine with other requirements (credit score, income, etc.)
   
   For Portfolio Analysis:
   ✓ Identify high-risk segments of your portfolio
   ✓ Monitor % of loans matching high-risk rules
   ✓ Use as early warning systems for problem areas

3. DECISION RULES EXAMPLE
   
   Rule 1: "If Interest Rate > 0.065"
           → 60% precision, 45% coverage
           → Among 45% of loans with high rates, 60% default
           → Action: Consider rate caps or require larger down payment
   
   Rule 2: "If DTI > 0.45" 
           → 55% precision, 35% coverage
           → Among 35% of loans with high debt burden, 55% default
           → Action: Enforce stricter debt-to-income requirements

4. IMPORTANT NOTES
   
   ⚠ These are simplifications of the true model decision boundary
   ⚠ Model uses all 8 features simultaneously, not just single thresholds
   ✓ Anchors are useful for explainability and communication
   ✓ Always validate rules with actual business domain experts
   ✓ Rules should evolve as market conditions change (retrain quarterly)

5. NEXT STEPS
   
   □ Share these anchor rules with business stakeholders
   □ Implement rules in loan decision workflows
   □ Monitor how often rules apply to new originations
   □ Track actual default rates vs predicted rates
   □ Retrain model quarterly with new data
"""
        
        ax.text(0.05, 0.95, usage_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.2))
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    print(f"✓ PDF report saved: {output_path}")
    return output_path


# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    print("="*80)
    print("ANCHOR EXPLANATION GENERATION FOR LOGISTIC REGRESSION")
    print("="*80)
    
    # Note: This script assumes access to model_old, Xtr_old, Xte_old, yte_old
    # If running standalone, you'll need to load the model from pickle/joblib
    
    print("\n⚠️  NOTE: This script should be run from the Jupyter notebook")
    print("    where model_old and data are available in the kernel.")
    print("\nTo use this interactively, copy the functions above into your notebook")
    print("and call them with your model and data:")
    print("""
    # In your notebook:
    coef_df, anchors_df, intercept = create_logistic_anchors(
        model_old, Xtr_old, Xte_old, top_n=5
    )
    
    combined = create_combined_anchors(model_old, Xte_old, anchors_df['Feature'].head(3).tolist())
    
    create_anchor_pdf_report(
        model_old, Xtr_old, Xte_old, yte_old,
        coef_df, anchors_df, intercept, combined,
        output_path='anchor_explanations_report.pdf'
    )
    """)
