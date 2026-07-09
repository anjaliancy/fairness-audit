
CREDIT RISK MODEL - PICKLE FILES SUMMARY
Generated: 2026-04-16 19:44:29

======================================================================
FILES CREATED:
======================================================================

1. credit_risk_model_woe_trained.pkl (1.1 KB)
   - Main production model (LogisticRegression)
   - Trained on 8 WoE-transformed features
   - Test AUC: 0.7511
   - Test Gini: 0.5022
   - Ready to use directly: model = pickle.load(open(...))

2. credit_risk_model_filtered_features.pkl (1.1 KB)
   - Model trained on 7 filtered original features
   - Used for interpretability analysis
   - Same performance as model_old
   - Alternative if you need feature-level explainability

3. model_feature_metadata.pkl
   - WoE feature names (used in production model)
   - Filtered feature names (7 key variables)
   - Original feature names (all 28 available features)

4. model_performance_metrics.pkl
   - AUC, Gini, coefficients, intercept
   - Training and test metrics
   - Model creation timestamp

5. model_preprocessing_info.pkl
   - Data transformation details
   - Train/test split information
   - Default rates
   - Feature counts

======================================================================
HOW TO USE THESE FILES:
======================================================================

LOAD THE MAIN MODEL:
    import pickle
    model = pickle.load(open('credit_risk_model_woe_trained.pkl', 'rb'))

MAKE PREDICTIONS:
    predictions = model.predict(X_test_woe)
    probabilities = model.predict_proba(X_test_woe)

LOAD METADATA:
    metadata = pickle.load(open('model_feature_metadata.pkl', 'rb'))
    woe_features = metadata['woe_features']

======================================================================
MODEL DETAILS:
======================================================================

Production Model: credit_risk_model_woe_trained.pkl
Features: Credit Score, First Time Homebuyer Flag, Loan Purpose, Mortgage Insurance Percentage (MI %), Original Combined Loan-to-Value (CLTV), Original Debt-to-Income (DTI) Ratio, Original Interest Rate, Original Loan Term, Program Indicator
Test AUC: 0.7511
Test Gini: 0.5022
Training Set Size: 70,000
Test Set Size: 30,000
Default Rate (Train): 14.93%
Default Rate (Test): 10.21%
