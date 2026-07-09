"""
Weight of Evidence (WoE) and Information Value (IV) Calculation Functions
For Credit Risk Modeling
"""

import pandas as pd
import numpy as np

def calculate_woe_iv_simple(data, variable, target, n_bins=5):
    """
    Calculate WoE and IV for a variable (simple version)
    
    Parameters:
    - data: DataFrame
    - variable: column name of variable to analyze
    - target: column name of target (0/1)
    - n_bins: number of bins for binning
    
    Returns:
    - woe: series of WoE values per bin
    - iv: total Information Value
    """
    temp_df = data[[variable, target]].dropna().copy()
    
    # Binning
    if temp_df[variable].dtype in ['int64', 'float64']:
        try:
            if len(temp_df[variable].unique()) > 10:
                binned = pd.qcut(temp_df[variable], q=n_bins, duplicates='drop')
            else:
                binned = temp_df[variable]
        except:
            binned = pd.qcut(temp_df[variable], q=n_bins, duplicates='drop')
    else:
        binned = temp_df[variable]
    
    # Contingency table
    contingency = pd.crosstab(binned, temp_df[target])
    
    # Totals
    total_events = temp_df[target].sum()
    total_non_events = len(temp_df) - total_events
    
    # Distributions
    if 1 in contingency.columns:
        events = contingency[1]
        non_events = contingency[0]
    else:
        events = contingency[0]
        non_events = contingency[1] if 1 in contingency.columns else contingency[0]
    
    event_pct = events / total_events
    non_event_pct = non_events / total_non_events
    
    # Handle zeros
    event_pct = event_pct.replace(0, 0.001)
    non_event_pct = non_event_pct.replace(0, 0.001)
    
    # WoE and IV
    woe = np.log(event_pct / non_event_pct)
    iv = ((event_pct - non_event_pct) * woe).sum()
    
    return woe, iv


def calculate_woe_iv_detailed(data, variable, target, n_bins=5):
    """
    Calculate WoE and IV with detailed bins breakdown
    
    Returns:
    - woe_table: DataFrame with bins, counts, WoE, IV
    - iv: total Information Value
    - is_monotonic: boolean indicating if WoE is monotonic
    """
    temp_df = data[[variable, target]].dropna().copy()
    
    # Binning
    if temp_df[variable].dtype in ['int64', 'float64']:
        try:
            if len(temp_df[variable].unique()) > 10:
                binned = pd.qcut(temp_df[variable], q=n_bins, duplicates='drop')
            else:
                binned = temp_df[variable]
        except:
            binned = pd.qcut(temp_df[variable], q=n_bins, duplicates='drop')
    else:
        binned = temp_df[variable]
    
    # Contingency
    contingency = pd.crosstab(binned, temp_df[target])
    
    # Totals
    total_events = temp_df[target].sum()
    total_non_events = len(temp_df) - total_events
    
    # Distributions
    if 1 in contingency.columns:
        events = contingency[1]
        non_events = contingency[0]
    else:
        events = contingency[0]
        non_events = contingency[1] if 1 in contingency.columns else contingency[0]
    
    event_pct = events / total_events
    non_event_pct = non_events / total_non_events
    
    # Handle zeros
    event_pct = event_pct.replace(0, 0.001)
    non_event_pct = non_event_pct.replace(0, 0.001)
    
    # WoE
    woe = np.log(event_pct / non_event_pct)
    iv = ((event_pct - non_event_pct) * woe).sum()
    
    # Build detailed table
    woe_table = pd.DataFrame({
        'Bin': binned.unique(),
        'Count': contingency.sum(axis=1).values,
        'Events': events.values,
        'Non_Events': non_events.values,
        'Event_Rate': (events / contingency.sum(axis=1)).values,
        'WoE': woe.values,
        'IV_Component': ((event_pct - non_event_pct) * woe).values
    }).sort_values('WoE')
    
    # Check monotonicity
    woe_vals = woe_table['WoE'].values
    is_monotonic = (np.all(np.diff(woe_vals) >= 0) or np.all(np.diff(woe_vals) <= 0))
    
    return woe_table, iv, is_monotonic
