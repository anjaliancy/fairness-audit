"""
Scorecard Scaling and Scoring Functions
For Credit Risk Probability of Default Models
"""

import pandas as pd
import numpy as np

def calculate_scorecard_coefficients(model_coefficients, base_score=600, pdo=50, target_odds=20):
    """
    Convert logistic regression coefficients to scorecard points
    
    Parameters:
    - model_coefficients: dict or Series with variable names and coefficients
    - base_score: base score (default: 600)
    - pdo: Points to Double Odds (default: 50)
    - target_odds: target odds ratio (default: 20, meaning 1:20)
    
    Returns:
    - scorecard: DataFrame with variables and their point values
    - scaling_factor: factor for scaling
    - intercept: scaled intercept value
    """
    
    scaling_factor = pdo / np.log(2)
    intercept = base_score + scaling_factor * np.log(target_odds)
    
    scorecard_data = []
    for var, coef in model_coefficients.items():
        points = scaling_factor * coef
        scorecard_data.append({
            'Variable': var,
            'Coefficient': coef,
            'Points_Per_Unit': points,
            'Odds_Ratio': np.exp(coef)
        })
    
    scorecard = pd.DataFrame(scorecard_data)
    
    return scorecard, scaling_factor, intercept


def calculate_application_score(data, coefficients, intercept):
    """
    Calculate application score for new data
    
    Parameters:
    - data: DataFrame with features (must match coefficient variable names)
    - coefficients: Series with variable coefficients
    - intercept: intercept value from scorecard
    
    Returns:
    - scores: array of scores
    """
    scores = intercept + data.dot(coefficients)
    return scores


def classify_score_band(score, low_cutoff=550, med_cutoff=650):
    """
    Classify score into risk bands
    
    Parameters:
    - score: application score
    - low_cutoff: threshold for low risk (default: 550)
    - med_cutoff: threshold for medium risk (default: 650)
    
    Returns:
    - risk_band: 'Low', 'Medium', or 'High'
    """
    if score >= med_cutoff:
        return 'Low'
    elif score >= low_cutoff:
        return 'Medium'
    else:
        return 'High'
