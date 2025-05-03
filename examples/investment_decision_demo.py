#!/usr/bin/env python
"""
Luminara CRF Model - Investment Decision Report Demo

This script demonstrates how to use the enhanced evaluation module
to generate comprehensive investment decision reports for AI product investments.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import json
import pickle

# Add the project root directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.evaluation import generate_investment_decision_report
from src.models.crf import InvestmentCRF
from src.data.synthetic import DEFAULT_CARDINALITIES

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_model(model_path):
    """Load a trained CRF model from disk."""
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Successfully loaded model from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None

def print_decision_report(report, case_name=None):
    """Pretty print an investment decision report."""
    if case_name:
        print(f"\n{'='*20} INVESTMENT CASE: {case_name} {'='*20}\n")
    else:
        print(f"\n{'='*60}\n")
    
    # Decision summary
    decision = "INVEST" if report["prediction"] == 1 else "DO NOT INVEST"
    confidence = report["risk_assessment"]["confidence"]
    confidence_text = "HIGH" if confidence > 0.7 else "MEDIUM" if confidence > 0.4 else "LOW"
    
    print(f"DECISION: {decision} (Probability: {report['probability']:.2f}, Confidence: {confidence_text})")
    
    # Key factors
    print("\nKEY FACTORS:")
    bridge_impacts = report["bridge_impacts"]
    for var, impact in sorted(bridge_impacts.items(), key=lambda x: x[1]["impact"], reverse=True):
        var_names = {
            "RP": "Revenue Potential",
            "DC": "Development Cost",
            "R": "Return Rate"
        }
        
        var_values = {
            "RP": ["Low", "Medium", "High"],
            "DC": ["Low", "Medium", "High"],
            "R": ["Low", "Medium", "High"]
        }
        
        var_name = var_names.get(var, var)
        current_value = var_values.get(var, ["0", "1", "2"])[bridge_impacts[var]["current_value"]]
        impact_value = bridge_impacts[var]["impact"]
        
        print(f"  - {var_name}: {current_value} (Impact: {impact_value:.3f})")
    
    # Secondary factors
    print("\nSECONDARY FACTORS:")
    secondary_impacts = report["secondary_impacts"]
    for var, impact in sorted(secondary_impacts.items(), key=lambda x: x[1]["impact"], reverse=True)[:3]:
        var_names = {
            "G": "Growth Rate",
            "M": "Market Size",
            "B": "Bug Density",
            "DT": "Development Time",
            "FC": "Feature Completion",
            "CAC": "Customer Acquisition Cost",
            "ATA": "Average Time to Action",
            "C": "Competitor Penetration"
        }
        
        var_values = {
            "G": ["Slow", "Moderate", "Rapid"],
            "M": ["Small", "Medium", "Large"],
            "B": ["Low", "Medium", "High"],
            "DT": ["Short", "Medium", "Long"],
            "FC": ["Incomplete", "Partial", "Complete"],
            "CAC": ["Low", "Medium", "High"],
            "ATA": ["Quick", "Medium", "Slow"],
            "C": ["Low", "Medium", "High"]
        }
        
        var_name = var_names.get(var, var)
        current_value = var_values.get(var, ["0", "1", "2"])[secondary_impacts[var]["current_value"]]
        impact_value = secondary_impacts[var]["impact"]
        
        print(f"  - {var_name}: {current_value} (Impact: {impact_value:.3f})")
    
    # Recommendations
    print("\nRECOMMENDATIONS:")
    recommendations = report["recommendations"]
    
    if decision == "INVEST":
        print(f"  • {recommendations['decision']} with {recommendations['confidence_level']} confidence")
        
        if recommendations.get("improvement_targets"):
            print("  • To further improve investment returns:")
            for target in recommendations["improvement_targets"]:
                var = target["variable"]
                direction = target["direction"]
                var_name = var_names.get(var, var)
                print(f"    - {direction.capitalize()} {var_name}")
    else:
        print(f"  • {recommendations['decision']} with {recommendations['confidence_level']} confidence")
        
        if recommendations.get("required_changes"):
            print("  • To make this investment worthy, these changes would be needed:")
            for change in recommendations["required_changes"]:
                var = change["variable"]
                var_name = var_names.get(var, var)
                current = var_values.get(var, ["0", "1", "2"])[change["current_value"]]
                target = var_values.get(var, ["0", "1", "2"])[change["target_value"]]
                impact = change["impact"]
                print(f"    - Change {var_name} from {current} to {target} (Impact: +{impact:.3f})")
    
    print(f"\n{'='*60}\n")

def main():
    # Load the most recent model
    model_path = "/Users/lianghaochen/Luminara/results/models/investment_crf_model_20250502_221330.pkl"
    model = load_model(model_path)
    
    if model is None:
        return
    
    # Define test cases
    test_cases = {
        "Promising Startup": {
            "RP": 2,  # High revenue potential
            "DC": 0,  # Low development cost
            "R": 2,   # High return rate
            "G": 2,   # Rapid growth
            "M": 2,   # Large market
            "B": 0,   # Low bug density
            "DT": 0,  # Short development time
            "FC": 2,  # Complete feature set
            "CAC": 0, # Low customer acquisition cost
            "ATA": 0, # Quick time to action
            "C": 0    # Low competition
        },
        "Risky Venture": {
            "RP": 1,  # Medium revenue potential
            "DC": 2,  # High development cost
            "R": 1,   # Medium return rate
            "G": 1,   # Moderate growth
            "M": 1,   # Medium market
            "B": 1,   # Medium bug density
            "DT": 2,  # Long development time
            "FC": 1,  # Partial feature set
            "CAC": 1, # Medium customer acquisition cost
            "ATA": 1, # Medium time to action
            "C": 1    # Medium competition
        },
        "Borderline Case": {
            "RP": 1,  # Medium revenue potential
            "DC": 1,  # Medium development cost
            "R": 1,   # Medium return rate
            "G": 1,   # Moderate growth
            "M": 1,   # Medium market
            "B": 1,   # Medium bug density
            "DT": 1,  # Medium development time
            "FC": 1,  # Partial feature set
            "CAC": 1, # Medium customer acquisition cost
            "ATA": 1, # Medium time to action
            "C": 1    # Medium competition
        },
        "Clear Rejection": {
            "RP": 0,  # Low revenue potential
            "DC": 2,  # High development cost
            "R": 0,   # Low return rate
            "G": 0,   # Slow growth
            "M": 0,   # Small market
            "B": 2,   # High bug density
            "DT": 2,  # Long development time
            "FC": 0,  # Incomplete feature set
            "CAC": 2, # High customer acquisition cost
            "ATA": 2, # Slow time to action
            "C": 2    # High competition
        }
    }
    
    # Generate and print reports for each test case
    for case_name, evidence in test_cases.items():
        report = generate_investment_decision_report(model, evidence)
        print_decision_report(report, case_name)
    
    # Allow user to input custom case
    print("\nWould you like to analyze a custom investment case? (y/n)")
    choice = input().strip().lower()
    
    if choice == 'y':
        custom_case = {}
        print("\nEnter values for each variable (0=Low, 1=Medium, 2=High):")
        
        for var in ["RP", "DC", "R", "G", "M", "B", "DT", "FC", "CAC", "ATA", "C"]:
            var_names = {
                "RP": "Revenue Potential",
                "DC": "Development Cost",
                "R": "Return Rate",
                "G": "Growth Rate",
                "M": "Market Size",
                "B": "Bug Density",
                "DT": "Development Time",
                "FC": "Feature Completion",
                "CAC": "Customer Acquisition Cost",
                "ATA": "Average Time to Action",
                "C": "Competitor Penetration"
            }
            
            while True:
                try:
                    val = int(input(f"{var_names.get(var, var)} (0-2): "))
                    if 0 <= val <= 2:
                        custom_case[var] = val
                        break
                    else:
                        print("Value must be 0, 1, or 2")
                except ValueError:
                    print("Please enter a valid number")
        
        report = generate_investment_decision_report(model, custom_case)
        print_decision_report(report, "Custom Investment Case")

if __name__ == "__main__":
    main()
