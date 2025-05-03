"""
Inference module for the Luminara CRF model.

This module implements inference algorithms for the CRF model,
including belief propagation for probabilistic inference and
Maximum a Posteriori (MAP) for final prediction.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Set, Union
from pgmpy.models import FactorGraph
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference import BeliefPropagation, VariableElimination
import logging
import time

logger = logging.getLogger(__name__)


def belief_propagation(
    model: FactorGraph,
    variables: List[str],
    evidence: Dict[str, int],
    max_iterations: int = 100,
    tolerance: float = 1e-6
) -> Dict[str, Dict[int, float]]:
    """
    Perform belief propagation inference on the factor graph model.
    
    Args:
        model: Factor graph model
        variables: List of variables to query
        evidence: Dictionary mapping observed variable names to their values
        max_iterations: Maximum number of belief propagation iterations
        tolerance: Convergence tolerance for belief propagation
        
    Returns:
        Dictionary mapping each queried variable to its posterior distribution
    """
    start_time = time.time()
    
    # Initialize the belief propagation inference engine
    bp = BeliefPropagation(model)
    
    # Set algorithm parameters
    bp.max_iterations = max_iterations
    bp.tolerance = tolerance
    
    logger.info(f"Starting belief propagation inference for variables {variables}")
    
    try:
        # Run calibration (message passing)
        bp.calibrate()
        
        # Query each variable
        results = {}
        for var in variables:
            # Compute marginal distribution for this variable
            marginal = bp.query(variables=[var], evidence=evidence)
            
            # Convert to dictionary for easier access
            var_dist = {i: marginal.values[i] for i in range(len(marginal.values))}
            results[var] = var_dist
        
        elapsed_time = time.time() - start_time
        logger.info(f"Belief propagation completed in {elapsed_time:.4f} seconds")
        
        return results
    
    except Exception as e:
        logger.error(f"Belief propagation failed: {str(e)}")
        raise
    

def map_inference(
    model: FactorGraph, 
    evidence: Dict[str, int] = None,
    target_variables: List[str] = None
) -> Dict[str, int]:
    """
    Perform Maximum a Posteriori (MAP) inference to find the most
    likely configuration of variables.
    
    Args:
        model: Factor graph model
        evidence: Dictionary mapping observed variable names to their values
        target_variables: List of variables to find MAP assignment for.
                        If None, all unobserved variables are included.
                        
    Returns:
        Dictionary mapping each variable to its MAP assignment
    """
    start_time = time.time()
    
    if evidence is None:
        evidence = {}
    
    # If no target variables specified, use all unobserved variables
    if target_variables is None:
        all_variables = set()
        for factor in model.get_factors():
            all_variables.update(factor.variables)
        target_variables = list(all_variables - set(evidence.keys()))
    
    logger.info(f"Starting MAP inference for variables {target_variables}")
    
    try:
        # Use Variable Elimination algorithm for MAP inference
        ve = VariableElimination(model)
        
        # Perform MAP inference
        map_result = ve.map_query(variables=target_variables, evidence=evidence)
        
        elapsed_time = time.time() - start_time
        logger.info(f"MAP inference completed in {elapsed_time:.4f} seconds")
        
        return map_result
    
    except KeyError as ke:
        logger.warning(f"MAP inference KeyError: {str(ke)}. Falling back to manual inference.")
        # Fall back to a simpler approach - find the most likely value for each variable
        result = {}
        for var in target_variables:
            # Find all factors containing this variable
            relevant_factors = [f for f in model.get_factors() if var in f.variables]
            if not relevant_factors:
                logger.warning(f"No factors found for variable {var}")
                # Use a default value (0) if no information available
                result[var] = 0
                continue
                
            # Find the most likely value by marginalizing over other variables
            # This is a simplification that might not give exact MAP results
            # but should work reasonably well for our investment model
            best_value = 0
            best_prob = -float('inf')
            
            # Get the cardinality of this variable
            var_card = 0
            for f in relevant_factors:
                var_idx = f.variables.index(var)
                var_card = f.cardinality[var_idx]
                break
                
            # Try different values and find the one with highest probability
            for val in range(var_card):
                temp_evidence = evidence.copy()
                temp_evidence[var] = val
                
                # Calculate probability for this assignment
                log_prob = 0
                for factor in relevant_factors:
                    # Create an assignment that matches this factor's variables
                    factor_assignment = {}
                    for v in factor.variables:
                        if v in temp_evidence:
                            factor_assignment[v] = temp_evidence[v]
                            
                    # Skip if we don't have complete evidence for this factor
                    if len(factor_assignment) != len(factor.variables):
                        continue
                        
                    # Find index in the factor's values array
                    idx = 0
                    for i, v in enumerate(factor.variables):
                        idx = idx * factor.cardinality[i] + factor_assignment[v]
                        
                    # Add log probability
                    try:
                        factor_val = factor.values.flatten()[idx]
                        if factor_val > 0:
                            log_prob += np.log(factor_val)
                    except IndexError:
                        # Skip factors with indexing issues
                        continue
                
                # Update best value if this has higher probability
                if log_prob > best_prob:
                    best_prob = log_prob
                    best_value = val
                    
            result[var] = best_value
            
        elapsed_time = time.time() - start_time
        logger.info(f"Manual MAP inference completed in {elapsed_time:.4f} seconds")
        
        return result
    
    except Exception as e:
        logger.error(f"MAP inference failed: {str(e)}")
        raise


def predict_investment_worthiness(
    model: FactorGraph,
    evidence: Dict[str, int],
    method: str = "map"
) -> Union[int, Dict[int, float]]:
    """
    Predict the investment worthiness (Y) using the given evidence.
    
    Args:
        model: Factor graph model
        evidence: Dictionary of observed variables and their values
        method: Inference method, either "map" for MAP inference or "bp" for belief propagation
        
    Returns:
        If method is "map": The most likely value for Y
        If method is "bp": Distribution over possible Y values
    """
    if method.lower() == "map":
        # Use MAP to get the most likely assignment
        result = map_inference(model, evidence, ["Y"])
        return result["Y"]
    
    elif method.lower() == "bp":
        # Use belief propagation to get posterior distribution
        result = belief_propagation(model, ["Y"], evidence)
        return result["Y"]
    
    else:
        raise ValueError(f"Unknown inference method: {method}. Use 'map' or 'bp'.")


def batch_inference(
    model: FactorGraph,
    data: pd.DataFrame,
    target_variable: str = "Y",
    method: str = "map",
    evidence_columns: List[str] = None
) -> pd.DataFrame:
    """
    Perform batch inference on a dataset.
    
    Args:
        model: Factor graph model
        data: DataFrame containing evidence variables
        target_variable: Target variable to predict
        method: Inference method, either "map" or "bp"
        evidence_columns: List of columns to use as evidence.
                         If None, all columns except target_variable are used.
                         
    Returns:
        DataFrame with original data and prediction results
    """
    # Make a copy to avoid modifying the original
    result_df = data.copy()
    
    # Determine evidence columns
    if evidence_columns is None:
        evidence_columns = [col for col in data.columns if col != target_variable]
    
    # Add prediction column
    result_df["predicted_" + target_variable] = None
    
    # If using BP, add probability columns
    if method.lower() == "bp":
        # Get target variable cardinality
        target_card = 0
        for factor in model.get_factors():
            if target_variable in factor.variables:
                idx = factor.variables.index(target_variable)
                target_card = factor.cardinality[idx]
                break
        
        # Add probability columns
        for i in range(target_card):
            result_df[f"prob_{target_variable}_{i}"] = None
    
    # Process each row
    total_rows = len(result_df)
    logger.info(f"Starting batch inference on {total_rows} samples using {method} method")
    
    for idx, row in result_df.iterrows():
        # Prepare evidence
        evidence = {col: int(row[col]) for col in evidence_columns if not pd.isna(row[col])}
        
        # Perform inference
        if method.lower() == "map":
            pred = predict_investment_worthiness(model, evidence, "map")
            result_df.at[idx, "predicted_" + target_variable] = pred
        
        elif method.lower() == "bp":
            dist = predict_investment_worthiness(model, evidence, "bp")
            
            # Store most likely value
            pred = max(dist.items(), key=lambda x: x[1])[0]
            result_df.at[idx, "predicted_" + target_variable] = pred
            
            # Store probabilities
            for val, prob in dist.items():
                result_df.at[idx, f"prob_{target_variable}_{val}"] = prob
    
    logger.info(f"Batch inference completed")
    
    return result_df