"""
Training module for the Luminara CRF model.

This module handles the training of Conditional Random Field models for 
AI investment worthiness prediction, using L-BFGS as the parameter 
estimation method.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from pgmpy.models import FactorGraph, DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.factors.discrete import DiscreteFactor
from sklearn.model_selection import train_test_split
import logging
from tqdm import tqdm

from .cliques import create_all_cliques, BaseClique
from .crf import InvestmentCRF

logger = logging.getLogger(__name__)


def prepare_training_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the dataset for training the CRF model.
    
    Args:
        data: Raw input data containing all variables
        
    Returns:
        Processed DataFrame ready for model training
    """
    # Make a copy to avoid modifying the original
    processed_data = data.copy()
    
    # Verify that all required variables are present
    required_vars = [
        "M", "C", "G", "RP",  # Market clique
        "B", "DC", "DT", "FC",  # Development clique
        "CAC", "R", "ATA",  # Customer clique
        "Y"  # Target variable (Investment worthiness)
    ]
    
    missing_vars = [var for var in required_vars if var not in processed_data.columns]
    if missing_vars:
        raise ValueError(f"Missing required variables in training data: {missing_vars}")
    
    # Ensure all variables are discrete (convert if needed)
    # For simplicity, we're not implementing discretization here,
    # but in a real implementation, you would add binning logic as needed
    
    logger.info(f"Training data prepared with {len(processed_data)} samples")
    
    return processed_data


def compute_clique_potentials(
    cliques: Dict[str, BaseClique],
    data: pd.DataFrame
) -> Dict[str, DiscreteFactor]:
    """
    Compute potential functions for each clique using the training data.
    
    Args:
        cliques: Dictionary of clique objects
        data: Training data
        
    Returns:
        Dictionary mapping clique names to their potential functions
    """
    potentials = {}
    
    for name, clique in cliques.items():
        # Get variables in the clique
        variables = clique.variables
        
        # For single-variable cliques
        if len(variables) == 1:
            var = variables[0]
            # Compute distribution directly
            values = data[var].value_counts(normalize=True).sort_index().values
            factor = DiscreteFactor(
                variables=[var],
                cardinality=[clique.cardinalities[var]],
                values=values
            )
            potentials[name] = factor
            logger.debug(f"Computed potential for single-variable clique {name} with variable {var}")
            continue
        
        try:
            # For multi-variable cliques, use a direct approach with joint counts
            # Create contingency table with all combinations
            contingency = pd.crosstab(
                index=[data[var] for var in variables[:-1]],
                columns=data[variables[-1]],
                normalize=True
            ).fillna(0.001)  # Small value for smoothing
            
            # Convert to the format needed for DiscreteFactor
            shape = [clique.cardinalities[var] for var in variables]
            values = np.zeros(np.prod(shape))
            
            # Flatten and populate the values array
            indices = np.array(list(np.ndindex(tuple(shape))))
            for idx, combo in enumerate(indices):
                # Convert index to tuple of variable values
                var_values = tuple(combo)
                
                try:
                    # Try to get the probability from the contingency table
                    if len(variables) == 2:
                        # Simple case with two variables
                        prob = contingency.iloc[var_values[0], var_values[1]]
                    else:
                        # More complex case, need to index properly
                        index_key = tuple(var_values[:-1])
                        col_key = var_values[-1]
                        prob = contingency.loc[index_key, col_key]
                    
                    values[idx] = max(prob, 0.001)  # Ensure no zeros
                except (KeyError, IndexError):
                    # Use a small default probability if combination not in data
                    values[idx] = 0.001
            
            # Create the factor
            factor = DiscreteFactor(
                variables=variables,
                cardinality=[clique.cardinalities[var] for var in variables],
                values=values
            )
            factor.normalize()
            
        except Exception as e:
            logger.warning(f"Error estimating joint distribution for clique {name}: {str(e)}")
            # Fallback to uniform distribution
            card_list = clique.get_cardinality_list()
            values = np.ones(np.prod(card_list))
            factor = DiscreteFactor(
                variables=variables,
                cardinality=card_list,
                values=values
            )
            factor.normalize()
        
        # Store the potential function
        potentials[name] = factor
        
        logger.debug(f"Computed potential for clique {name} with variables {variables}")
    
    return potentials


def train_model(
    data: pd.DataFrame,
    cardinalities: Optional[Dict[str, int]] = None,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[InvestmentCRF, Dict[str, Any]]:
    """
    Train the Conditional Random Field model for investment worthiness prediction.
    
    Args:
        data: Training data containing all variables
        cardinalities: Optional dictionary mapping variable names to their cardinalities
                      If None, will be inferred from the data
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (trained CRF model, training statistics)
    """
    # Prepare the data
    processed_data = prepare_training_data(data)
    
    # Split into training and validation sets
    train_data, val_data = train_test_split(
        processed_data, test_size=test_size, random_state=random_state
    )
    
    logger.info(f"Split data into {len(train_data)} training and {len(val_data)} validation samples")
    
    # Infer cardinalities if not provided
    if cardinalities is None:
        cardinalities = {}
        for col in processed_data.columns:
            cardinalities[col] = processed_data[col].nunique()
        
        logger.info(f"Inferred cardinalities: {cardinalities}")
    
    # Create cliques based on the Luminara CRF structure
    cliques = create_all_cliques(cardinalities)
    
    # Compute potential functions for each clique
    potentials = compute_clique_potentials(cliques, train_data)
    
    # Create and initialize the CRF model
    crf_model = InvestmentCRF(cliques=cliques, cardinalities=cardinalities)
    
    # Add all factors to the model
    for name, potential in potentials.items():
        crf_model.add_factor(potential)
    
    # Train the model using L-BFGS
    logger.info("Starting parameter estimation using L-BFGS...")
    
    # Perform L-BFGS parameter optimization
    # Since pgmpy doesn't directly expose L-BFGS for factor graphs,
    # we're using a workaround here to optimize the parameters
    # In a real implementation, this would use the proper optimizer
    crf_model.optimize_parameters(train_data, method='L-BFGS-B', max_iter=100)
    
    # Compute training statistics
    train_accuracy = crf_model.evaluate(train_data)
    val_accuracy = crf_model.evaluate(val_data)
    
    training_stats = {
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "train_accuracy": train_accuracy,
        "val_accuracy": val_accuracy,
        "cardinalities": cardinalities
    }
    
    logger.info(f"Model training completed. Train accuracy: {train_accuracy:.4f}, "
               f"Validation accuracy: {val_accuracy:.4f}")
    
    return crf_model, training_stats