"""
Synthetic data generator for the Luminara CRF model.

This module creates realistic simulated data for testing and development
of the Conditional Random Field model for AI investment worthiness prediction.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def generate_synthetic_data(
    n_samples: int = 1000,
    random_state: Optional[int] = 42,
    cardinalities: Optional[Dict[str, int]] = None
) -> pd.DataFrame:
    """
    Generate synthetic data for the Luminara CRF model.
    
    Args:
        n_samples: Number of data samples to generate
        random_state: Random seed for reproducibility
        cardinalities: Dictionary mapping variable names to their cardinalities (number of possible values)
                      If None, default cardinalities will be used
        
    Returns:
        DataFrame containing synthetic data with all model variables
    """
    # Set random seed for reproducibility
    np.random.seed(random_state)
    
    # Default cardinalities (discrete states for each variable)
    if cardinalities is None:
        cardinalities = {
            # Market clique
            "M": 3,    # Market size: small (0), medium (1), large (2)
            "C": 3,    # Competitor penetration: low (0), medium (1), high (2)
            "G": 3,    # Growth rate: slow (0), moderate (1), fast (2)
            "RP": 3,   # Revenue potential: low (0), medium (1), high (2)
            
            # Development clique
            "B": 3,    # Bug density: low (0), medium (1), high (2)
            "DC": 3,   # Development cost: low (0), medium (1), high (2)
            "DT": 3,   # Development time: short (0), medium (1), long (2)
            "FC": 3,   # Feature completion: low (0), medium (1), high (2)
            
            # Customer clique
            "CAC": 3,  # Customer acquisition cost: low (0), medium (1), high (2)
            "R": 3,    # Return rate: low (0), medium (1), high (2)
            "ATA": 3,  # Average time to action: fast (0), medium (1), slow (2)
            
            # Target variable
            "Y": 2     # Investment worthiness: not worthy (0), worthy (1)
        }
    
    # Create empty DataFrame
    data = pd.DataFrame()
    
    # Generate data for Market clique variables
    data["M"] = np.random.randint(0, cardinalities["M"], n_samples)
    data["C"] = np.random.randint(0, cardinalities["C"], n_samples)
    data["G"] = np.random.randint(0, cardinalities["G"], n_samples)
    
    # Revenue potential depends on Market size, Competitor penetration, and Growth rate
    # Higher Market size and Growth rate increase Revenue potential
    # Higher Competitor penetration decreases Revenue potential
    data["RP"] = np.clip(
        np.round(0.5 * data["M"] - 0.3 * data["C"] + 0.4 * data["G"] + 
                 np.random.normal(0, 0.5, n_samples)).astype(int),
        0, cardinalities["RP"] - 1
    )
    
    # Generate data for Development clique variables
    data["B"] = np.random.randint(0, cardinalities["B"], n_samples)
    data["DC"] = np.random.randint(0, cardinalities["DC"], n_samples)
    data["DT"] = np.random.randint(0, cardinalities["DT"], n_samples)
    
    # Feature completion depends on Bug density, Development cost, and Development time
    # Higher Development cost and time increase Feature completion
    # Higher Bug density decreases Feature completion
    data["FC"] = np.clip(
        np.round(-0.4 * data["B"] + 0.3 * data["DC"] + 0.5 * data["DT"] + 
                 np.random.normal(0, 0.5, n_samples)).astype(int),
        0, cardinalities["FC"] - 1
    )
    
    # Generate data for Customer clique variables
    data["CAC"] = np.random.randint(0, cardinalities["CAC"], n_samples)
    data["R"] = np.random.randint(0, cardinalities["R"], n_samples)
    
    # Average time to action depends on Customer acquisition cost and Return rate
    # Higher Return rate decreases Average time to action (faster action)
    # Higher Customer acquisition cost increases Average time to action (slower action)
    data["ATA"] = np.clip(
        np.round(0.4 * data["CAC"] - 0.5 * data["R"] + 
                 np.random.normal(0, 0.5, n_samples)).astype(int),
        0, cardinalities["ATA"] - 1
    )
    
    # Generate Investment worthiness (target variable)
    # Depends on Revenue potential (RP), Development cost (DC), and Return rate (R)
    # as specified in bridge cliques
    
    # Create a utility score that determines worthiness
    utility = (
        0.6 * data["RP"]                # Higher Revenue potential increases worthiness
        - 0.4 * data["DC"]              # Higher Development cost decreases worthiness
        + 0.3 * data["R"]               # Higher Return rate increases worthiness
        + 0.2 * data["FC"]              # Higher Feature completion increases worthiness
        - 0.1 * data["B"]               # Higher Bug density decreases worthiness
        - 0.2 * data["ATA"]             # Higher time to action decreases worthiness
        + np.random.normal(0, 0.7, n_samples)  # Add random noise
    )
    
    # Convert utility to binary worthiness (0 or 1)
    threshold = np.median(utility)  # Use median as threshold for balanced classes
    data["Y"] = (utility > threshold).astype(int)
    
    logger.info(f"Generated {n_samples} synthetic data samples")
    
    return data


def save_synthetic_data(
    data: pd.DataFrame,
    base_dir: str = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    random_state: int = 42
) -> Dict[str, str]:
    """
    Save synthetic data to train, validation, and test sets.
    
    Args:
        data: DataFrame containing the synthetic data
        base_dir: Base directory to save the data (default: Luminara/data)
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary mapping set names to file paths
    """
    # Set random seed
    np.random.seed(random_state)
    
    # Determine base directory
    if base_dir is None:
        # Find the Luminara project root
        luminara_root = Path(__file__).resolve().parents[2]  # Go up from src/data to root
        base_dir = luminara_root / "data"
    
    # Create directories if they don't exist
    processed_dir = Path(base_dir) / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Shuffle the data
    data = data.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Calculate split indices
    n_samples = len(data)
    train_end = int(n_samples * train_ratio)
    val_end = train_end + int(n_samples * val_ratio)
    
    # Split the data
    train_data = data.iloc[:train_end]
    val_data = data.iloc[train_end:val_end]
    test_data = data.iloc[val_end:]
    
    # Save to CSV files
    train_path = processed_dir / "train.csv"
    val_path = processed_dir / "val.csv"
    test_path = processed_dir / "test.csv"
    
    train_data.to_csv(train_path, index=False)
    val_data.to_csv(val_path, index=False)
    test_data.to_csv(test_path, index=False)
    
    # Also save the full dataset
    full_path = processed_dir / "full_dataset.csv"
    data.to_csv(full_path, index=False)
    
    logger.info(f"Saved {len(train_data)} training samples to {train_path}")
    logger.info(f"Saved {len(val_data)} validation samples to {val_path}")
    logger.info(f"Saved {len(test_data)} test samples to {test_path}")
    
    return {
        "train": str(train_path),
        "val": str(val_path),
        "test": str(test_path),
        "full": str(full_path)
    }


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Generate synthetic data
    data = generate_synthetic_data(n_samples=2000)
    
    # Save the data
    save_synthetic_data(data)
    
    logger.info("Synthetic data generation completed successfully")
