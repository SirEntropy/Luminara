"""
Synthetic data generation for the Luminara CRF model.

This module creates synthetic datasets that align with the clique structure
of the Luminara CRF model. The generated data is designed to validate the model
constraints rather than represent real-world distributions.
"""

import numpy as np
import pandas as pd
import os
from typing import Dict, Tuple, List
from sklearn.model_selection import train_test_split
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define cardinalities for each variable
DEFAULT_CARDINALITIES = {
    "M": 3,    # Market size: Small (0), Medium (1), Large (2)
    "C": 3,    # Competitor penetration: Low (0), Medium (1), High (2)
    "G": 3,    # Growth rate: Slow (0), Moderate (1), Rapid (2)
    "RP": 3,   # Revenue potential: Low (0), Medium (1), High (2)
    "B": 3,    # Bug density: Low (0), Medium (1), High (2)
    "DC": 3,   # Development cost: Low (0), Medium (1), High (2)
    "DT": 3,   # Development time: Short (0), Medium (1), Long (2)
    "FC": 3,   # Feature completion: Incomplete (0), Partial (1), Complete (2)
    "CAC": 3,  # Customer acquisition cost: Low (0), Medium (1), High (2)
    "R": 3,    # Return rate: Low (0), Medium (1), High (2)
    "ATA": 3,  # Average time to action: Quick (0), Medium (1), Slow (2)
    "Y": 2     # Investment worthiness: Not worthy (0), Worthy (1)
}

# Define market clique relationships
def generate_market_clique(n_samples: int, cardinalities: Dict[str, int]) -> pd.DataFrame:
    """
    Generate data for the market clique with strong correlations.
    
    Market clique: M, C, G, RP
    
    Returns:
        DataFrame with synthetic market clique data
    """
    # Create a dataframe for market clique
    df = pd.DataFrame()
    
    # Market size (M)
    df["M"] = np.random.randint(0, cardinalities["M"], size=n_samples)
    
    # Competitor penetration (C) - loosely correlated with M
    probabilities = {
        0: [0.7, 0.2, 0.1],  # Small market tends to have low competition
        1: [0.3, 0.5, 0.2],  # Medium market has medium competition
        2: [0.1, 0.3, 0.6]   # Large market attracts high competition
    }
    
    df["C"] = [np.random.choice(range(cardinalities["C"]), p=probabilities[m]) 
                for m in df["M"]]
    
    # Growth rate (G) - correlated with M and C
    # If market is large and competition is low, growth tends to be high
    df["G"] = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        m, c = df.loc[i, ["M", "C"]]
        
        if m == 2 and c <= 1:  # Large market with low/medium competition
            p = [0.1, 0.2, 0.7]  # Higher chance of rapid growth
        elif m == 0 and c == 2:  # Small market with high competition
            p = [0.7, 0.2, 0.1]  # Higher chance of slow growth
        else:
            p = [0.3, 0.4, 0.3]  # Otherwise balanced
        
        df.loc[i, "G"] = np.random.choice(range(cardinalities["G"]), p=p)
    
    # Revenue potential (RP) - strongly influenced by M, G, and somewhat by C
    df["RP"] = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        m, c, g = df.loc[i, ["M", "C", "G"]]
        
        # Logic: High M + High G - High C = High RP
        score = m + g - (c/2)  # weight competition less
        
        if score > 2.5:
            p = [0.1, 0.2, 0.7]  # High probability of high revenue potential
        elif score < 0.5:
            p = [0.7, 0.2, 0.1]  # High probability of low revenue potential
        else:
            p = [0.3, 0.4, 0.3]  # Balanced probability for medium cases
        
        df.loc[i, "RP"] = np.random.choice(range(cardinalities["RP"]), p=p)
    
    return df

# Define development clique relationships
def generate_development_clique(n_samples: int, cardinalities: Dict[str, int]) -> pd.DataFrame:
    """
    Generate data for the development clique with strong correlations.
    
    Development clique: B, DC, DT, FC
    
    Returns:
        DataFrame with synthetic development clique data
    """
    # Create a dataframe for development clique
    df = pd.DataFrame()
    
    # Bug density (B)
    df["B"] = np.random.randint(0, cardinalities["B"], size=n_samples)
    
    # Development cost (DC) - correlated with B
    df["DC"] = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        b = df.loc[i, "B"]
        
        if b == 2:  # High bug density
            p = [0.1, 0.3, 0.6]  # Higher chance of high development cost
        elif b == 0:  # Low bug density
            p = [0.6, 0.3, 0.1]  # Higher chance of low development cost
        else:
            p = [0.3, 0.4, 0.3]  # Balanced
        
        df.loc[i, "DC"] = np.random.choice(range(cardinalities["DC"]), p=p)
    
    # Development time (DT) - correlated with B and DC
    df["DT"] = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        b, dc = df.loc[i, ["B", "DC"]]
        
        # High bugs and high cost usually mean longer development time
        score = (b + dc) / 2
        
        if score > 1.5:
            p = [0.1, 0.3, 0.6]  # Higher chance of long development time
        elif score < 0.5:
            p = [0.6, 0.3, 0.1]  # Higher chance of short development time
        else:
            p = [0.3, 0.4, 0.3]  # Balanced
        
        df.loc[i, "DT"] = np.random.choice(range(cardinalities["DT"]), p=p)
    
    # Feature completion (FC) - inversely correlated with B, but still influenced by DT
    df["FC"] = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        b, dt = df.loc[i, ["B", "DT"]]
        
        # More bugs = lower feature completion, but longer development time can mitigate
        score = dt - b
        
        if score > 0.5:
            p = [0.1, 0.3, 0.6]  # Higher chance of complete features
        elif score < -0.5:
            p = [0.6, 0.3, 0.1]  # Higher chance of incomplete features
        else:
            p = [0.3, 0.4, 0.3]  # Balanced
        
        df.loc[i, "FC"] = np.random.choice(range(cardinalities["FC"]), p=p)
    
    return df

# Define customer clique relationships
def generate_customer_clique(n_samples: int, cardinalities: Dict[str, int]) -> pd.DataFrame:
    """
    Generate data for the customer clique with strong correlations.
    
    Customer clique: CAC, R, ATA
    
    Returns:
        DataFrame with synthetic customer clique data
    """
    # Create a dataframe for customer clique
    df = pd.DataFrame()
    
    # Customer acquisition cost (CAC)
    df["CAC"] = np.random.randint(0, cardinalities["CAC"], size=n_samples)
    
    # Return rate (R) - inversely correlated with CAC
    df["R"] = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        cac = df.loc[i, "CAC"]
        
        if cac == 2:  # High acquisition cost
            p = [0.6, 0.3, 0.1]  # Higher chance of low return rate
        elif cac == 0:  # Low acquisition cost
            p = [0.1, 0.3, 0.6]  # Higher chance of high return rate
        else:
            p = [0.3, 0.4, 0.3]  # Balanced
        
        df.loc[i, "R"] = np.random.choice(range(cardinalities["R"]), p=p)
    
    # Average time to action (ATA) - moderate correlation with R
    df["ATA"] = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        r = df.loc[i, "R"]
        
        if r == 2:  # High return rate
            p = [0.5, 0.3, 0.2]  # More likely to have quick action time
        elif r == 0:  # Low return rate
            p = [0.2, 0.3, 0.5]  # More likely to have slow action time
        else:
            p = [0.3, 0.4, 0.3]  # Balanced
        
        df.loc[i, "ATA"] = np.random.choice(range(cardinalities["ATA"]), p=p)
    
    return df

# Generate target variable based on bridge cliques
def generate_target_variable(df: pd.DataFrame, cardinalities: Dict[str, int]) -> pd.DataFrame:
    """
    Generate the target variable Y (Investment worthiness) based on bridge cliques.
    
    Bridge cliques: (RP, Y), (DC, Y), (R, Y)
    
    Args:
        df: DataFrame with all features
        
    Returns:
        DataFrame with added target variable Y
    """
    # Make a copy to avoid modifying the original
    result_df = df.copy()
    
    # Create Y variable (Investment worthiness)
    n_samples = len(df)
    result_df["Y"] = np.zeros(n_samples, dtype=int)
    
    # Track scores for evaluation
    scores = []
    
    for i in range(n_samples):
        # Get primary bridge variables
        rp = df.loc[i, "RP"]  # Revenue potential
        dc = df.loc[i, "DC"]  # Development cost
        r = df.loc[i, "R"]    # Return rate
        
        # Get secondary variables that provide additional signal
        g = df.loc[i, "G"]    # Growth rate
        m = df.loc[i, "M"]    # Market size
        b = df.loc[i, "B"]    # Bug density
        dt = df.loc[i, "DT"]  # Development time
        
        # Primary scoring - bridge variables have strongest influence
        rp_score = rp * 2.0           # 0, 2, 4 (higher weight)
        dc_score = (2 - dc) * 2.0     # 4, 2, 0 (inverse relationship, higher weight)
        r_score = r * 2.0             # 0, 2, 4 (higher weight)
        
        # Secondary scoring - indirect influences with lower weights
        g_score = g * 0.5             # 0, 0.5, 1 (growth boosts worthiness)
        m_score = m * 0.5             # 0, 0.5, 1 (market size boosts worthiness)
        b_score = (2 - b) * 0.5       # 1, 0.5, 0 (fewer bugs is better - inverse)
        dt_score = (2 - dt) * 0.5     # 1, 0.5, 0 (shorter dev time is better - inverse)
        
        # Calculate total score (max possible: 15)
        total_score = rp_score + dc_score + r_score + g_score + m_score + b_score + dt_score
        
        # Calculate probability with sigmoid to ensure smooth transition and proper bounds
        # Sigmoid function: 1 / (1 + exp(-k * (x - x0)))
        # where k controls steepness and x0 is center point
        
        # Center around 7.5 (half of max score) with moderate steepness
        p_worthy = 1 / (1 + np.exp(-0.8 * (total_score - 7.5)))
        
        # Special case handling for extreme configurations
        if rp == 2 and dc == 0 and r == 2:
            p_worthy = max(p_worthy, 0.98)  # Almost certainly worthy
        elif rp == 0 and dc == 2 and r == 0:
            p_worthy = min(p_worthy, 0.02)  # Almost certainly not worthy
        
        # Record score for analysis
        scores.append(total_score)
        
        # Use a deterministic threshold with small randomness
        # This makes the model more learnable while keeping some realism
        base_threshold = 0.5
        
        # Only add randomness to borderline cases
        if 0.35 < p_worthy < 0.65:
            # Add up to ±0.15 randomness for borderline cases
            random_adjustment = (np.random.random() * 0.3) - 0.15
            threshold = base_threshold + random_adjustment
        else:
            # Use fixed threshold for clear cases
            threshold = base_threshold
        
        # Set Y value
        result_df.loc[i, "Y"] = 1 if p_worthy > threshold else 0
    
    # Log score distribution
    scores = np.array(scores)
    logger.info(f"Target score distribution: min={scores.min():.2f}, max={scores.max():.2f}, "
                f"mean={scores.mean():.2f}, median={np.median(scores):.2f}")
    
    return result_df

def generate_synthetic_dataset(
    n_samples: int = 1000, 
    cardinalities: Dict[str, int] = None,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate a synthetic dataset that follows the Luminara CRF model structure.
    
    Args:
        n_samples: Number of samples to generate
        cardinalities: Dictionary mapping variable names to their cardinalities
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with synthetic data
    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Use default cardinalities if not provided
    if cardinalities is None:
        cardinalities = DEFAULT_CARDINALITIES
    
    # Generate data for each clique
    market_df = generate_market_clique(n_samples, cardinalities)
    development_df = generate_development_clique(n_samples, cardinalities)
    customer_df = generate_customer_clique(n_samples, cardinalities)
    
    # Combine all features
    combined_df = pd.concat([market_df, development_df, customer_df], axis=1)
    
    # Generate target variable based on bridge cliques
    final_df = generate_target_variable(combined_df, cardinalities)
    
    logger.info(f"Generated synthetic dataset with {n_samples} samples")
    
    return final_df

def generate_and_split_dataset(
    n_samples: int = 1000,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    output_dir: str = "data/synthetic",
    seed: int = 42
) -> Dict[str, pd.DataFrame]:
    """
    Generate a synthetic dataset and split it into train, validation, and test sets.
    
    Args:
        n_samples: Number of samples to generate
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        test_ratio: Proportion of data for testing
        output_dir: Directory to save the datasets
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with train, val, and test DataFrames
    """
    # Validate ratios
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-10:
        raise ValueError("Train, validation, and test ratios must sum to 1")
    
    # Generate full dataset
    full_dataset = generate_synthetic_dataset(n_samples=n_samples, seed=seed)
    
    # Split into train and temp (val + test)
    train_data, temp_data = train_test_split(
        full_dataset, 
        train_size=train_ratio, 
        random_state=seed
    )
    
    # Calculate the relative ratio for val from the temp data
    val_test_ratio = val_ratio / (val_ratio + test_ratio)
    
    # Split temp into val and test
    val_data, test_data = train_test_split(
        temp_data, 
        train_size=val_test_ratio, 
        random_state=seed
    )
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save datasets to CSV files
    full_dataset.to_csv(os.path.join(output_dir, "full_dataset.csv"), index=False)
    train_data.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val_data.to_csv(os.path.join(output_dir, "val.csv"), index=False)
    test_data.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    
    logger.info(f"Saved datasets to {output_dir}")
    logger.info(f"Train: {len(train_data)} samples")
    logger.info(f"Validation: {len(val_data)} samples")
    logger.info(f"Test: {len(test_data)} samples")
    
    return {
        "full_dataset": full_dataset,
        "train": train_data,
        "val": val_data,
        "test": test_data
    }

if __name__ == "__main__":
    # Generate and save datasets
    datasets = generate_and_split_dataset(
        n_samples=2000,  # Generate 2000 samples
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        output_dir="/Users/lianghaochen/Luminara/data/synthetic",
        seed=42
    )
    
    # Print class distribution
    logger.info(f"Class distribution in full dataset: {datasets['full_dataset']['Y'].value_counts(normalize=True)}")
    logger.info(f"Class distribution in train dataset: {datasets['train']['Y'].value_counts(normalize=True)}")
    logger.info(f"Class distribution in validation dataset: {datasets['val']['Y'].value_counts(normalize=True)}")
    logger.info(f"Class distribution in test dataset: {datasets['test']['Y'].value_counts(normalize=True)}")