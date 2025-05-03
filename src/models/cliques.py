"""
Clique definitions for the Luminara CRF model.

This module defines the clique structure used in the Conditional Random Field model
for predicting AI investment worthiness. Cliques represent groups of variables
that have direct probabilistic relationships with each other.

The cliques defined are:
- Market clique: Market size, Competitor penetration, Growth rate, Revenue potential
- Development clique: Bug density, Development cost, Development time, Feature completion
- Customer clique: Customer acquisition cost, Return rate, Average time to action
- Bridge cliques: Connect various cliques to the target variable (Investment worthiness)
"""

import numpy as np
from pgmpy.factors.discrete import DiscreteFactor
from typing import Dict, List, Tuple, Any, Optional


class BaseClique:
    """Base class for all cliques in the Luminara CRF model."""
    
    def __init__(self, name: str, variables: List[str], cardinalities: Dict[str, int]):
        """
        Initialize a base clique.
        
        Args:
            name: Name of the clique
            variables: List of variable names in this clique
            cardinalities: Dictionary mapping variable names to their cardinalities
        """
        self.name = name
        self.variables = variables
        self.cardinalities = cardinalities
        self._verify_variables_in_cardinalities()
    
    def _verify_variables_in_cardinalities(self):
        """Verify that all variables have defined cardinalities."""
        for var in self.variables:
            if var not in self.cardinalities:
                raise ValueError(f"Variable {var} in clique {self.name} has no defined cardinality")
    
    def get_cardinality_list(self) -> List[int]:
        """Get cardinalities for all variables in the clique in order."""
        return [self.cardinalities[var] for var in self.variables]
    
    def create_factor(self, values: Optional[np.ndarray] = None) -> DiscreteFactor:
        """
        Create a factor for this clique.
        
        Args:
            values: Values for the factor. If None, initialized with ones.
            
        Returns:
            DiscreteFactor representing this clique
        """
        card_list = self.get_cardinality_list()
        
        if values is None:
            # Initialize with uniform distribution
            values = np.ones(np.prod(card_list))
        
        return DiscreteFactor(
            variables=self.variables,
            cardinality=card_list,
            values=values
        )


class MarketClique(BaseClique):
    """
    Market clique representing the relationship between:
    - Market size (M)
    - Competitor penetration (C)
    - Growth rate (G)
    - Revenue potential (RP)
    """
    
    def __init__(self, cardinalities: Dict[str, int]):
        """
        Initialize the Market clique.
        
        Args:
            cardinalities: Dictionary mapping variable names to their cardinalities
        """
        variables = ["M", "C", "G", "RP"]
        super().__init__("Market", variables, cardinalities)


class DevelopmentClique(BaseClique):
    """
    Development clique representing the relationship between:
    - Bug density (B)
    - Development cost (DC)
    - Development time (DT)
    - Feature completion (FC)
    """
    
    def __init__(self, cardinalities: Dict[str, int]):
        """
        Initialize the Development clique.
        
        Args:
            cardinalities: Dictionary mapping variable names to their cardinalities
        """
        variables = ["B", "DC", "DT", "FC"]
        super().__init__("Development", variables, cardinalities)


class CustomerClique(BaseClique):
    """
    Customer clique representing the relationship between:
    - Customer acquisition cost (CAC)
    - Return rate (R)
    - Average time to action (ATA)
    """
    
    def __init__(self, cardinalities: Dict[str, int]):
        """
        Initialize the Customer clique.
        
        Args:
            cardinalities: Dictionary mapping variable names to their cardinalities
        """
        variables = ["CAC", "R", "ATA"]
        super().__init__("Customer", variables, cardinalities)


class BridgeClique(BaseClique):
    """
    Bridge clique connecting one or more domain variables to the target variable Y 
    (Investment worthiness).
    """
    
    def __init__(self, domain_var: str, cardinalities: Dict[str, int]):
        """
        Initialize a Bridge clique that connects a domain variable to the target variable.
        
        Args:
            domain_var: The domain variable to connect to Y
            cardinalities: Dictionary mapping variable names to their cardinalities
        """
        variables = [domain_var, "Y"]
        super().__init__(f"Bridge_{domain_var}_Y", variables, cardinalities)


def create_all_cliques(cardinalities: Dict[str, int]) -> Dict[str, BaseClique]:
    """
    Create all cliques defined in the Luminara CRF model.
    
    Args:
        cardinalities: Dictionary mapping all variable names to their cardinalities
        
    Returns:
        Dictionary mapping clique names to clique objects
    """
    # Main domain cliques
    market_clique = MarketClique(cardinalities)
    development_clique = DevelopmentClique(cardinalities)
    customer_clique = CustomerClique(cardinalities)
    
    # Bridge cliques connecting to the target variable Y
    rp_y_bridge = BridgeClique("RP", cardinalities)  # Revenue potential to Y
    dc_y_bridge = BridgeClique("DC", cardinalities)  # Development cost to Y
    r_y_bridge = BridgeClique("R", cardinalities)    # Return rate to Y
    
    cliques = {
        market_clique.name: market_clique,
        development_clique.name: development_clique,
        customer_clique.name: customer_clique,
        rp_y_bridge.name: rp_y_bridge,
        dc_y_bridge.name: dc_y_bridge,
        r_y_bridge.name: r_y_bridge
    }
    
    return cliques