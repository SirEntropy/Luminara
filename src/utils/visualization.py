"""
Visualization utilities for the Luminara CRF model.

This module provides functions for visualizing the factor graph
and other aspects of the Conditional Random Field model.
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import logging

logger = logging.getLogger(__name__)


def plot_factor_graph(factor_graph, figsize=(12, 10), node_size=1000, 
                      variable_color='skyblue', factor_color='lightgreen'):
    """
    Plot the factor graph structure.
    
    Parameters
    ----------
    factor_graph : pgmpy.models.FactorGraph
        Factor graph to visualize
    figsize : tuple, optional
        Figure size (width, height)
    node_size : int, optional
        Size of nodes in the plot
    variable_color : str, optional
        Color for variable nodes
    factor_color : str, optional
        Color for factor nodes
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the factor graph plot
    """
    # Convert pgmpy factor graph to networkx for visualization
    G = nx.Graph()
    
    # Add all nodes
    for node in factor_graph.nodes():
        G.add_node(node)
    
    # Add all edges
    for u, v in factor_graph.edges():
        G.add_edge(u, v)
    
    # Separate variable and factor nodes
    variable_nodes = [node for node in G.nodes() if not isinstance(node, str) 
                      or not node.startswith('phi_')]
    factor_nodes = [node for node in G.nodes() if isinstance(node, str) 
                    and node.startswith('phi_')]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Define position layout for nodes
    pos = nx.spring_layout(G, seed=42)
    
    # Draw variable nodes
    nx.draw_networkx_nodes(G, pos, 
                           nodelist=variable_nodes,
                           node_color=variable_color, 
                           node_size=node_size, 
                           ax=ax,
                           edgecolors='black')
    
    # Draw factor nodes
    nx.draw_networkx_nodes(G, pos, 
                           nodelist=factor_nodes,
                           node_color=factor_color, 
                           node_size=node_size*0.8,
                           node_shape='s',  # squares for factors
                           ax=ax,
                           edgecolors='black')
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=2, ax=ax)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold', ax=ax)
    
    # Set title and remove axis
    plt.title('Factor Graph Structure', fontsize=18)
    plt.axis('off')
    
    return fig


def plot_clique_structure(cliques, figsize=(10, 8)):
    """
    Plot the clique structure of the CRF model.
    
    Parameters
    ----------
    cliques : list
        List of clique objects
    figsize : tuple, optional
        Figure size (width, height)
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the clique structure plot
    """
    # Create graph to represent clique connections
    G = nx.Graph()
    
    # Add all variables as nodes
    all_variables = set()
    for clique in cliques:
        all_variables.update(clique.variables)
    
    for var in all_variables:
        G.add_node(var)
    
    # Add edges for variables that appear in the same clique
    for clique in cliques:
        for i, var1 in enumerate(clique.variables):
            for var2 in clique.variables[i+1:]:
                G.add_edge(var1, var2)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Define node colors
    # Market clique (blue)
    market_vars = ['M', 'C', 'G', 'RP']
    # Development clique (green)
    dev_vars = ['B', 'DC', 'DT', 'FC']
    # Customer clique (orange)
    customer_vars = ['CAC', 'R', 'ATA']
    # Target variable (red)
    target_vars = ['Y']
    
    # Define position layout
    pos = nx.spring_layout(G, seed=42)
    
    # Draw nodes by clique
    nx.draw_networkx_nodes(G, pos, nodelist=market_vars, 
                         node_color='royalblue', node_size=800, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=dev_vars, 
                         node_color='forestgreen', node_size=800, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=customer_vars, 
                         node_color='darkorange', node_size=800, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=target_vars, 
                         node_color='crimson', node_size=1000, ax=ax)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=2, ax=ax)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold', ax=ax)
    
    # Set title and remove axis
    plt.title('Clique Structure', fontsize=18)
    plt.axis('off')
    
    # Add legend
    market_patch = plt.Circle((0, 0), 0.1, fc="royalblue")
    dev_patch = plt.Circle((0, 0), 0.1, fc="forestgreen")
    customer_patch = plt.Circle((0, 0), 0.1, fc="darkorange")
    target_patch = plt.Circle((0, 0), 0.1, fc="crimson")
    
    plt.legend([market_patch, dev_patch, customer_patch, target_patch],
               ['Market', 'Development', 'Customer', 'Target'],
               loc='upper right')
    
    return fig


def plot_factor_parameters(model, factor_index, figsize=(10, 8)):
    """
    Plot the parameters of a specific factor in the CRF model.
    
    Parameters
    ----------
    model : InvestmentCRF
        Trained CRF model
    factor_index : int
        Index of the factor to visualize
    figsize : tuple, optional
        Figure size (width, height)
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the factor parameters plot
    """
    # Get the factors from the model
    factors = list(model.model.get_factors())
    
    if factor_index >= len(factors):
        raise ValueError(f"Factor index {factor_index} out of range, model has {len(factors)} factors")
    
    # Get the selected factor
    factor = factors[factor_index]
    variables = factor.variables
    values = factor.values
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # For 2D factors, display as heatmap
    if len(variables) == 2:
        # Reshape values if needed
        values_2d = values.reshape(factor.cardinality)
        
        # Create heatmap
        im = ax.imshow(values_2d, cmap='viridis')
        
        # Add colorbar
        plt.colorbar(im, ax=ax)
        
        # Add labels
        ax.set_title(f"Factor Parameters: {variables[0]}-{variables[1]}", fontsize=14)
        ax.set_xlabel(variables[1], fontsize=12)
        ax.set_ylabel(variables[0], fontsize=12)
        
        # Add ticks
        ax.set_xticks(np.arange(factor.cardinality[1]))
        ax.set_yticks(np.arange(factor.cardinality[0]))
        
        # Add tick labels
        ax.set_xticklabels([f"{variables[1]}={i}" for i in range(factor.cardinality[1])])
        ax.set_yticklabels([f"{variables[0]}={i}" for i in range(factor.cardinality[0])])
    
    # For 1D factors, display as bar chart
    elif len(variables) == 1:
        ax.bar(range(len(values)), values)
        ax.set_title(f"Factor Parameters: {variables[0]}", fontsize=14)
        ax.set_xlabel(variables[0], fontsize=12)
        ax.set_ylabel("Potential Value", fontsize=12)
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels([f"{variables[0]}={i}" for i in range(len(values))])
    
    # For higher dimensional factors, display first two dimensions
    else:
        logger.warning(f"Factor has {len(variables)} dimensions, displaying first two dimensions only")
        
        # Calculate average over all other dimensions
        values_nd = values.reshape(factor.cardinality)
        axes_to_avg = tuple(range(2, len(variables)))
        if axes_to_avg:
            values_2d = np.mean(values_nd, axis=axes_to_avg)
        else:
            values_2d = values_nd
            
        # Create heatmap
        im = ax.imshow(values_2d, cmap='viridis')
        
        # Add colorbar
        plt.colorbar(im, ax=ax)
        
        # Add labels
        ax.set_title(f"Factor Parameters: {variables[0]}-{variables[1]} (averaged)", fontsize=14)
        ax.set_xlabel(variables[1], fontsize=12)
        ax.set_ylabel(variables[0], fontsize=12)
        
        # Add ticks
        ax.set_xticks(np.arange(factor.cardinality[1]))
        ax.set_yticks(np.arange(factor.cardinality[0]))
        
        # Add tick labels
        ax.set_xticklabels([f"{variables[1]}={i}" for i in range(factor.cardinality[1])])
        ax.set_yticklabels([f"{variables[0]}={i}" for i in range(factor.cardinality[0])])
    
    plt.tight_layout()
    return fig


def plot_confusion_matrix(y_true, y_pred, figsize=(8, 6)):
    """
    Plot confusion matrix for investment worthiness prediction.
    
    Parameters
    ----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    figsize : tuple, optional
        Figure size (width, height)
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the confusion matrix plot
    """
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot confusion matrix as heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    
    # Add labels
    ax.set_title('Confusion Matrix', fontsize=14)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    
    # Set tick labels
    ax.set_xticklabels(['Not Worthy (0)', 'Worthy (1)'])
    ax.set_yticklabels(['Not Worthy (0)', 'Worthy (1)'])
    
    return fig


def plot_prediction_distribution(predictions, figsize=(10, 6)):
    """
    Plot distribution of investment worthiness predictions.
    
    Parameters
    ----------
    predictions : dict or array-like
        If dict: Dictionary mapping class labels to probabilities
        If array-like: Binary predictions
    figsize : tuple, optional
        Figure size (width, height)
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the prediction distribution plot
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Check if predictions is a dictionary of probabilities
    if isinstance(predictions, dict):
        # Plot probability distribution
        classes = sorted(predictions.keys())
        probabilities = [predictions[c] for c in classes]
        
        ax.bar(classes, probabilities, color=['salmon', 'skyblue'])
        ax.set_title('Prediction Probability Distribution', fontsize=14)
        ax.set_xlabel('Investment Worthiness Class', fontsize=12)
        ax.set_ylabel('Probability', fontsize=12)
        ax.set_xticks(classes)
        ax.set_xticklabels(['Not Worthy (0)', 'Worthy (1)'])
        ax.set_ylim(0, 1)
        
    else:
        # Count occurrences of each class
        unique, counts = np.unique(predictions, return_counts=True)
        
        # Plot histogram
        ax.bar(unique, counts, color=['salmon', 'skyblue'])
        ax.set_title('Prediction Distribution', fontsize=14)
        ax.set_xlabel('Investment Worthiness Class', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Not Worthy (0)', 'Worthy (1)'])
    
    return fig


def plot_variable_distributions(data, figsize=(12, 10)):
    """
    Plot distributions of all variables in the dataset.
    
    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing the variables
    figsize : tuple, optional
        Figure size (width, height)
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the variable distributions plot
    """
    import seaborn as sns
    
    # Get all variables
    variables = data.columns
    
    # Calculate number of rows and columns for subplots
    n_vars = len(variables)
    n_cols = 3
    n_rows = (n_vars + n_cols - 1) // n_cols
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    # Plot distribution for each variable
    for i, var in enumerate(variables):
        ax = axes[i]
        
        # Get variable data
        var_data = data[var]
        
        # Plot count distribution
        sns.countplot(x=var_data, ax=ax)
        
        # Add labels
        ax.set_title(f'Distribution of {var}', fontsize=12)
        ax.set_xlabel(var, fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        
        # For Y variable, use more descriptive labels
        if var == 'Y':
            ax.set_xticklabels(['Not Worthy (0)', 'Worthy (1)'])
    
    # Remove any unused subplots
    for i in range(n_vars, len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    return fig


def plot_variable_correlations(data, figsize=(12, 10)):
    """
    Plot correlation matrix between all variables.
    
    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing the variables
    figsize : tuple, optional
        Figure size (width, height)
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the correlation matrix plot
    """
    import seaborn as sns
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate correlation matrix
    corr = data.corr()
    
    # Plot heatmap
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
    
    # Add labels
    ax.set_title('Variable Correlations', fontsize=14)
    
    return fig


def plot_feature_importance(data, target='Y', figsize=(10, 6)):
    """
    Plot feature importance for predicting the target variable.
    
    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing features and target
    target : str, optional
        Name of the target variable
    figsize : tuple, optional
        Figure size (width, height)
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the feature importance plot
    """
    from sklearn.ensemble import RandomForestClassifier
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Separate features and target
    X = data.drop(columns=[target])
    y = data[target]
    
    # Train a random forest to get feature importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    # Get feature importance
    importance = rf.feature_importances_
    
    # Sort features by importance
    indices = np.argsort(importance)
    features = X.columns[indices]
    
    # Plot feature importance
    ax.barh(range(len(indices)), importance[indices], align='center')
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels(features)
    
    # Add labels
    ax.set_title(f'Feature Importance for Predicting {target}', fontsize=14)
    ax.set_xlabel('Importance', fontsize=12)
    
    plt.tight_layout()
    return fig


def plot_marginal_distributions(beliefs):
    """
    Plot marginal distributions for all variables.
    
    Parameters
    ----------
    beliefs : dict
        Dictionary of belief distributions from belief propagation
        
    Returns
    -------
    dict
        Dictionary mapping variable names to figure objects
    """
    figures = {}
    
    for var, belief in beliefs.items():
        # Skip factors
        if isinstance(var, str) and var.startswith('phi_'):
            continue
            
        # Get the values
        values = belief.values
        card = len(values)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot the marginal distribution
        bars = ax.bar(range(card), values)
        
        # Add value labels on top of bars
        for i, v in enumerate(values):
            ax.text(i, v + 0.01, f"{v:.3f}", ha='center')
        
        # Set labels and title
        ax.set_title(f"Marginal Distribution for {var}")
        ax.set_xlabel("State")
        ax.set_ylabel("Probability")
        ax.set_xticks(range(card))
        ax.set_ylim(0, 1.1)
        
        # Store the figure
        figures[var] = fig
    
    return figures


def plot_factor_importance(importance_scores, figsize=(10, 6)):
    """
    Plot the importance of different factors in the model.
    
    Parameters
    ----------
    importance_scores : dict
        Dictionary mapping factor names to importance scores
    figsize : tuple, optional
        Figure size (width, height)
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the factor importance plot
    """
    # Sort factors by importance
    sorted_items = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
    factors = [item[0] for item in sorted_items]
    scores = [item[1] for item in sorted_items]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create horizontal bar chart
    bars = ax.barh(factors, scores, color='skyblue')
    
    # Add value labels
    for i, v in enumerate(scores):
        ax.text(v + 0.01, i, f"{v:.4f}", va='center')
    
    # Set labels and title
    ax.set_title("Factor Importance")
    ax.set_xlabel("Importance Score")
    ax.set_ylabel("Factor")
    
    # Adjust layout
    plt.tight_layout()
    
    return fig