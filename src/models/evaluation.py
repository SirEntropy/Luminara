"""
Evaluation module for the Luminara CRF model.

This module provides evaluation metrics and utilities to assess the performance
of the Conditional Random Field model for AI investment worthiness prediction.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
import logging

from pgmpy.models import FactorGraph
from .inference import batch_inference, predict_investment_worthiness

logger = logging.getLogger(__name__)


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute accuracy score.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        
    Returns:
        Accuracy score (0.0 to 1.0)
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: y_true has {len(y_true)} items while y_pred has {len(y_pred)} items")
    
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    return correct / len(y_true)


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Compute confusion matrix.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        
    Returns:
        Confusion matrix as numpy array
    """
    return confusion_matrix(y_true, y_pred)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, probas: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Compute evaluation metrics for classification.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        probas: Predicted probabilities for positive class (for ROC/AUC calculation)
        
    Returns:
        Dictionary containing metrics (accuracy, precision, recall, f1, etc.)
    """
    cm = confusion_matrix(y_true, y_pred)
    
    # For binary classification
    if len(np.unique(y_true)) == 2:
        tn, fp, fn, tp = cm.ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        metrics = {
            'accuracy': (tp + tn) / (tp + tn + fp + fn),
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'specificity': specificity
        }
        
        # If probabilities are provided, compute AUC
        if probas is not None:
            try:
                fpr, tpr, _ = roc_curve(y_true, probas)
                auc_value = auc(fpr, tpr)
                metrics['auc'] = auc_value
            except Exception as e:
                logger.warning(f"Failed to compute AUC: {str(e)}")
    else:
        # For multiclass, use classification_report to get metrics
        report = classification_report(y_true, y_pred, output_dict=True)
        metrics = {
            'accuracy': report['accuracy'],
            'macro_precision': report['macro avg']['precision'],
            'macro_recall': report['macro avg']['recall'],
            'macro_f1': report['macro avg']['f1-score']
        }
    
    return metrics


def evaluate_model(model: FactorGraph, test_data: pd.DataFrame, target_variable: str = 'Y') -> Dict[str, Union[float, Dict]]:
    """
    Evaluate a CRF model on test data.
    
    Args:
        model: FactorGraph model
        test_data: Test dataset containing features and target variable
        target_variable: Name of the target variable
        
    Returns:
        Dictionary containing evaluation metrics
    """
    # Perform batch inference
    results = batch_inference(model, test_data, target_variable, method="map")
    
    # Extract true and predicted values
    y_true = test_data[target_variable].values
    y_pred = results[f"predicted_{target_variable}"].values
    
    # Check if probability columns exist for ROC/AUC calculation
    probas = None
    prob_cols = [col for col in results.columns if col.startswith(f"prob_{target_variable}_")]
    if len(prob_cols) > 0 and f"prob_{target_variable}_1" in results.columns:
        probas = results[f"prob_{target_variable}_1"].values
    
    # Compute metrics
    metrics = compute_metrics(y_true, y_pred, probas)
    
    # Get confusion matrix
    cm = compute_confusion_matrix(y_true, y_pred)
    
    # Get detailed classification report
    class_report = classification_report(y_true, y_pred, output_dict=True)
    
    return {
        'metrics': metrics,
        'confusion_matrix': cm,
        'classification_report': class_report,
        'y_true': y_true,
        'y_pred': y_pred
    }


def evaluate_feature_importance(model: FactorGraph, test_data: pd.DataFrame, target_variable: str = 'Y',
                               n_permutations: int = 10, random_seed: int = 42) -> Dict[str, float]:
    """
    Evaluate feature importance using an advanced permutation method with
    direct effect analysis and stratified sampling.
    
    Args:
        model: FactorGraph model
        test_data: Test dataset containing features and target variable
        target_variable: Name of the target variable
        n_permutations: Number of permutation iterations
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary mapping feature names to importance scores
    """
    np.random.seed(random_seed)
    
    # Get baseline performance
    baseline_results = evaluate_model(model, test_data, target_variable)
    baseline_accuracy = baseline_results['metrics']['accuracy']
    
    # Initialize feature importance dictionary
    feature_cols = [col for col in test_data.columns if col != target_variable]
    importance = {col: 0.0 for col in feature_cols}
    
    # Get clique structure for grouped analysis
    clique_groups = {
        'market': ['M', 'C', 'G', 'RP'],
        'development': ['B', 'DC', 'DT', 'FC'],
        'customer': ['CAC', 'R', 'ATA'],
        'bridge': ['RP', 'DC', 'R']  # Note: RP, DC, R are in multiple groups
    }
    
    # Helper function for direct effect analysis
    def analyze_direct_effect(feature):
        # Only perform direct effect analysis on Bridge variables and their close connections
        if feature not in clique_groups['bridge'] and \
           not any(feature in group for name, group in clique_groups.items() 
                  if name != 'bridge' and any(b in group for b in clique_groups['bridge'])):
            return 0.0
            
        # Sample data points to use as baselines
        n_samples = min(50, len(test_data))
        sample_indices = np.random.choice(len(test_data), n_samples, replace=False)
        
        # For each sample, measure effect of changing this feature across its range
        feature_effect = 0.0
        unique_values = sorted(test_data[feature].unique())
        
        if len(unique_values) <= 1:
            return 0.0
            
        for idx in sample_indices:
            # Create a baseline evidence dictionary from this sample
            evidence = {col: test_data.iloc[idx][col] for col in feature_cols}
            original_pred = model.predict(evidence)
            
            # Measure effect of each possible value of this feature
            predictions = []
            for val in unique_values:
                evidence_modified = evidence.copy()
                evidence_modified[feature] = val
                predictions.append(model.predict(evidence_modified))
            
            # Calculate variance of predictions as a measure of feature effect
            feature_effect += np.var(predictions)
        
        # Average effect across samples
        return feature_effect / n_samples
    
    # For each feature, combine permutation importance with direct effect analysis
    for feature in feature_cols:
        # Skip any feature with constant values
        if test_data[feature].nunique() <= 1:
            importance[feature] = 0.0
            continue

        # Determine which group(s) this feature belongs to
        feature_groups = [name for name, group in clique_groups.items() if feature in group]
        
        # Bridge variables and their direct connections get more permutations
        local_n_permutations = n_permutations * 3 if 'bridge' in feature_groups else n_permutations
            
        # 1. PERMUTATION IMPORTANCE
        feature_importance_scores = []
        
        for _ in range(local_n_permutations):
            # Make a copy of the test data
            permuted_data = test_data.copy()
            
            # Enhanced permutation approach: conditionally permute within strata
            # to preserve important distributional properties
            
            # First stratify by target variable
            for target_val in test_data[target_variable].unique():
                mask = test_data[target_variable] == target_val
                if sum(mask) > 1:  # Only permute if we have multiple samples
                    # For variables in known cliques, stratify by other clique variables
                    for group_name in feature_groups:
                        other_vars = [v for v in clique_groups[group_name] if v != feature]
                        if other_vars:
                            # Create strata based on combinations of other variables in the clique
                            for combo in test_data.loc[mask, other_vars].drop_duplicates().itertuples(index=False):
                                combo_mask = mask.copy()
                                for i, var in enumerate(other_vars):
                                    combo_mask &= (test_data[var] == combo[i])
                                
                                if sum(combo_mask) > 1:
                                    permuted_data.loc[combo_mask, feature] = np.random.permutation(
                                        permuted_data.loc[combo_mask, feature].values)
                        else:
                            # If no other variables in clique or not in clique, simple permutation
                            permuted_data.loc[mask, feature] = np.random.permutation(
                                permuted_data.loc[mask, feature].values)
            
            # Evaluate the model with the permuted feature
            permuted_results = evaluate_model(model, permuted_data, target_variable)
            permuted_accuracy = permuted_results['metrics']['accuracy']
            
            # Calculate importance as decrease in performance
            feature_importance = max(0, baseline_accuracy - permuted_accuracy)
            feature_importance_scores.append(feature_importance)
        
        # 2. DIRECT EFFECT ANALYSIS (for bridge variables and their connections)
        direct_effect = analyze_direct_effect(feature)
        
        # Combine both measures with weights
        # Permutation importance gets 70% weight, direct effect gets 30%
        permutation_weight = 0.7
        direct_effect_weight = 0.3
        
        # For some variables, permutation might not work well due to correlations
        # In these cases, rely more on direct effect
        if np.mean(feature_importance_scores) < 0.01 and direct_effect > 0:
            permutation_weight = 0.3
            direct_effect_weight = 0.7
        
        # Calculate mean importance from permutation
        mean_importance = np.mean(feature_importance_scores)
        
        # Combine both measures
        combined_importance = (mean_importance * permutation_weight + 
                              direct_effect * direct_effect_weight)
        
        # Store importance score
        importance[feature] = combined_importance
    
    # Add multipliers for known important variables based on domain knowledge
    importance_multipliers = {
        'RP': 1.1,  # Revenue potential is slightly more important
        'DC': 1.1,  # Development cost is slightly more important
        'R': 1.1,   # Return rate is slightly more important
        'G': 1.05,  # Growth rate has moderate importance
        'M': 1.05,  # Market size has moderate importance
    }
    
    for feature, multiplier in importance_multipliers.items():
        if feature in importance:
            importance[feature] *= multiplier
    
    # Normalize importance scores to [0, 1] range
    max_importance = max(max(importance.values()), 0.001)  # Avoid division by zero
    importance = {feature: imp / max_importance for feature, imp in importance.items()}
    
    # Apply smooth scaling to better differentiate mid-range values
    # This helps avoid many features having the same importance value
    for feature in importance:
        # Apply cubic root scaling for better differentiation of low values
        # This preserves order but spreads out smaller values
        if importance[feature] > 0:
            importance[feature] = importance[feature] ** (1/3)
    
    # Re-normalize after scaling
    max_importance = max(max(importance.values()), 0.001)
    importance = {feature: imp / max_importance for feature, imp in importance.items()}
    
    return importance


def evaluate_variable_interactions(model: FactorGraph, test_data: pd.DataFrame, 
                                  target_variable: str = 'Y') -> Dict[str, float]:
    """
    Evaluate interactions between variables to understand how they influence each other.
    
    Args:
        model: FactorGraph model
        test_data: Test dataset containing features and target variable
        target_variable: Name of the target variable
        
    Returns:
        Dictionary mapping variable pairs to interaction strength
    """
    # Get all features
    feature_cols = [col for col in test_data.columns if col != target_variable]
    
    # Initialize interaction dictionary
    interactions = {}
    
    # For each pair of variables, measure their joint effect
    for i, feat1 in enumerate(feature_cols):
        for feat2 in enumerate(feature_cols[i+1:], i+1):
            pair_name = f"{feat1}_{feat2}"
            
            # Skip if either variable has only one unique value
            if test_data[feat1].nunique() <= 1 or test_data[feat2].nunique() <= 1:
                interactions[pair_name] = 0.0
                continue
            
            # Sample data points
            n_samples = min(50, len(test_data))
            sample_indices = np.random.choice(len(test_data), n_samples, replace=False)
            
            interaction_effect = 0.0
            for idx in sample_indices:
                # Create baseline evidence
                evidence = {col: test_data.iloc[idx][col] for col in feature_cols}
                
                # Original prediction
                original_pred = model.predict(evidence)
                
                # Get unique values for both features
                vals1 = sorted(test_data[feat1].unique())
                vals2 = sorted(test_data[feat2].unique())
                
                # Measure individual effects
                effect1 = []
                for v1 in vals1:
                    ev1 = evidence.copy()
                    ev1[feat1] = v1
                    effect1.append(model.predict(ev1) - original_pred)
                
                effect2 = []
                for v2 in vals2:
                    ev2 = evidence.copy()
                    ev2[feat2] = v2
                    effect2.append(model.predict(ev2) - original_pred)
                
                # Measure joint effects
                joint_effects = []
                for v1 in vals1:
                    for v2 in vals2:
                        ev_joint = evidence.copy()
                        ev_joint[feat1] = v1
                        ev_joint[feat2] = v2
                        joint_effect = model.predict(ev_joint) - original_pred
                        
                        # Expected combined effect if no interaction
                        v1_idx = vals1.index(v1)
                        v2_idx = vals2.index(v2)
                        expected_effect = effect1[v1_idx] + effect2[v2_idx]
                        
                        # Interaction is difference between actual and expected
                        interaction_effect += abs(joint_effect - expected_effect)
            
            # Average interaction effect
            interactions[pair_name] = interaction_effect / (n_samples * len(vals1) * len(vals2))
    
    # Normalize interaction strengths
    max_interaction = max(max(interactions.values()), 0.001)
    interactions = {pair: effect / max_interaction for pair, effect in interactions.items()}
    
    return interactions


def generate_investment_decision_report(model: FactorGraph, test_case: Dict[str, int]) -> Dict[str, Any]:
    """
    Generate a comprehensive investment decision report for a specific test case.
    
    Args:
        model: FactorGraph model
        test_case: Dictionary with variable values to analyze
        
    Returns:
        Dictionary with decision report components
    """
    from .crf import InvestmentCRF
    
    report = {}
    
    # Check if it's an InvestmentCRF model
    if not isinstance(model, InvestmentCRF):
        return {"error": "Model is not an InvestmentCRF instance"}
    
    # Get prediction and probability
    probability = model.predict(test_case)
    prediction = 1 if probability > 0.5 else 0
    
    report["prediction"] = prediction
    report["probability"] = probability
    
    # Get primary factors
    bridge_vars = ["RP", "DC", "R"]
    bridge_impacts = {}
    
    # Analyze each bridge variable impact
    for var in bridge_vars:
        if var in test_case:
            # Create modified evidence with each possible value
            var_impact = []
            for val in range(3):  # Assuming cardinality of 3
                modified_evidence = test_case.copy()
                modified_evidence[var] = val
                modified_prob = model.predict(modified_evidence)
                var_impact.append(modified_prob)
            
            # Calculate impact as range of predictions
            impact_range = max(var_impact) - min(var_impact)
            bridge_impacts[var] = {
                "impact": impact_range,
                "current_value": test_case.get(var, "unknown"),
                "prediction_range": var_impact
            }
    
    report["bridge_impacts"] = bridge_impacts
    
    # Get secondary factors impact
    secondary_vars = ["G", "M", "B", "DT", "FC", "CAC", "ATA", "C"]
    secondary_impacts = {}
    
    for var in secondary_vars:
        if var in test_case:
            # Calculate impact by changing this variable
            base_value = test_case[var]
            modified_evidence = test_case.copy()
            
            # Try other possible values
            other_values = [v for v in range(3) if v != base_value]  # Assuming cardinality of 3
            impact_values = []
            
            for val in other_values:
                modified_evidence[var] = val
                impact_values.append(model.predict(modified_evidence))
            
            # Calculate max impact
            base_prediction = model.predict(test_case)
            max_impact = max(abs(pred - base_prediction) for pred in impact_values)
            
            secondary_impacts[var] = {
                "impact": max_impact,
                "current_value": base_value
            }
    
    report["secondary_impacts"] = secondary_impacts
    
    # Calculate risk and opportunity metrics
    report["risk_assessment"] = {
        "confidence": abs(probability - 0.5) * 2,  # 0 to 1 scale
        "decision_stability": min(
            min(abs(probability - bi["prediction_range"][0]), 
                abs(probability - bi["prediction_range"][2])) 
            for var, bi in bridge_impacts.items()
        )
    }
    
    # Generate investment recommendations
    if prediction == 1:  # Worthy
        # Find which variables could be improved further
        improvement_targets = []
        
        for var in bridge_vars:
            if var in test_case:
                if var == "RP" and test_case[var] < 2:
                    improvement_targets.append({"variable": var, "direction": "increase"})
                elif var == "DC" and test_case[var] > 0:
                    improvement_targets.append({"variable": var, "direction": "decrease"})
                elif var == "R" and test_case[var] < 2:
                    improvement_targets.append({"variable": var, "direction": "increase"})
        
        report["recommendations"] = {
            "decision": "Invest",
            "confidence_level": "High" if probability > 0.8 else "Medium" if probability > 0.65 else "Low",
            "improvement_targets": improvement_targets
        }
    else:  # Not worthy
        # Find what needs to change to make it worthy
        required_changes = []
        
        for var in bridge_vars:
            if var in test_case:
                # Check if changing this variable alone could flip the decision
                modified_evidence = test_case.copy()
                
                if var == "RP":
                    modified_evidence[var] = 2  # Set to highest value
                elif var == "DC":
                    modified_evidence[var] = 0  # Set to lowest value
                elif var == "R":
                    modified_evidence[var] = 2  # Set to highest value
                
                modified_prob = model.predict(modified_evidence)
                
                if modified_prob > 0.5 and probability <= 0.5:
                    required_changes.append({
                        "variable": var, 
                        "current_value": test_case[var],
                        "target_value": modified_evidence[var],
                        "impact": modified_prob - probability
                    })
        
        report["recommendations"] = {
            "decision": "Do Not Invest",
            "confidence_level": "High" if probability < 0.2 else "Medium" if probability < 0.35 else "Low",
            "required_changes": required_changes
        }
    
    return report


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str] = None, title: str = 'Confusion Matrix',
                          figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
    """
    Plot confusion matrix with percentages.
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        title: Plot title
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    if class_names is None:
        if len(cm) == 2:
            class_names = ['Not Worthy (0)', 'Worthy (1)']
        else:
            class_names = [str(i) for i in range(len(cm))]
    
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    
    # Calculate and display percentages
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j + 0.5, i + 0.7, f'({cm_norm[i, j]:.2%})', 
                    ha='center', va='center', color='black' if cm[i, j] < cm.max()/2 else 'white')
    
    return plt.gcf()


def plot_roc_curve(y_true: np.ndarray, y_score: np.ndarray, figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
    """
    Plot ROC curve.
    
    Args:
        y_true: True binary labels
        y_score: Target scores (probability estimates of the positive class)
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    
    return plt.gcf()


def plot_precision_recall_curve(y_true: np.ndarray, y_score: np.ndarray, 
                                figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
    """
    Plot precision-recall curve.
    
    Args:
        y_true: True binary labels
        y_score: Target scores (probability estimates of the positive class)
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    
    plt.figure(figsize=figsize)
    plt.plot(recall, precision, color='blue', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    
    return plt.gcf()


def plot_feature_importance(importance: Dict[str, float], top_n: int = 10, 
                           figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    Plot feature importance.
    
    Args:
        importance: Dictionary mapping feature names to importance scores
        top_n: Number of top features to display
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    # Sort features by importance
    sorted_features = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)
    
    # Take top_n features
    top_features = sorted_features[:top_n]
    feature_names = [f[0] for f in top_features]
    importance_values = [f[1] for f in top_features]
    
    # Create barplot
    plt.figure(figsize=figsize)
    bars = plt.barh(range(len(top_features)), importance_values, align='center')
    
    # Color bars based on positive/negative importance
    for i, value in enumerate(importance_values):
        bars[i].set_color('darkred' if value < 0 else 'darkblue')
    
    plt.yticks(range(len(top_features)), feature_names)
    plt.xlabel('Feature Importance')
    plt.title('Top Feature Importance')
    
    return plt.gcf()


def cross_validation(model_factory, data: pd.DataFrame, n_folds: int = 5, 
                    target_variable: str = 'Y', random_seed: int = 42) -> Dict[str, Union[List, float]]:
    """
    Perform k-fold cross-validation.
    
    Args:
        model_factory: Function that creates and trains a new model instance
        data: Dataset containing features and target variable
        n_folds: Number of folds for cross-validation
        target_variable: Name of the target variable
        random_seed: Random seed for reproducibility
    
    Returns:
        Dictionary containing evaluation metrics across folds
    """
    from sklearn.model_selection import KFold
    
    np.random.seed(random_seed)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    
    # Initialize metrics storage
    metrics_by_fold = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(data)):
        logger.info(f"Training fold {fold+1}/{n_folds}")
        
        # Split data
        train_data = data.iloc[train_idx]
        val_data = data.iloc[val_idx]
        
        try:
            # Train model
            model = model_factory(train_data)
            
            # Evaluate model
            fold_metrics = evaluate_model(model, val_data, target_variable)
            
            # Store metrics
            metrics_by_fold.append(fold_metrics['metrics'])
            
            logger.info(f"Fold {fold+1} accuracy: {fold_metrics['metrics']['accuracy']:.4f}")
            
        except Exception as e:
            logger.error(f"Error in fold {fold+1}: {str(e)}")
    
    # Compute average metrics across folds
    avg_metrics = {}
    if metrics_by_fold:
        # Get all metric keys from the first fold
        metric_keys = metrics_by_fold[0].keys()
        
        # Compute average for each metric
        for key in metric_keys:
            avg_metrics[key] = sum(fold[key] for fold in metrics_by_fold if key in fold) / len(metrics_by_fold)
    
    return {
        'fold_metrics': metrics_by_fold,
        'avg_metrics': avg_metrics
    }


def evaluate_effect_of_evidence(model: FactorGraph, evidence_variables: List[str], 
                               test_case: Dict[str, int], target_variable: str = 'Y') -> Dict[str, Any]:
    """
    Evaluate the effect of different evidence variables on prediction.
    
    Args:
        model: FactorGraph model
        evidence_variables: List of evidence variables to analyze
        test_case: Dictionary representing a single test case with values for all variables
        target_variable: Name of the target variable
    
    Returns:
        Dictionary containing results of the analysis
    """
    results = {}
    baseline_evidence = {}
    
    # Run prediction with no evidence (only essential variables if needed)
    baseline_prediction = predict_investment_worthiness(model, baseline_evidence, method="bp")
    results['baseline'] = baseline_prediction
    
    # Evaluate effect of each variable individually
    individual_effects = {}
    for var in evidence_variables:
        if var in test_case:
            evidence = {var: test_case[var]}
            prediction = predict_investment_worthiness(model, evidence, method="bp")
            individual_effects[var] = prediction
    
    results['individual_effects'] = individual_effects
    
    # Evaluate cumulative effect by adding one variable at a time
    cumulative_effects = {}
    cumulative_evidence = {}
    
    for var in evidence_variables:
        if var in test_case:
            cumulative_evidence[var] = test_case[var]
            prediction = predict_investment_worthiness(model, cumulative_evidence, method="bp")
            cumulative_effects[var] = prediction
    
    results['cumulative_effects'] = cumulative_effects
    
    # Full evidence prediction
    full_evidence = {var: test_case[var] for var in evidence_variables if var in test_case}
    full_prediction = predict_investment_worthiness(model, full_evidence, method="bp")
    results['full_evidence'] = full_prediction
    
    return results


def evaluate_probability_calibration(model: FactorGraph, test_data: pd.DataFrame, 
                                   target_variable: str = 'Y', n_bins: int = 10) -> Dict[str, Any]:
    """
    Evaluate probability calibration of the model.
    
    Args:
        model: FactorGraph model
        test_data: Test dataset containing features and target variable
        target_variable: Name of the target variable
        n_bins: Number of bins for calibration assessment
    
    Returns:
        Dictionary containing calibration information
    """
    # Perform batch inference with belief propagation to get probabilities
    results = batch_inference(model, test_data, target_variable, method="bp")
    
    # Extract true values and predicted probabilities
    y_true = test_data[target_variable].values
    
    # Extract probabilities for positive class (assuming binary classification)
    prob_col = f"prob_{target_variable}_1"
    if prob_col not in results.columns:
        raise ValueError(f"Probability column {prob_col} not found in results")
    
    y_prob = results[prob_col].values
    
    # Create bins and calculate calibration
    bins = np.linspace(0, 1, n_bins + 1)
    binned_probs = np.digitize(y_prob, bins) - 1
    
    bin_counts = np.zeros(n_bins)
    bin_total_true = np.zeros(n_bins)
    bin_avg_pred_prob = np.zeros(n_bins)
    
    for bin_idx in range(n_bins):
        mask = binned_probs == bin_idx
        bin_counts[bin_idx] = np.sum(mask)
        if bin_counts[bin_idx] > 0:
            bin_total_true[bin_idx] = np.sum(y_true[mask])
            bin_avg_pred_prob[bin_idx] = np.mean(y_prob[mask])
    
    bin_actual_prob = np.divide(bin_total_true, bin_counts, out=np.zeros_like(bin_total_true), where=bin_counts > 0)
    
    # Calculate calibration error
    bins_with_samples = bin_counts > 0
    if np.any(bins_with_samples):
        calibration_error = np.mean(np.abs(bin_actual_prob[bins_with_samples] - bin_avg_pred_prob[bins_with_samples]))
    else:
        calibration_error = np.nan
    
    return {
        'bin_edges': bins,
        'bin_counts': bin_counts,
        'bin_pred_prob': bin_avg_pred_prob,
        'bin_actual_prob': bin_actual_prob,
        'calibration_error': calibration_error
    }


def plot_calibration_curve(calibration_results: Dict[str, Any], 
                          figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
    """
    Plot calibration curve.
    
    Args:
        calibration_results: Results from evaluate_probability_calibration
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    bin_pred_prob = calibration_results['bin_pred_prob']
    bin_actual_prob = calibration_results['bin_actual_prob']
    bin_counts = calibration_results['bin_counts']
    
    plt.figure(figsize=figsize)
    
    # Plot calibration curve
    plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
    
    # Plot only bins with samples
    valid_bins = bin_counts > 0
    plt.plot(bin_pred_prob[valid_bins], bin_actual_prob[valid_bins], 'o-', label='Model calibration')
    
    plt.xlabel('Predicted probability')
    plt.ylabel('Actual probability')
    plt.title(f'Calibration curve (error = {calibration_results["calibration_error"]:.4f})')
    plt.legend()
    plt.grid(True)
    
    # Add histogram of predicted probabilities
    ax2 = plt.gca().twinx()
    ax2.hist(bin_pred_prob[valid_bins], weights=bin_counts[valid_bins], bins=len(bin_counts), 
             alpha=0.3, color='gray', label='Samples')
    ax2.set_ylabel('Count')
    
    return plt.gcf()