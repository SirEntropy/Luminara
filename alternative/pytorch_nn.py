"""
PyTorch Neural Network model for comparison with Luminara CRF model.

This module implements a neural network model using PyTorch to compare with 
the CRF approach for AI investment worthiness prediction.
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    roc_auc_score
)

import logging
logger = logging.getLogger(__name__)


class InvestmentDataset(Dataset):
    """PyTorch Dataset for Investment Worthiness data."""
    
    def __init__(self, features, labels=None):
        """
        Initialize dataset.
        
        Args:
            features: Feature matrix
            labels: Labels (can be None for inference-only datasets)
        """
        self.features = torch.tensor(features.values, dtype=torch.float32)
        self.feature_names = features.columns.tolist()
        
        if labels is not None:
            self.labels = torch.tensor(labels.values, dtype=torch.long)
        else:
            self.labels = None
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        if self.labels is not None:
            return self.features[idx], self.labels[idx]
        else:
            return self.features[idx]


class InvestmentNN(nn.Module):
    """Neural Network model for investment worthiness prediction."""
    
    def __init__(self, input_size, hidden_sizes=[32, 16], output_size=2, dropout_rate=0.2):
        """
        Initialize neural network architecture.
        
        Args:
            input_size: Number of input features
            hidden_sizes: List of hidden layer sizes
            output_size: Number of output classes
            dropout_rate: Dropout probability for regularization
        """
        super(InvestmentNN, self).__init__()
        
        # Build layers dynamically based on hidden_sizes
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        # Final output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        # Combine all layers
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass through the network."""
        return self.model(x)


class InvestmentNNModel:
    """Wrapper class for PyTorch NN model with training and evaluation methods."""
    
    def __init__(self, input_size, hidden_sizes=[32, 16], output_size=2, 
                 learning_rate=0.001, batch_size=32, epochs=100, device=None):
        """
        Initialize the model wrapper.
        
        Args:
            input_size: Number of input features
            hidden_sizes: List of hidden layer sizes
            output_size: Number of output classes
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            epochs: Number of training epochs
            device: Device to use (cpu or cuda)
        """
        self.name = "PyTorch Neural Network"
        self.hidden_sizes = hidden_sizes
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.feature_names = None
        self.training_time = None
        self.inference_time = None
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        logger.info(f"Using device: {self.device}")
        
        # Initialize model
        self.model = InvestmentNN(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            output_size=output_size
        ).to(self.device)
        
        # Initialize optimizer and loss function
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
        # Initialize best model weights for early stopping
        self.best_val_loss = float('inf')
        self.best_model_weights = None
    
    def train(self, X_train, y_train, X_val=None, y_val=None, patience=10):
        """
        Train the neural network.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            patience: Early stopping patience
        """
        # Save feature names
        self.feature_names = X_train.columns.tolist()
        
        # Create datasets and dataloaders
        train_dataset = InvestmentDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        if X_val is not None and y_val is not None:
            val_dataset = InvestmentDataset(X_val, y_val)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
            use_validation = True
        else:
            use_validation = False
        
        # Training loop
        start_time = time.time()
        logger.info(f"Starting training for {self.epochs} epochs...")
        
        epochs_without_improvement = 0
        train_losses = []
        val_losses = []
        
        for epoch in range(self.epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item() * inputs.size(0)
            
            train_loss /= len(train_loader.dataset)
            train_losses.append(train_loss)
            
            # Validation phase
            if use_validation:
                val_loss, val_accuracy = self._evaluate_loss(val_loader)
                val_losses.append(val_loss)
                
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_model_weights = self.model.state_dict().copy()
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                
                # Log progress
                if (epoch + 1) % 10 == 0 or epoch == 0:
                    logger.info(f"Epoch {epoch+1}/{self.epochs}, "
                               f"Train Loss: {train_loss:.4f}, "
                               f"Val Loss: {val_loss:.4f}, "
                               f"Val Accuracy: {val_accuracy:.4f}")
                
                # Early stopping
                if epochs_without_improvement >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                # Log progress without validation
                if (epoch + 1) % 10 == 0 or epoch == 0:
                    logger.info(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {train_loss:.4f}")
        
        # Load best model if validation was used
        if use_validation and self.best_model_weights is not None:
            self.model.load_state_dict(self.best_model_weights)
        
        self.training_time = time.time() - start_time
        logger.info(f"Training completed in {self.training_time:.2f} seconds")
        
        # Return losses for plotting
        return {
            'train_losses': train_losses,
            'val_losses': val_losses if use_validation else None
        }
    
    def _evaluate_loss(self, dataloader):
        """
        Evaluate model on a dataloader and return loss and accuracy.
        
        Args:
            dataloader: DataLoader containing evaluation data
            
        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                # Compute predictions
                _, preds = torch.max(outputs, 1)
                
                total_loss += loss.item() * inputs.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        avg_loss = total_loss / len(dataloader.dataset)
        accuracy = accuracy_score(all_targets, all_preds)
        
        return avg_loss, accuracy
    
    def predict(self, X):
        """
        Make predictions using the trained model.
        
        Args:
            X: Features to predict
            
        Returns:
            Predicted labels
        """
        # Create dataset and dataloader
        dataset = InvestmentDataset(X)
        dataloader = DataLoader(dataset, batch_size=self.batch_size)
        
        # Make predictions
        start_time = time.time()
        self.model.eval()
        all_preds = []
        
        with torch.no_grad():
            for inputs in dataloader:
                if isinstance(inputs, tuple):
                    inputs = inputs[0]  # If dataset returns a tuple
                inputs = inputs.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
        
        self.inference_time = time.time() - start_time
        
        return np.array(all_preds)
    
    def predict_proba(self, X):
        """
        Predict probability of classes.
        
        Args:
            X: Features to predict
            
        Returns:
            Predicted probabilities for each class
        """
        # Create dataset and dataloader
        dataset = InvestmentDataset(X)
        dataloader = DataLoader(dataset, batch_size=self.batch_size)
        
        # Make predictions
        self.model.eval()
        all_probs = []
        
        with torch.no_grad():
            for inputs in dataloader:
                if isinstance(inputs, tuple):
                    inputs = inputs[0]  # If dataset returns a tuple
                inputs = inputs.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                probs = torch.softmax(outputs, dim=1)
                
                all_probs.extend(probs.cpu().numpy())
        
        return np.array(all_probs)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary containing evaluation metrics
        """
        y_pred = self.predict(X_test)
        probas = None
        
        try:
            probas = self.predict_proba(X_test)[:, 1]  # Probabilities for class 1
        except:
            pass
        
        metrics = self._compute_metrics(y_test, y_pred, probas)
        metrics['training_time'] = self.training_time
        metrics['inference_time'] = self.inference_time
        metrics['avg_inference_time_per_sample'] = self.inference_time / len(X_test) if len(X_test) > 0 else 0
        
        return metrics
    
    def _compute_metrics(self, y_true, y_pred, probas=None):
        """
        Compute evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            probas: Probability predictions for positive class
            
        Returns:
            Dictionary with metrics
        """
        cm = confusion_matrix(y_true, y_pred)
        
        # Extract values from confusion matrix
        if len(np.unique(y_true)) == 2:  # Binary classification
            tn, fp, fn, tp = cm.ravel()
            
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, zero_division=0),
                'recall': recall_score(y_true, y_pred, zero_division=0),
                'f1_score': f1_score(y_true, y_pred, zero_division=0),
                'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
                'confusion_matrix': cm
            }
            
            # Add AUC if probabilities are provided
            if probas is not None:
                try:
                    metrics['auc'] = roc_auc_score(y_true, probas)
                except:
                    metrics['auc'] = float('nan')
        
        else:  # Multi-class classification
            report = classification_report(y_true, y_pred, output_dict=True)
            metrics = {
                'accuracy': report['accuracy'],
                'macro_precision': report['macro avg']['precision'],
                'macro_recall': report['macro avg']['recall'],
                'macro_f1': report['macro avg']['f1-score'],
                'confusion_matrix': cm
            }
        
        return metrics
    
    def plot_training_history(self, history, figsize=(10, 6)):
        """
        Plot training history.
        
        Args:
            history: Dictionary containing training history
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        plt.figure(figsize=figsize)
        
        plt.plot(history['train_losses'], label='Training Loss')
        if history['val_losses'] is not None:
            plt.plot(history['val_losses'], label='Validation Loss')
        
        plt.title('Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        return plt.gcf()


def load_datasets(base_dir, sets=None):
    """
    Load datasets from the processed data directory.
    
    Args:
        base_dir: Base directory containing data (should have 'data/processed' subdirectory)
        sets: List of sets to load ('train', 'val', 'test'). If None, load all.
        
    Returns:
        Dictionary mapping set names to DataFrames
    """
    if sets is None:
        sets = ['train', 'val', 'test']
    
    data_dir = os.path.join(base_dir, 'data', 'processed')
    datasets = {}
    
    for set_name in sets:
        file_path = os.path.join(data_dir, f"{set_name}.csv")
        if os.path.exists(file_path):
            datasets[set_name] = pd.read_csv(file_path)
            logger.info(f"Loaded {set_name} dataset with {len(datasets[set_name])} samples")
        else:
            logger.warning(f"Dataset file not found: {file_path}")
    
    return datasets


def visualize_results(model, results, X_test, y_test, output_dir=None):
    """
    Visualize model evaluation results.
    
    Args:
        model: Trained model
        results: Dictionary with evaluation results
        X_test: Test features
        y_test: Test labels
        output_dir: Directory to save plots (optional)
    """
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    
    # Confusion matrix
    cm = results['confusion_matrix']
    class_names = ['Not Worthy (0)', 'Worthy (1)']
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    if output_dir is not None:
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # ROC curve (for binary classification)
    if 'auc' in results:
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'AUC = {results["auc"]:.4f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        
        if output_dir is not None:
            plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
        plt.show()


def train_pytorch_nn(base_dir, output_dir=None):
    """
    Train and evaluate PyTorch neural network model.
    
    Args:
        base_dir: Base directory of the project
        output_dir: Directory to save results and plots
        
    Returns:
        Tuple of (trained model, evaluation results)
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    logger.info("Loading datasets...")
    datasets = load_datasets(base_dir)
    
    # Extract features and target
    X_train = datasets['train'].drop('Y', axis=1)
    y_train = datasets['train']['Y']
    
    X_val = datasets['val'].drop('Y', axis=1)
    y_val = datasets['val']['Y']
    
    X_test = datasets['test'].drop('Y', axis=1)
    y_test = datasets['test']['Y']
    
    logger.info(f"Training data shape: {X_train.shape}")
    logger.info(f"Validation data shape: {X_val.shape}")
    logger.info(f"Test data shape: {X_test.shape}")
    
    # Create neural network model
    input_size = X_train.shape[1]
    hidden_sizes = [64, 32]  # More complex architecture for better performance
    
    model = InvestmentNNModel(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        output_size=2,  # Binary classification
        learning_rate=0.001,
        batch_size=32,
        epochs=200  # More epochs with early stopping
    )
    
    # Train model
    logger.info("Training neural network model...")
    history = model.train(X_train, y_train, X_val, y_val, patience=20)
    
    # Evaluate model
    logger.info("Evaluating model on test data...")
    results = model.evaluate(X_test, y_test)
    
    # Print evaluation metrics
    logger.info(f"Test Accuracy: {results['accuracy']:.4f}")
    logger.info(f"Test Precision: {results['precision']:.4f}")
    logger.info(f"Test Recall: {results['recall']:.4f}")
    logger.info(f"Test F1 Score: {results['f1_score']:.4f}")
    if 'auc' in results:
        logger.info(f"Test AUC: {results['auc']:.4f}")
    
    # Visualize results
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot training history
        plt.figure(figsize=(10, 6))
        plt.plot(history['train_losses'], label='Training Loss')
        if history['val_losses'] is not None:
            plt.plot(history['val_losses'], label='Validation Loss')
        plt.title('Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(output_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Visualize evaluation results
        visualize_results(model, results, X_test, y_test, output_dir)
        
        # Save model
        torch.save(model.model.state_dict(), os.path.join(output_dir, 'pytorch_nn_model.pt'))
        logger.info(f"Saved model to {os.path.join(output_dir, 'pytorch_nn_model.pt')}")
        
        # Save metrics
        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC (if applicable)', 
                      'Training Time (s)', 'Avg Inference Time (ms/sample)'],
            'Value': [
                results['accuracy'],
                results['precision'],
                results['recall'],
                results['f1_score'],
                results.get('auc', float('nan')),
                results['training_time'],
                results['avg_inference_time_per_sample'] * 1000  # Convert to ms
            ]
        })
        metrics_df.to_csv(os.path.join(output_dir, 'nn_metrics.csv'), index=False)
        logger.info(f"Saved metrics to {os.path.join(output_dir, 'nn_metrics.csv')}")
    
    return model, results


if __name__ == "__main__":
    # Get base directory (project root)
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # Set output directory for results and plots
    output_dir = os.path.join(base_dir, 'results', 'pytorch_nn')
    os.makedirs(output_dir, exist_ok=True)
    
    # Train and evaluate PyTorch neural network model
    model, results = train_pytorch_nn(base_dir, output_dir)
    
    # Print summary of results
    print("\nPyTorch Neural Network Model Results:")
    print("=" * 80)
    print(f"Accuracy:      {results['accuracy']:.4f}")
    print(f"Precision:     {results['precision']:.4f}")
    print(f"Recall:        {results['recall']:.4f}")
    print(f"F1 Score:      {results['f1_score']:.4f}")
    if 'auc' in results:
        print(f"AUC:           {results['auc']:.4f}")
    print(f"Training Time: {results['training_time']:.2f} seconds")
    print(f"Inference Time: {results['avg_inference_time_per_sample'] * 1000:.2f} ms/sample")
    print("=" * 80)
