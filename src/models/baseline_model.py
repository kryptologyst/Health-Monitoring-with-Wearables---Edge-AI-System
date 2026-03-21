"""Baseline health monitoring model using PyTorch."""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset


class HealthDataset(Dataset):
    """Dataset class for health monitoring data."""
    
    def __init__(
        self, 
        features: np.ndarray, 
        targets: Optional[np.ndarray] = None
    ) -> None:
        """Initialize the dataset.
        
        Args:
            features: Feature array (n_samples, n_features).
            targets: Target array (n_samples,) or None for inference.
        """
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets) if targets is not None else None
        
    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Get a sample from the dataset.
        
        Args:
            idx: Sample index.
            
        Returns:
            Tuple of (features, targets) where targets can be None.
        """
        if self.targets is not None:
            return self.features[idx], self.targets[idx]
        else:
            return self.features[idx], None


class BaselineHealthModel(nn.Module):
    """Baseline health monitoring model for alert detection.
    
    A multi-layer perceptron that takes physiological features as input
    and outputs a probability of health alert.
    """
    
    def __init__(
        self,
        input_size: int = 4,
        hidden_sizes: List[int] = [64, 32, 16],
        dropout_rate: float = 0.2,
        num_classes: int = 1,
    ) -> None:
        """Initialize the baseline model.
        
        Args:
            input_size: Number of input features.
            hidden_sizes: List of hidden layer sizes.
            dropout_rate: Dropout rate for regularization.
            num_classes: Number of output classes (1 for binary classification).
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        
        # Build the network
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.BatchNorm1d(hidden_size),
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            x: Input tensor (batch_size, input_size).
            
        Returns:
            Output tensor (batch_size, num_classes).
        """
        return self.network(x)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Predict class probabilities.
        
        Args:
            x: Input tensor (batch_size, input_size).
            
        Returns:
            Probability tensor (batch_size, num_classes).
        """
        with torch.no_grad():
            logits = self.forward(x)
            if self.num_classes == 1:
                return torch.sigmoid(logits)
            else:
                return F.softmax(logits, dim=1)
    
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Predict binary classes.
        
        Args:
            x: Input tensor (batch_size, input_size).
            threshold: Classification threshold.
            
        Returns:
            Binary predictions (batch_size,).
        """
        with torch.no_grad():
            if self.num_classes == 1:
                proba = self.predict_proba(x)
                return (proba > threshold).float()
            else:
                proba = self.predict_proba(x)
                return torch.argmax(proba, dim=1)


class HealthModelTrainer:
    """Trainer class for health monitoring models."""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
    ) -> None:
        """Initialize the trainer.
        
        Args:
            model: PyTorch model to train.
            device: Device to use for training.
            learning_rate: Learning rate for optimizer.
            weight_decay: Weight decay for regularization.
        """
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Initialize optimizer and loss function
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
    
    def train_epoch(
        self, 
        dataloader: DataLoader
    ) -> Tuple[float, float]:
        """Train for one epoch.
        
        Args:
            dataloader: Training data loader.
            
        Returns:
            Tuple of (average_loss, accuracy).
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output.squeeze(), target)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            pred = (torch.sigmoid(output.squeeze()) > 0.5).float()
            correct += (pred == target).sum().item()
            total += target.size(0)
        
        avg_loss = total_loss / len(dataloader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def validate_epoch(
        self, 
        dataloader: DataLoader
    ) -> Tuple[float, float]:
        """Validate for one epoch.
        
        Args:
            dataloader: Validation data loader.
            
        Returns:
            Tuple of (average_loss, accuracy).
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output.squeeze(), target)
                
                total_loss += loss.item()
                pred = (torch.sigmoid(output.squeeze()) > 0.5).float()
                correct += (pred == target).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(dataloader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
        patience: int = 10,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """Train the model.
        
        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            epochs: Number of training epochs.
            patience: Early stopping patience.
            verbose: Whether to print training progress.
            
        Returns:
            Dictionary with training history.
        """
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validation
            val_loss, val_acc = self.validate_epoch(val_loader)
            
            # Store history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch:3d}: "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                break
        
        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "train_accuracies": self.train_accuracies,
            "val_accuracies": self.val_accuracies,
        }
    
    def evaluate(
        self, 
        dataloader: DataLoader
    ) -> Dict[str, float]:
        """Evaluate the model.
        
        Args:
            dataloader: Test data loader.
            
        Returns:
            Dictionary with evaluation metrics.
        """
        self.model.eval()
        all_preds = []
        all_targets = []
        total_loss = 0.0
        
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output.squeeze(), target)
                total_loss += loss.item()
                
                pred = (torch.sigmoid(output.squeeze()) > 0.5).float()
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        # Calculate metrics
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        accuracy = (all_preds == all_targets).mean()
        precision = np.sum((all_preds == 1) & (all_targets == 1)) / np.sum(all_preds == 1) if np.sum(all_preds == 1) > 0 else 0
        recall = np.sum((all_preds == 1) & (all_targets == 1)) / np.sum(all_targets == 1) if np.sum(all_targets == 1) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "loss": total_loss / len(dataloader),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        }
