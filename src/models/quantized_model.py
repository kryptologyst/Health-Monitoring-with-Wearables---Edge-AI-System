"""Quantized health monitoring model for edge deployment."""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.quantization import quantize_dynamic, quantize_static


class QuantizedHealthModel(nn.Module):
    """Quantized health monitoring model for edge deployment.
    
    This model uses quantization techniques to reduce memory usage
    and improve inference speed on edge devices.
    """
    
    def __init__(
        self,
        input_size: int = 4,
        hidden_size: int = 16,
        dropout_rate: float = 0.1,
        quantization_type: str = "dynamic",  # "dynamic", "static", "qat"
    ) -> None:
        """Initialize the quantized model.
        
        Args:
            input_size: Number of input features.
            hidden_size: Hidden layer size.
            dropout_rate: Dropout rate for regularization.
            quantization_type: Type of quantization to use.
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.quantization_type = quantization_type
        
        # Build the base model
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size // 2)
        self.output_layer = nn.Linear(hidden_size // 2, 1)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # Quantization settings
        self.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            x: Input tensor (batch_size, input_size).
            
        Returns:
            Output tensor (batch_size, 1).
        """
        # Input layer
        x = self.input_layer(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Hidden layer
        x = self.hidden_layer(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Output layer
        x = self.output_layer(x)
        
        return x
    
    def prepare_for_quantization(self) -> None:
        """Prepare the model for quantization."""
        if self.quantization_type == "static":
            # Set quantization configuration
            self.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            torch.quantization.prepare(self, inplace=True)
        elif self.quantization_type == "qat":
            # Set quantization-aware training configuration
            self.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
            torch.quantization.prepare_qat(self, inplace=True)
    
    def quantize_model(self) -> nn.Module:
        """Quantize the model.
        
        Returns:
            Quantized model.
        """
        if self.quantization_type == "dynamic":
            # Dynamic quantization
            quantized_model = quantize_dynamic(
                self, 
                {nn.Linear}, 
                dtype=torch.qint8
            )
            return quantized_model
            
        elif self.quantization_type == "static":
            # Static quantization (requires calibration)
            self.eval()
            quantized_model = quantize_static(
                self,
                {nn.Linear},
                self.qconfig,
                inplace=False
            )
            return quantized_model
            
        elif self.quantization_type == "qat":
            # Quantization-aware training
            self.eval()
            quantized_model = torch.quantization.convert(self, inplace=False)
            return quantized_model
            
        else:
            raise ValueError(f"Unknown quantization type: {self.quantization_type}")
    
    def calibrate_model(self, calibration_data: torch.Tensor) -> None:
        """Calibrate the model for static quantization.
        
        Args:
            calibration_data: Calibration dataset.
        """
        if self.quantization_type != "static":
            raise ValueError("Calibration only supported for static quantization")
        
        self.eval()
        with torch.no_grad():
            for data in calibration_data:
                self(data)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Predict class probabilities.
        
        Args:
            x: Input tensor (batch_size, input_size).
            
        Returns:
            Probability tensor (batch_size, 1).
        """
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)
    
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Predict binary classes.
        
        Args:
            x: Input tensor (batch_size, input_size).
            threshold: Classification threshold.
            
        Returns:
            Binary predictions (batch_size,).
        """
        with torch.no_grad():
            proba = self.predict_proba(x)
            return (proba > threshold).float()


class QuantizationAwareTrainer:
    """Trainer for quantization-aware training."""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
    ) -> None:
        """Initialize the QAT trainer.
        
        Args:
            model: Model to train.
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
    
    def train_epoch(self, dataloader: torch.utils.data.DataLoader) -> Tuple[float, float]:
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
    
    def validate_epoch(self, dataloader: torch.utils.data.DataLoader) -> Tuple[float, float]:
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
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        epochs: int = 30,
        patience: int = 5,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """Train the model with quantization-aware training.
        
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
            
            if verbose and epoch % 5 == 0:
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


class ModelQuantizationAnalyzer:
    """Analyzer for model quantization performance."""
    
    def __init__(self, original_model: nn.Module, quantized_model: nn.Module) -> None:
        """Initialize the analyzer.
        
        Args:
            original_model: Original floating-point model.
            quantized_model: Quantized model.
        """
        self.original_model = original_model
        self.quantized_model = quantized_model
    
    def compare_model_sizes(self) -> Dict[str, Union[int, float]]:
        """Compare model sizes.
        
        Returns:
            Dictionary with size comparison metrics.
        """
        # Original model size
        original_params = sum(p.numel() for p in self.original_model.parameters())
        original_size = sum(p.numel() * p.element_size() for p in self.original_model.parameters())
        
        # Quantized model size
        quantized_params = sum(p.numel() for p in self.quantized_model.parameters())
        quantized_size = sum(p.numel() * p.element_size() for p in self.quantized_model.parameters())
        
        # Calculate compression ratio
        compression_ratio = original_size / quantized_size if quantized_size > 0 else 0
        
        return {
            "original_parameters": original_params,
            "quantized_parameters": quantized_params,
            "original_size_bytes": original_size,
            "quantized_size_bytes": quantized_size,
            "original_size_kb": original_size / 1024,
            "quantized_size_kb": quantized_size / 1024,
            "original_size_mb": original_size / (1024 * 1024),
            "quantized_size_mb": quantized_size / (1024 * 1024),
            "compression_ratio": compression_ratio,
            "size_reduction_percent": (1 - quantized_size / original_size) * 100 if original_size > 0 else 0,
        }
    
    def compare_inference_speed(
        self, 
        test_data: torch.Tensor, 
        num_runs: int = 100
    ) -> Dict[str, float]:
        """Compare inference speed between models.
        
        Args:
            test_data: Test data for inference.
            num_runs: Number of inference runs for timing.
            
        Returns:
            Dictionary with timing comparison metrics.
        """
        import time
        
        # Warm up
        with torch.no_grad():
            _ = self.original_model(test_data)
            _ = self.quantized_model(test_data)
        
        # Time original model
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_runs):
                _ = self.original_model(test_data)
        original_time = time.time() - start_time
        
        # Time quantized model
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_runs):
                _ = self.quantized_model(test_data)
        quantized_time = time.time() - start_time
        
        # Calculate speedup
        speedup = original_time / quantized_time if quantized_time > 0 else 0
        
        return {
            "original_inference_time": original_time / num_runs,
            "quantized_inference_time": quantized_time / num_runs,
            "speedup": speedup,
            "time_reduction_percent": (1 - quantized_time / original_time) * 100 if original_time > 0 else 0,
        }
    
    def compare_accuracy(
        self, 
        test_data: torch.Tensor, 
        test_labels: torch.Tensor
    ) -> Dict[str, float]:
        """Compare accuracy between models.
        
        Args:
            test_data: Test data.
            test_labels: Test labels.
            
        Returns:
            Dictionary with accuracy comparison metrics.
        """
        # Original model predictions
        with torch.no_grad():
            original_logits = self.original_model(test_data)
            original_proba = torch.sigmoid(original_logits.squeeze())
            original_pred = (original_proba > 0.5).float()
        
        # Quantized model predictions
        with torch.no_grad():
            quantized_logits = self.quantized_model(test_data)
            quantized_proba = torch.sigmoid(quantized_logits.squeeze())
            quantized_pred = (quantized_proba > 0.5).float()
        
        # Calculate accuracies
        original_acc = (original_pred == test_labels).float().mean().item()
        quantized_acc = (quantized_pred == test_labels).float().mean().item()
        
        # Calculate accuracy drop
        accuracy_drop = original_acc - quantized_acc
        
        return {
            "original_accuracy": original_acc,
            "quantized_accuracy": quantized_acc,
            "accuracy_drop": accuracy_drop,
            "accuracy_drop_percent": accuracy_drop * 100,
        }
