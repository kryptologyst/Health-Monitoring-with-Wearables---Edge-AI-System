"""Edge-optimized health monitoring model for resource-constrained devices."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeHealthModel(nn.Module):
    """Edge-optimized health monitoring model.
    
    A lightweight model designed for deployment on resource-constrained
    edge devices with minimal memory and computational requirements.
    """
    
    def __init__(
        self,
        input_size: int = 4,
        hidden_size: int = 16,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = False,
    ) -> None:
        """Initialize the edge model.
        
        Args:
            input_size: Number of input features.
            hidden_size: Hidden layer size (kept small for edge deployment).
            dropout_rate: Dropout rate for regularization.
            use_batch_norm: Whether to use batch normalization.
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        
        # Minimal network architecture
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size // 2)
        self.output_layer = nn.Linear(hidden_size // 2, 1)
        
        # Optional batch normalization
        if use_batch_norm:
            self.bn1 = nn.BatchNorm1d(hidden_size)
            self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        """Initialize model weights using Xavier initialization."""
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
        if self.use_batch_norm:
            x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Hidden layer
        x = self.hidden_layer(x)
        if self.use_batch_norm:
            x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Output layer
        x = self.output_layer(x)
        
        return x
    
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
    
    def get_model_size(self) -> Dict[str, int]:
        """Get model size information.
        
        Returns:
            Dictionary with model size metrics.
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Estimate model size in bytes (assuming float32)
        model_size_bytes = total_params * 4
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_bytes": model_size_bytes,
            "model_size_kb": model_size_bytes / 1024,
            "model_size_mb": model_size_bytes / (1024 * 1024),
        }


class TinyHealthModel(nn.Module):
    """Ultra-lightweight model for microcontrollers.
    
    A minimal model with only essential parameters for deployment
    on microcontrollers with very limited resources.
    """
    
    def __init__(self, input_size: int = 4) -> None:
        """Initialize the tiny model.
        
        Args:
            input_size: Number of input features.
        """
        super().__init__()
        
        self.input_size = input_size
        
        # Single hidden layer with minimal parameters
        self.hidden = nn.Linear(input_size, 8)
        self.output = nn.Linear(8, 1)
        
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
        x = F.relu(self.hidden(x))
        x = self.output(x)
        return x
    
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


class EdgeModelOptimizer:
    """Optimizer for edge model deployment."""
    
    def __init__(self, model: nn.Module) -> None:
        """Initialize the optimizer.
        
        Args:
            model: PyTorch model to optimize.
        """
        self.model = model
    
    def prune_model(
        self, 
        sparsity: float = 0.5,
        method: str = "magnitude"
    ) -> nn.Module:
        """Prune the model to reduce size.
        
        Args:
            sparsity: Target sparsity (0-1).
            method: Pruning method ("magnitude", "random").
            
        Returns:
            Pruned model.
        """
        # Simple magnitude-based pruning
        if method == "magnitude":
            self._magnitude_pruning(sparsity)
        elif method == "random":
            self._random_pruning(sparsity)
        else:
            raise ValueError(f"Unknown pruning method: {method}")
        
        return self.model
    
    def _magnitude_pruning(self, sparsity: float) -> None:
        """Apply magnitude-based pruning.
        
        Args:
            sparsity: Target sparsity (0-1).
        """
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                # Calculate threshold
                weights = module.weight.data
                threshold = torch.quantile(torch.abs(weights), sparsity)
                
                # Prune weights below threshold
                mask = torch.abs(weights) > threshold
                module.weight.data *= mask.float()
    
    def _random_pruning(self, sparsity: float) -> None:
        """Apply random pruning.
        
        Args:
            sparsity: Target sparsity (0-1).
        """
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                # Create random mask
                mask = torch.rand_like(module.weight.data) > sparsity
                module.weight.data *= mask.float()
    
    def quantize_model(
        self, 
        precision: str = "int8"
    ) -> nn.Module:
        """Quantize the model for edge deployment.
        
        Args:
            precision: Quantization precision ("int8", "int16").
            
        Returns:
            Quantized model.
        """
        if precision == "int8":
            self._quantize_int8()
        elif precision == "int16":
            self._quantize_int16()
        else:
            raise ValueError(f"Unknown precision: {precision}")
        
        return self.model
    
    def _quantize_int8(self) -> None:
        """Quantize model to int8."""
        for name, module in self.modules():
            if isinstance(module, nn.Linear):
                # Simple quantization to int8
                weights = module.weight.data
                scale = 127.0 / torch.max(torch.abs(weights))
                quantized_weights = torch.round(weights * scale).clamp(-128, 127)
                module.weight.data = quantized_weights / scale
    
    def _quantize_int16(self) -> None:
        """Quantize model to int16."""
        for name, module in self.modules():
            if isinstance(module, nn.Linear):
                # Simple quantization to int16
                weights = module.weight.data
                scale = 32767.0 / torch.max(torch.abs(weights))
                quantized_weights = torch.round(weights * scale).clamp(-32768, 32767)
                module.weight.data = quantized_weights / scale
    
    def get_optimization_report(self) -> Dict[str, any]:
        """Get optimization report.
        
        Returns:
            Dictionary with optimization metrics.
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        zero_params = sum((p == 0).sum().item() for p in self.model.parameters())
        sparsity = zero_params / total_params if total_params > 0 else 0
        
        # Estimate memory usage
        memory_usage = sum(p.numel() * p.element_size() for p in self.model.parameters())
        
        return {
            "total_parameters": total_params,
            "zero_parameters": zero_params,
            "sparsity": sparsity,
            "memory_usage_bytes": memory_usage,
            "memory_usage_kb": memory_usage / 1024,
            "memory_usage_mb": memory_usage / (1024 * 1024),
        }
