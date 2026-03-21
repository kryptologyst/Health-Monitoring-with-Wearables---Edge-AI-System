"""Model factory for creating different types of health monitoring models."""

from typing import Dict, List, Optional, Type, Union

import torch
import torch.nn as nn

from .baseline_model import BaselineHealthModel, HealthModelTrainer
from .edge_model import EdgeHealthModel, TinyHealthModel, EdgeModelOptimizer
from .quantized_model import QuantizedHealthModel, QuantizationAwareTrainer


class ModelFactory:
    """Factory class for creating health monitoring models."""
    
    # Model registry
    MODEL_REGISTRY = {
        "baseline": BaselineHealthModel,
        "edge": EdgeHealthModel,
        "tiny": TinyHealthModel,
        "quantized": QuantizedHealthModel,
    }
    
    @classmethod
    def create_model(
        cls,
        model_type: str,
        input_size: int = 4,
        **kwargs
    ) -> nn.Module:
        """Create a health monitoring model.
        
        Args:
            model_type: Type of model to create.
            input_size: Number of input features.
            **kwargs: Additional model parameters.
            
        Returns:
            PyTorch model instance.
        """
        if model_type not in cls.MODEL_REGISTRY:
            raise ValueError(f"Unknown model type: {model_type}. "
                           f"Available types: {list(cls.MODEL_REGISTRY.keys())}")
        
        model_class = cls.MODEL_REGISTRY[model_type]
        return model_class(input_size=input_size, **kwargs)
    
    @classmethod
    def create_trainer(
        cls,
        model: nn.Module,
        trainer_type: str = "standard",
        device: str = "cpu",
        **kwargs
    ) -> Union[HealthModelTrainer, QuantizationAwareTrainer]:
        """Create a model trainer.
        
        Args:
            model: Model to train.
            trainer_type: Type of trainer ("standard", "qat").
            device: Device to use for training.
            **kwargs: Additional trainer parameters.
            
        Returns:
            Trainer instance.
        """
        if trainer_type == "standard":
            return HealthModelTrainer(model, device=device, **kwargs)
        elif trainer_type == "qat":
            return QuantizationAwareTrainer(model, device=device, **kwargs)
        else:
            raise ValueError(f"Unknown trainer type: {trainer_type}")
    
    @classmethod
    def create_optimizer(
        cls,
        model: nn.Module,
        optimizer_type: str = "edge"
    ) -> EdgeModelOptimizer:
        """Create a model optimizer.
        
        Args:
            model: Model to optimize.
            optimizer_type: Type of optimizer.
            
        Returns:
            Optimizer instance.
        """
        if optimizer_type == "edge":
            return EdgeModelOptimizer(model)
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    @classmethod
    def get_model_info(cls, model_type: str) -> Dict[str, any]:
        """Get information about a model type.
        
        Args:
            model_type: Type of model.
            
        Returns:
            Dictionary with model information.
        """
        if model_type not in cls.MODEL_REGISTRY:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model_class = cls.MODEL_REGISTRY[model_type]
        
        # Create a dummy model to get info
        dummy_model = model_class(input_size=4)
        
        info = {
            "model_type": model_type,
            "model_class": model_class.__name__,
            "description": model_class.__doc__ or "No description available",
            "default_parameters": cls._get_default_parameters(model_class),
        }
        
        # Get model size info if available
        if hasattr(dummy_model, 'get_model_size'):
            info.update(dummy_model.get_model_size())
        
        return info
    
    @classmethod
    def _get_default_parameters(cls, model_class: Type[nn.Module]) -> Dict[str, any]:
        """Get default parameters for a model class.
        
        Args:
            model_class: Model class.
            
        Returns:
            Dictionary with default parameters.
        """
        import inspect
        
        # Get constructor signature
        sig = inspect.signature(model_class.__init__)
        
        defaults = {}
        for name, param in sig.parameters.items():
            if name != 'self' and param.default != inspect.Parameter.empty:
                defaults[name] = param.default
        
        return defaults
    
    @classmethod
    def list_available_models(cls) -> List[str]:
        """List all available model types.
        
        Returns:
            List of available model type names.
        """
        return list(cls.MODEL_REGISTRY.keys())
    
    @classmethod
    def register_model(cls, name: str, model_class: Type[nn.Module]) -> None:
        """Register a new model type.
        
        Args:
            name: Model type name.
            model_class: Model class.
        """
        cls.MODEL_REGISTRY[name] = model_class
    
    @classmethod
    def create_model_comparison(
        cls,
        input_size: int = 4,
        test_data: Optional[torch.Tensor] = None
    ) -> Dict[str, Dict[str, any]]:
        """Create a comparison of all available models.
        
        Args:
            input_size: Number of input features.
            test_data: Test data for model analysis.
            
        Returns:
            Dictionary with model comparison results.
        """
        comparison = {}
        
        for model_type in cls.MODEL_REGISTRY.keys():
            try:
                # Create model
                model = cls.create_model(model_type, input_size)
                
                # Get basic info
                info = cls.get_model_info(model_type)
                
                # Add model size info
                if hasattr(model, 'get_model_size'):
                    info.update(model.get_model_size())
                
                # Add inference test if test data provided
                if test_data is not None:
                    info.update(cls._test_model_inference(model, test_data))
                
                comparison[model_type] = info
                
            except Exception as e:
                comparison[model_type] = {
                    "error": str(e),
                    "model_type": model_type,
                }
        
        return comparison
    
    @classmethod
    def _test_model_inference(
        cls,
        model: nn.Module,
        test_data: torch.Tensor
    ) -> Dict[str, any]:
        """Test model inference performance.
        
        Args:
            model: Model to test.
            test_data: Test data.
            
        Returns:
            Dictionary with inference test results.
        """
        import time
        
        model.eval()
        
        # Warm up
        with torch.no_grad():
            _ = model(test_data)
        
        # Time inference
        start_time = time.time()
        with torch.no_grad():
            for _ in range(100):
                _ = model(test_data)
        inference_time = time.time() - start_time
        
        return {
            "avg_inference_time_ms": (inference_time / 100) * 1000,
            "inference_fps": 100 / inference_time,
        }
