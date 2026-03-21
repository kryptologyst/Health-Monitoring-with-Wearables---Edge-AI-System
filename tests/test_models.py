"""Tests for health monitoring models."""

import pytest
import torch
import numpy as np

from src.models import ModelFactory
from src.pipelines import SensorDataGenerator, DataProcessor
from src.utils import ModelEvaluator


class TestModelFactory:
    """Test cases for ModelFactory."""
    
    def test_create_baseline_model(self):
        """Test baseline model creation."""
        model = ModelFactory.create_model("baseline", input_size=4)
        assert isinstance(model, torch.nn.Module)
        assert model.input_size == 4
    
    def test_create_edge_model(self):
        """Test edge model creation."""
        model = ModelFactory.create_model("edge", input_size=4)
        assert isinstance(model, torch.nn.Module)
        assert model.input_size == 4
    
    def test_create_tiny_model(self):
        """Test tiny model creation."""
        model = ModelFactory.create_model("tiny", input_size=4)
        assert isinstance(model, torch.nn.Module)
        assert model.input_size == 4
    
    def test_create_quantized_model(self):
        """Test quantized model creation."""
        model = ModelFactory.create_model("quantized", input_size=4)
        assert isinstance(model, torch.nn.Module)
        assert model.input_size == 4
    
    def test_invalid_model_type(self):
        """Test invalid model type raises error."""
        with pytest.raises(ValueError):
            ModelFactory.create_model("invalid", input_size=4)
    
    def test_model_info(self):
        """Test model info retrieval."""
        info = ModelFactory.get_model_info("baseline")
        assert "model_type" in info
        assert "model_class" in info
        assert "description" in info
        assert "default_parameters" in info


class TestSensorDataGenerator:
    """Test cases for SensorDataGenerator."""
    
    def test_generate_reading(self):
        """Test single reading generation."""
        generator = SensorDataGenerator(seed=42)
        reading = generator.generate_reading("test_device", "normal")
        
        assert reading.device_id == "test_device"
        assert 40 <= reading.heart_rate <= 200
        assert 35.0 <= reading.skin_temperature <= 40.0
        assert 0.0 <= reading.activity_level <= 1.0
        assert 70.0 <= reading.blood_oxygen <= 100.0
    
    def test_generate_dataset(self):
        """Test dataset generation."""
        generator = SensorDataGenerator(seed=42)
        df = generator.generate_dataset(n_samples=100)
        
        assert len(df) == 100
        assert "heart_rate" in df.columns
        assert "skin_temperature" in df.columns
        assert "activity_level" in df.columns
        assert "blood_oxygen" in df.columns
        assert "device_id" in df.columns
    
    def test_health_status_variations(self):
        """Test different health status variations."""
        generator = SensorDataGenerator(seed=42)
        
        normal_reading = generator.generate_reading("device", "normal")
        stressed_reading = generator.generate_reading("device", "stressed")
        
        # Stressed should generally have higher HR
        assert stressed_reading.heart_rate > normal_reading.heart_rate


class TestDataProcessor:
    """Test cases for DataProcessor."""
    
    def test_process_dataframe(self):
        """Test DataFrame processing."""
        # Create test data
        generator = SensorDataGenerator(seed=42)
        df = generator.generate_dataset(n_samples=100)
        df['alert'] = (df['heart_rate'] > 90).astype(int)
        
        processor = DataProcessor()
        processed_df = processor.process_dataframe(df)
        
        # Check that scaled columns exist
        assert "heart_rate_scaled" in processed_df.columns
        assert "skin_temperature_scaled" in processed_df.columns
        assert "activity_level_scaled" in processed_df.columns
        assert "blood_oxygen_scaled" in processed_df.columns
    
    def test_outlier_removal(self):
        """Test outlier removal."""
        # Create data with outliers
        data = {
            'heart_rate': [75, 80, 85, 200, 90],  # 200 is an outlier
            'skin_temperature': [36.5, 36.6, 36.7, 36.8, 36.9],
            'activity_level': [0.3, 0.4, 0.5, 0.6, 0.7],
            'blood_oxygen': [98, 97, 96, 95, 94],
        }
        df = pd.DataFrame(data)
        
        processor = DataProcessor(remove_outliers=True, outlier_threshold=2.0)
        processed_df = processor.process_dataframe(df)
        
        # Should remove the outlier
        assert len(processed_df) < len(df)


class TestModelEvaluator:
    """Test cases for ModelEvaluator."""
    
    def test_evaluate_model(self):
        """Test model evaluation."""
        # Create test model and data
        model = ModelFactory.create_model("baseline", input_size=4)
        
        # Create dummy test data
        X = torch.randn(100, 4)
        y = torch.randint(0, 2, (100,)).float()
        dataset = torch.utils.data.TensorDataset(X, y)
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=32)
        
        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate_model(model, test_loader, "test_model")
        
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert "auc" in metrics
    
    def test_compare_models(self):
        """Test model comparison."""
        # Create test models
        models = {
            "baseline": ModelFactory.create_model("baseline", input_size=4),
            "edge": ModelFactory.create_model("edge", input_size=4),
        }
        
        # Create dummy test data
        X = torch.randn(100, 4)
        y = torch.randint(0, 2, (100,)).float()
        dataset = torch.utils.data.TensorDataset(X, y)
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=32)
        
        evaluator = ModelEvaluator()
        comparison_df = evaluator.compare_models(models, test_loader)
        
        assert len(comparison_df) == 2
        assert "model_name" in comparison_df.columns
        assert "accuracy" in comparison_df.columns
        assert "f1_score" in comparison_df.columns


class TestModelInference:
    """Test cases for model inference."""
    
    def test_model_forward_pass(self):
        """Test model forward pass."""
        model = ModelFactory.create_model("baseline", input_size=4)
        model.eval()
        
        # Test forward pass
        x = torch.randn(1, 4)
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (1, 1)
        assert not torch.isnan(output).any()
    
    def test_model_prediction(self):
        """Test model prediction methods."""
        model = ModelFactory.create_model("baseline", input_size=4)
        model.eval()
        
        x = torch.randn(1, 4)
        
        # Test probability prediction
        with torch.no_grad():
            proba = model.predict_proba(x)
        
        assert 0 <= proba.item() <= 1
        
        # Test binary prediction
        with torch.no_grad():
            pred = model.predict(x)
        
        assert pred.item() in [0, 1]


if __name__ == "__main__":
    pytest.main([__file__])
