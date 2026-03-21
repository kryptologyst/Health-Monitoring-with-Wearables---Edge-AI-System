"""Tests for data pipelines."""

import pytest
import pandas as pd
import numpy as np

from src.pipelines import SensorDataGenerator, DataProcessor, FeatureExtractor, StreamingPipeline


class TestSensorDataGenerator:
    """Test cases for SensorDataGenerator."""
    
    def test_generator_initialization(self):
        """Test generator initialization."""
        generator = SensorDataGenerator(sample_rate=1.0, noise_level=0.1, seed=42)
        assert generator.sample_rate == 1.0
        assert generator.noise_level == 0.1
        assert generator.rng is not None
    
    def test_generate_reading_normal(self):
        """Test normal health status reading generation."""
        generator = SensorDataGenerator(seed=42)
        reading = generator.generate_reading("test_device", "normal")
        
        # Check reading attributes
        assert reading.device_id == "test_device"
        assert isinstance(reading.timestamp, float)
        assert isinstance(reading.heart_rate, float)
        assert isinstance(reading.skin_temperature, float)
        assert isinstance(reading.activity_level, float)
        assert isinstance(reading.blood_oxygen, float)
        
        # Check value ranges
        assert 40 <= reading.heart_rate <= 200
        assert 35.0 <= reading.skin_temperature <= 40.0
        assert 0.0 <= reading.activity_level <= 1.0
        assert 70.0 <= reading.blood_oxygen <= 100.0
    
    def test_generate_reading_stressed(self):
        """Test stressed health status reading generation."""
        generator = SensorDataGenerator(seed=42)
        reading = generator.generate_reading("test_device", "stressed")
        
        # Stressed readings should have elevated values
        assert reading.heart_rate > 75  # Higher than baseline
        assert reading.skin_temperature > 36.5  # Higher than baseline
    
    def test_generate_dataset(self):
        """Test dataset generation."""
        generator = SensorDataGenerator(seed=42)
        df = generator.generate_dataset(n_samples=50)
        
        # Check DataFrame structure
        assert len(df) == 50
        expected_columns = [
            "timestamp", "heart_rate", "skin_temperature", 
            "activity_level", "blood_oxygen", "device_id",
            "battery_level", "signal_quality"
        ]
        for col in expected_columns:
            assert col in df.columns
    
    def test_health_status_distribution(self):
        """Test health status distribution in dataset."""
        generator = SensorDataGenerator(seed=42)
        health_distribution = {
            "normal": 0.5,
            "stressed": 0.3,
            "fatigued": 0.2,
        }
        
        df = generator.generate_dataset(
            n_samples=1000, 
            health_distribution=health_distribution
        )
        
        # Check that we have data from different health statuses
        assert len(df) == 1000


class TestDataProcessor:
    """Test cases for DataProcessor."""
    
    def test_processor_initialization(self):
        """Test processor initialization."""
        processor = DataProcessor(
            scaler_type="robust",
            remove_outliers=True,
            outlier_threshold=3.0
        )
        assert processor.scaler_type == "robust"
        assert processor.remove_outliers is True
        assert processor.outlier_threshold == 3.0
        assert not processor.is_fitted
    
    def test_process_dataframe(self):
        """Test DataFrame processing."""
        # Create test data
        generator = SensorDataGenerator(seed=42)
        df = generator.generate_dataset(n_samples=100)
        df['alert'] = (df['heart_rate'] > 90).astype(int)
        
        processor = DataProcessor()
        processed_df = processor.process_dataframe(df, fit_scaler=True)
        
        # Check that processor is fitted
        assert processor.is_fitted
        
        # Check that scaled columns exist
        scaled_columns = [
            "heart_rate_scaled", "skin_temperature_scaled",
            "activity_level_scaled", "blood_oxygen_scaled"
        ]
        for col in scaled_columns:
            assert col in processed_df.columns
    
    def test_outlier_removal(self):
        """Test outlier removal functionality."""
        # Create data with clear outliers
        data = {
            'heart_rate': [75, 80, 85, 200, 90, 95],  # 200 is an outlier
            'skin_temperature': [36.5, 36.6, 36.7, 36.8, 36.9, 37.0],
            'activity_level': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            'blood_oxygen': [98, 97, 96, 95, 94, 93],
        }
        df = pd.DataFrame(data)
        
        processor = DataProcessor(remove_outliers=True, outlier_threshold=2.0)
        processed_df = processor.process_dataframe(df)
        
        # Should remove the outlier (200 BPM)
        assert len(processed_df) < len(df)
        assert processed_df['heart_rate'].max() < 200
    
    def test_process_single_reading(self):
        """Test processing single sensor reading."""
        # Create test data
        generator = SensorDataGenerator(seed=42)
        df = generator.generate_dataset(n_samples=100)
        
        processor = DataProcessor()
        processor.process_dataframe(df, fit_scaler=True)
        
        # Create a single reading
        reading = generator.generate_reading("test_device", "normal")
        
        # Process single reading
        processed_features = processor.process_single_reading(reading)
        
        assert processed_features.shape == (1, 4)
        assert not np.isnan(processed_features).any()


class TestFeatureExtractor:
    """Test cases for FeatureExtractor."""
    
    def test_extractor_initialization(self):
        """Test extractor initialization."""
        extractor = FeatureExtractor(
            window_size=60,
            overlap=0.5,
            sample_rate=1.0
        )
        assert extractor.window_size == 60
        assert extractor.overlap == 0.5
        assert extractor.sample_rate == 1.0
        assert extractor.window_samples == 60
        assert extractor.step_size == 30
    
    def test_extract_features(self):
        """Test feature extraction."""
        # Create test data
        generator = SensorDataGenerator(seed=42)
        df = generator.generate_dataset(n_samples=200)
        df['alert'] = (df['heart_rate'] > 90).astype(int)
        
        extractor = FeatureExtractor(window_size=30, overlap=0.5)
        features, targets = extractor.extract_features(df, target_col='alert')
        
        assert features.shape[0] > 0
        assert features.shape[1] > 0  # Should have extracted features
        assert targets is not None
        assert len(features) == len(targets)
    
    def test_window_feature_extraction(self):
        """Test window-based feature extraction."""
        extractor = FeatureExtractor(window_size=10, overlap=0.5)
        
        # Create dummy window data
        window = np.random.randn(10, 4)
        
        features = extractor._extract_window_features(window)
        
        assert isinstance(features, np.ndarray)
        assert len(features) > 0
    
    def test_statistical_features(self):
        """Test statistical feature extraction."""
        extractor = FeatureExtractor()
        window = np.random.randn(20, 4)
        
        features = extractor._extract_statistical_features(window)
        
        # Should extract multiple statistical features per sensor
        assert len(features) > 0
        assert len(features) % 4 == 0  # Should be multiple of 4 sensors


class TestStreamingPipeline:
    """Test cases for StreamingPipeline."""
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        pipeline = StreamingPipeline(
            buffer_size=1000,
            window_size=60,
            sample_rate=1.0
        )
        assert pipeline.buffer_size == 1000
        assert pipeline.window_size == 60
        assert pipeline.sample_rate == 1.0
        assert len(pipeline.data_buffer) == 0
        assert len(pipeline.processors) == 0
    
    def test_add_data(self):
        """Test adding data to pipeline."""
        pipeline = StreamingPipeline(window_size=5)
        
        # Add some data
        for i in range(10):
            data = {"heart_rate": 70 + i, "temperature": 36.5}
            pipeline.add_data(data)
        
        assert len(pipeline.data_buffer) == 10
        assert pipeline.stats["total_samples"] == 10
    
    def test_add_processor(self):
        """Test adding processor to pipeline."""
        pipeline = StreamingPipeline()
        
        def test_processor(data, timestamps):
            pass
        
        pipeline.add_processor(test_processor)
        assert len(pipeline.processors) == 1
    
    def test_buffer_overflow(self):
        """Test buffer overflow handling."""
        pipeline = StreamingPipeline(buffer_size=5)
        
        # Add more data than buffer size
        for i in range(10):
            data = {"heart_rate": 70 + i}
            pipeline.add_data(data)
        
        assert len(pipeline.data_buffer) == 5
        assert pipeline.stats["buffer_overflows"] > 0
    
    def test_window_processing(self):
        """Test window processing."""
        pipeline = StreamingPipeline(window_size=3)
        
        processed_windows = []
        
        def test_processor(data, timestamps):
            processed_windows.append(len(data))
        
        pipeline.add_processor(test_processor)
        
        # Add enough data to trigger window processing
        for i in range(10):
            data = {"heart_rate": 70 + i}
            pipeline.add_data(data)
        
        assert len(processed_windows) > 0
        assert all(window_size == 3 for window_size in processed_windows)
    
    def test_get_stats(self):
        """Test statistics retrieval."""
        pipeline = StreamingPipeline()
        
        # Add some data
        for i in range(5):
            pipeline.add_data({"heart_rate": 70 + i})
        
        stats = pipeline.get_stats()
        
        assert "total_samples" in stats
        assert "processed_windows" in stats
        assert "buffer_overflows" in stats
        assert stats["total_samples"] == 5
    
    def test_clear_buffer(self):
        """Test buffer clearing."""
        pipeline = StreamingPipeline()
        
        # Add some data
        for i in range(5):
            pipeline.add_data({"heart_rate": 70 + i})
        
        assert len(pipeline.data_buffer) == 5
        
        pipeline.clear_buffer()
        
        assert len(pipeline.data_buffer) == 0
        assert pipeline.stats["total_samples"] == 0


if __name__ == "__main__":
    pytest.main([__file__])
