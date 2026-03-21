"""Data processing and feature extraction for health monitoring."""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import signal, stats
from sklearn.preprocessing import StandardScaler, RobustScaler


class DataProcessor:
    """Data processor for wearable sensor data.
    
    Handles data cleaning, normalization, and preprocessing for health
    monitoring applications.
    """
    
    def __init__(
        self,
        scaler_type: str = "robust",
        remove_outliers: bool = True,
        outlier_threshold: float = 3.0,
    ) -> None:
        """Initialize the data processor.
        
        Args:
            scaler_type: Type of scaler ("standard", "robust", "minmax").
            remove_outliers: Whether to remove outliers.
            outlier_threshold: Z-score threshold for outlier detection.
        """
        self.scaler_type = scaler_type
        self.remove_outliers = remove_outliers
        self.outlier_threshold = outlier_threshold
        
        # Initialize scaler
        if scaler_type == "standard":
            self.scaler = StandardScaler()
        elif scaler_type == "robust":
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")
        
        self.is_fitted = False
        
    def process_dataframe(
        self, 
        df: pd.DataFrame, 
        fit_scaler: bool = True
    ) -> pd.DataFrame:
        """Process a DataFrame of sensor readings.
        
        Args:
            df: DataFrame with sensor readings.
            fit_scaler: Whether to fit the scaler on this data.
            
        Returns:
            Processed DataFrame.
        """
        df_processed = df.copy()
        
        # Remove outliers if requested
        if self.remove_outliers:
            df_processed = self._remove_outliers(df_processed)
        
        # Extract features
        feature_cols = ["heart_rate", "skin_temperature", "activity_level", "blood_oxygen"]
        X = df_processed[feature_cols].values
        
        # Scale features
        if fit_scaler:
            X_scaled = self.scaler.fit_transform(X)
            self.is_fitted = True
        else:
            if not self.is_fitted:
                raise ValueError("Scaler must be fitted before transforming data")
            X_scaled = self.scaler.transform(X)
        
        # Update DataFrame with scaled values
        for i, col in enumerate(feature_cols):
            df_processed[f"{col}_scaled"] = X_scaled[:, i]
        
        return df_processed
    
    def process_single_reading(
        self, 
        reading: "SensorReading"
    ) -> np.ndarray:
        """Process a single sensor reading.
        
        Args:
            reading: SensorReading object.
            
        Returns:
            Processed feature array.
        """
        features = np.array([
            reading.heart_rate,
            reading.skin_temperature,
            reading.activity_level,
            reading.blood_oxygen,
        ]).reshape(1, -1)
        
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before processing single readings")
        
        return self.scaler.transform(features)
    
    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers from the DataFrame.
        
        Args:
            df: Input DataFrame.
            
        Returns:
            DataFrame with outliers removed.
        """
        feature_cols = ["heart_rate", "skin_temperature", "activity_level", "blood_oxygen"]
        
        # Calculate Z-scores
        z_scores = np.abs(stats.zscore(df[feature_cols]))
        
        # Keep rows where all features are within threshold
        mask = (z_scores < self.outlier_threshold).all(axis=1)
        
        return df[mask].reset_index(drop=True)


class FeatureExtractor:
    """Feature extractor for health monitoring data.
    
    Extracts temporal, statistical, and frequency-domain features
    from wearable sensor data.
    """
    
    def __init__(
        self,
        window_size: int = 60,  # seconds
        overlap: float = 0.5,
        sample_rate: float = 1.0,  # Hz
    ) -> None:
        """Initialize the feature extractor.
        
        Args:
            window_size: Window size in seconds for feature extraction.
            overlap: Overlap ratio between windows (0-1).
            sample_rate: Sampling rate in Hz.
        """
        self.window_size = window_size
        self.overlap = overlap
        self.sample_rate = sample_rate
        self.window_samples = int(window_size * sample_rate)
        self.step_size = int(self.window_samples * (1 - overlap))
    
    def extract_features(
        self, 
        df: pd.DataFrame, 
        target_col: Optional[str] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Extract features from sensor data.
        
        Args:
            df: DataFrame with sensor readings sorted by timestamp.
            target_col: Target column name for supervised learning.
            
        Returns:
            Tuple of (features, targets) where targets can be None.
        """
        feature_cols = ["heart_rate", "skin_temperature", "activity_level", "blood_oxygen"]
        data = df[feature_cols].values
        
        features = []
        targets = [] if target_col else None
        
        # Extract features from sliding windows
        for i in range(0, len(data) - self.window_samples + 1, self.step_size):
            window = data[i:i + self.window_samples]
            
            # Extract features for this window
            window_features = self._extract_window_features(window)
            features.append(window_features)
            
            # Extract target for this window
            if target_col is not None:
                window_target = self._extract_window_target(df, i, target_col)
                targets.append(window_target)
        
        features = np.array(features)
        if targets is not None:
            targets = np.array(targets)
        
        return features, targets
    
    def _extract_window_features(self, window: np.ndarray) -> np.ndarray:
        """Extract features from a single window.
        
        Args:
            window: Window of sensor data (n_samples, n_features).
            
        Returns:
            Feature vector.
        """
        features = []
        
        # Statistical features
        features.extend(self._extract_statistical_features(window))
        
        # Temporal features
        features.extend(self._extract_temporal_features(window))
        
        # Frequency domain features
        features.extend(self._extract_frequency_features(window))
        
        # Health-specific features
        features.extend(self._extract_health_features(window))
        
        return np.array(features)
    
    def _extract_statistical_features(self, window: np.ndarray) -> List[float]:
        """Extract statistical features from window.
        
        Args:
            window: Window of sensor data.
            
        Returns:
            List of statistical features.
        """
        features = []
        
        for i in range(window.shape[1]):  # For each sensor
            signal = window[:, i]
            
            # Basic statistics
            features.extend([
                np.mean(signal),
                np.std(signal),
                np.var(signal),
                np.median(signal),
                np.percentile(signal, 25),
                np.percentile(signal, 75),
                np.min(signal),
                np.max(signal),
                stats.skew(signal),
                stats.kurtosis(signal),
            ])
        
        return features
    
    def _extract_temporal_features(self, window: np.ndarray) -> List[float]:
        """Extract temporal features from window.
        
        Args:
            window: Window of sensor data.
            
        Returns:
            List of temporal features.
        """
        features = []
        
        for i in range(window.shape[1]):  # For each sensor
            signal = window[:, i]
            
            # Trend features
            x = np.arange(len(signal))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, signal)
            features.extend([slope, r_value, p_value])
            
            # Change features
            diff = np.diff(signal)
            features.extend([
                np.mean(np.abs(diff)),
                np.std(diff),
                np.sum(diff > 0) / len(diff),  # Proportion of increases
            ])
        
        return features
    
    def _extract_frequency_features(self, window: np.ndarray) -> List[float]:
        """Extract frequency domain features from window.
        
        Args:
            window: Window of sensor data.
            
        Returns:
            List of frequency features.
        """
        features = []
        
        for i in range(window.shape[1]):  # For each sensor
            signal = window[:, i]
            
            # FFT features
            fft = np.fft.fft(signal)
            power_spectrum = np.abs(fft) ** 2
            freqs = np.fft.fftfreq(len(signal), 1/self.sample_rate)
            
            # Keep only positive frequencies
            positive_freqs = freqs[:len(freqs)//2]
            positive_power = power_spectrum[:len(power_spectrum)//2]
            
            # Spectral features
            features.extend([
                np.sum(positive_power),  # Total power
                np.mean(positive_power),  # Average power
                np.std(positive_power),   # Power variability
                positive_freqs[np.argmax(positive_power)],  # Dominant frequency
            ])
        
        return features
    
    def _extract_health_features(self, window: np.ndarray) -> List[float]:
        """Extract health-specific features from window.
        
        Args:
            window: Window of sensor data [hr, temp, activity, spo2].
            
        Returns:
            List of health-specific features.
        """
        features = []
        
        if window.shape[1] >= 4:
            hr = window[:, 0]
            temp = window[:, 1]
            activity = window[:, 2]
            spo2 = window[:, 3]
            
            # Heart rate variability (simplified)
            hr_diff = np.diff(hr)
            features.extend([
                np.std(hr_diff),  # HRV (simplified)
                np.mean(hr),      # Average HR
                np.std(hr),       # HR variability
            ])
            
            # Temperature trends
            temp_diff = np.diff(temp)
            features.extend([
                np.mean(temp_diff),  # Temperature trend
                np.std(temp),        # Temperature stability
            ])
            
            # Activity patterns
            features.extend([
                np.mean(activity),   # Average activity
                np.std(activity),    # Activity variability
                np.sum(activity > 0.5) / len(activity),  # High activity proportion
            ])
            
            # SpO2 stability
            features.extend([
                np.mean(spo2),       # Average SpO2
                np.std(spo2),        # SpO2 variability
                np.sum(spo2 < 95) / len(spo2),  # Low SpO2 proportion
            ])
            
            # Cross-sensor correlations
            if len(hr) > 1:
                features.extend([
                    np.corrcoef(hr, temp)[0, 1] if not np.isnan(np.corrcoef(hr, temp)[0, 1]) else 0,
                    np.corrcoef(hr, activity)[0, 1] if not np.isnan(np.corrcoef(hr, activity)[0, 1]) else 0,
                    np.corrcoef(temp, spo2)[0, 1] if not np.isnan(np.corrcoef(temp, spo2)[0, 1]) else 0,
                ])
        
        return features
    
    def _extract_window_target(
        self, 
        df: pd.DataFrame, 
        start_idx: int, 
        target_col: str
    ) -> float:
        """Extract target value for a window.
        
        Args:
            df: Full DataFrame.
            start_idx: Starting index of the window.
            target_col: Target column name.
            
        Returns:
            Target value for the window.
        """
        window_df = df.iloc[start_idx:start_idx + self.window_samples]
        
        if target_col in df.columns:
            # For binary classification, use majority vote
            if df[target_col].dtype in ['int64', 'bool']:
                return int(window_df[target_col].mode().iloc[0] if len(window_df[target_col].mode()) > 0 else 0)
            else:
                # For regression, use mean
                return float(window_df[target_col].mean())
        else:
            return 0.0
