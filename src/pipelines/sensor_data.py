"""Wearable sensor data generation and reading utilities."""

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import signal


@dataclass
class SensorReading:
    """Container for a single sensor reading."""
    
    timestamp: float
    heart_rate: float  # BPM
    skin_temperature: float  # Celsius
    activity_level: float  # Normalized 0-1
    blood_oxygen: float  # SpO2 percentage
    device_id: str
    battery_level: Optional[float] = None  # Percentage
    signal_quality: Optional[float] = None  # 0-1


class SensorDataGenerator:
    """Generates realistic wearable sensor data for health monitoring.
    
    This class simulates various physiological signals that would be collected
    by wearable devices such as smartwatches, fitness bands, or medical-grade
    wearables.
    """
    
    def __init__(
        self,
        sample_rate: float = 1.0,  # Hz
        noise_level: float = 0.1,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize the sensor data generator.
        
        Args:
            sample_rate: Sampling rate in Hz.
            noise_level: Noise level for generated signals (0-1).
            seed: Random seed for reproducibility.
        """
        self.sample_rate = sample_rate
        self.noise_level = noise_level
        self.rng = np.random.RandomState(seed)
        
        # Physiological parameters
        self.baseline_hr = 75.0  # BPM
        self.baseline_temp = 36.5  # Celsius
        self.baseline_spo2 = 98.0  # Percentage
        self.baseline_activity = 0.3  # Normalized
        
    def generate_reading(
        self,
        device_id: str = "device_001",
        health_status: str = "normal",  # "normal", "stressed", "fatigued", "dehydrated"
        timestamp: Optional[float] = None,
    ) -> SensorReading:
        """Generate a single sensor reading.
        
        Args:
            device_id: Unique device identifier.
            health_status: Health status affecting the readings.
            timestamp: Unix timestamp (uses current time if None).
            
        Returns:
            SensorReading object with simulated data.
        """
        if timestamp is None:
            timestamp = time.time()
            
        # Generate base readings based on health status
        hr, temp, activity, spo2 = self._generate_physiological_data(health_status)
        
        # Add realistic noise
        hr += self.rng.normal(0, 2.0)
        temp += self.rng.normal(0, 0.1)
        activity += self.rng.normal(0, 0.05)
        spo2 += self.rng.normal(0, 0.5)
        
        # Clamp values to realistic ranges
        hr = np.clip(hr, 40, 200)
        temp = np.clip(temp, 35.0, 40.0)
        activity = np.clip(activity, 0.0, 1.0)
        spo2 = np.clip(spo2, 70.0, 100.0)
        
        # Generate additional metadata
        battery_level = self.rng.uniform(20, 100)
        signal_quality = self.rng.uniform(0.7, 1.0)
        
        return SensorReading(
            timestamp=timestamp,
            heart_rate=float(hr),
            skin_temperature=float(temp),
            activity_level=float(activity),
            blood_oxygen=float(spo2),
            device_id=device_id,
            battery_level=float(battery_level),
            signal_quality=float(signal_quality),
        )
    
    def generate_dataset(
        self,
        n_samples: int = 1000,
        device_ids: Optional[List[str]] = None,
        health_distribution: Optional[Dict[str, float]] = None,
    ) -> pd.DataFrame:
        """Generate a dataset of sensor readings.
        
        Args:
            n_samples: Number of samples to generate.
            device_ids: List of device IDs to use.
            health_distribution: Distribution of health statuses.
            
        Returns:
            DataFrame with sensor readings.
        """
        if device_ids is None:
            device_ids = [f"device_{i:03d}" for i in range(1, 6)]
            
        if health_distribution is None:
            health_distribution = {
                "normal": 0.7,
                "stressed": 0.15,
                "fatigued": 0.1,
                "dehydrated": 0.05,
            }
        
        readings = []
        health_statuses = list(health_distribution.keys())
        health_probs = list(health_distribution.values())
        
        for i in range(n_samples):
            device_id = self.rng.choice(device_ids)
            health_status = self.rng.choice(health_statuses, p=health_probs)
            
            reading = self.generate_reading(device_id, health_status)
            readings.append(reading)
        
        return pd.DataFrame(readings)
    
    def _generate_physiological_data(
        self, health_status: str
    ) -> Tuple[float, float, float, float]:
        """Generate physiological data based on health status.
        
        Args:
            health_status: Health status affecting the readings.
            
        Returns:
            Tuple of (heart_rate, temperature, activity, spo2).
        """
        if health_status == "normal":
            hr = self.baseline_hr + self.rng.normal(0, 5)
            temp = self.baseline_temp + self.rng.normal(0, 0.2)
            activity = self.baseline_activity + self.rng.normal(0, 0.1)
            spo2 = self.baseline_spo2 + self.rng.normal(0, 1)
            
        elif health_status == "stressed":
            hr = self.baseline_hr + 15 + self.rng.normal(0, 8)  # Elevated HR
            temp = self.baseline_temp + 0.3 + self.rng.normal(0, 0.2)
            activity = self.baseline_activity + 0.2 + self.rng.normal(0, 0.1)
            spo2 = self.baseline_spo2 - 2 + self.rng.normal(0, 1)
            
        elif health_status == "fatigued":
            hr = self.baseline_hr - 10 + self.rng.normal(0, 6)  # Lower HR
            temp = self.baseline_temp - 0.2 + self.rng.normal(0, 0.2)
            activity = self.baseline_activity - 0.15 + self.rng.normal(0, 0.1)
            spo2 = self.baseline_spo2 - 1 + self.rng.normal(0, 1)
            
        elif health_status == "dehydrated":
            hr = self.baseline_hr + 8 + self.rng.normal(0, 6)  # Slightly elevated
            temp = self.baseline_temp + 0.5 + self.rng.normal(0, 0.2)  # Higher temp
            activity = self.baseline_activity - 0.1 + self.rng.normal(0, 0.1)
            spo2 = self.baseline_spo2 - 3 + self.rng.normal(0, 1)  # Lower SpO2
            
        else:
            raise ValueError(f"Unknown health status: {health_status}")
        
        return hr, temp, activity, spo2


class WearableSensorReader(ABC):
    """Abstract base class for reading sensor data from wearable devices."""
    
    @abstractmethod
    def read_sensor_data(self) -> SensorReading:
        """Read sensor data from the device.
        
        Returns:
            SensorReading object with current sensor data.
        """
        pass
    
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if the device is connected.
        
        Returns:
            True if device is connected, False otherwise.
        """
        pass


class SimulatedWearableReader(WearableSensorReader):
    """Simulated wearable device reader for testing and development."""
    
    def __init__(
        self,
        device_id: str = "sim_device_001",
        generator: Optional[SensorDataGenerator] = None,
    ) -> None:
        """Initialize the simulated reader.
        
        Args:
            device_id: Unique device identifier.
            generator: Sensor data generator instance.
        """
        self.device_id = device_id
        self.generator = generator or SensorDataGenerator()
        self.connected = True
        
    def read_sensor_data(self) -> SensorReading:
        """Read simulated sensor data.
        
        Returns:
            SensorReading object with simulated data.
        """
        if not self.connected:
            raise ConnectionError("Device not connected")
            
        return self.generator.generate_reading(self.device_id)
    
    def is_connected(self) -> bool:
        """Check if the simulated device is connected.
        
        Returns:
            True if connected, False otherwise.
        """
        return self.connected
    
    def disconnect(self) -> None:
        """Simulate device disconnection."""
        self.connected = False
    
    def reconnect(self) -> None:
        """Simulate device reconnection."""
        self.connected = True
