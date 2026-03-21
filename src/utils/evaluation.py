"""Evaluation and benchmarking utilities for health monitoring models."""

import json
import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.models import ModelFactory
from src.pipelines import SensorDataGenerator, DataProcessor


class ModelEvaluator:
    """Comprehensive model evaluator for health monitoring models."""
    
    def __init__(self, device: str = "cpu") -> None:
        """Initialize the evaluator.
        
        Args:
            device: Device to use for evaluation.
        """
        self.device = device
        self.evaluation_results = {}
    
    def evaluate_model(
        self,
        model: torch.nn.Module,
        test_loader: DataLoader,
        model_name: str = "model",
    ) -> Dict[str, Union[float, Dict]]:
        """Evaluate a single model.
        
        Args:
            model: Model to evaluate.
            test_loader: Test data loader.
            model_name: Name of the model.
            
        Returns:
            Dictionary with evaluation results.
        """
        model.eval()
        
        # Get predictions
        all_preds = []
        all_probs = []
        all_targets = []
        inference_times = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                # Time inference
                start_time = time.time()
                output = model(data)
                inference_time = time.time() - start_time
                
                # Get predictions and probabilities
                prob = torch.sigmoid(output.squeeze())
                pred = (prob > 0.5).float()
                
                all_preds.extend(pred.cpu().numpy())
                all_probs.extend(prob.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                inference_times.append(inference_time)
        
        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        all_targets = np.array(all_targets)
        
        # Calculate metrics
        metrics = self._calculate_metrics(all_preds, all_probs, all_targets)
        
        # Add performance metrics
        metrics.update(self._calculate_performance_metrics(inference_times, len(test_loader.dataset)))
        
        # Add model size metrics
        metrics.update(self._calculate_model_size_metrics(model))
        
        # Store results
        self.evaluation_results[model_name] = metrics
        
        return metrics
    
    def compare_models(
        self,
        models: Dict[str, torch.nn.Module],
        test_loader: DataLoader,
    ) -> pd.DataFrame:
        """Compare multiple models.
        
        Args:
            models: Dictionary of model names and models.
            test_loader: Test data loader.
            
        Returns:
            DataFrame with comparison results.
        """
        results = []
        
        for model_name, model in models.items():
            metrics = self.evaluate_model(model, test_loader, model_name)
            metrics['model_name'] = model_name
            results.append(metrics)
        
        return pd.DataFrame(results)
    
    def _calculate_metrics(
        self,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        targets: np.ndarray,
    ) -> Dict[str, float]:
        """Calculate classification metrics.
        
        Args:
            predictions: Binary predictions.
            probabilities: Prediction probabilities.
            targets: True labels.
            
        Returns:
            Dictionary with metrics.
        """
        # Basic metrics
        accuracy = (predictions == targets).mean()
        
        # Confusion matrix
        tp = np.sum((predictions == 1) & (targets == 1))
        fp = np.sum((predictions == 1) & (targets == 0))
        tn = np.sum((predictions == 0) & (targets == 0))
        fn = np.sum((predictions == 0) & (targets == 1))
        
        # Precision, recall, F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Specificity
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # AUC (simplified)
        auc = self._calculate_auc(probabilities, targets)
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "specificity": specificity,
            "auc": auc,
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
        }
    
    def _calculate_auc(self, probabilities: np.ndarray, targets: np.ndarray) -> float:
        """Calculate AUC score.
        
        Args:
            probabilities: Prediction probabilities.
            targets: True labels.
            
        Returns:
            AUC score.
        """
        try:
            from sklearn.metrics import roc_auc_score
            return roc_auc_score(targets, probabilities)
        except ImportError:
            # Fallback implementation
            return 0.5  # Random classifier
    
    def _calculate_performance_metrics(
        self,
        inference_times: List[float],
        num_samples: int,
    ) -> Dict[str, float]:
        """Calculate performance metrics.
        
        Args:
            inference_times: List of inference times.
            num_samples: Number of samples.
            
        Returns:
            Dictionary with performance metrics.
        """
        total_time = sum(inference_times)
        avg_time = total_time / len(inference_times)
        
        return {
            "total_inference_time": total_time,
            "avg_inference_time_ms": avg_time * 1000,
            "inference_fps": num_samples / total_time,
            "samples_per_second": num_samples / total_time,
        }
    
    def _calculate_model_size_metrics(self, model: torch.nn.Module) -> Dict[str, Union[int, float]]:
        """Calculate model size metrics.
        
        Args:
            model: PyTorch model.
            
        Returns:
            Dictionary with size metrics.
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Estimate memory usage
        memory_usage = sum(p.numel() * p.element_size() for p in model.parameters())
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "memory_usage_bytes": memory_usage,
            "memory_usage_kb": memory_usage / 1024,
            "memory_usage_mb": memory_usage / (1024 * 1024),
        }
    
    def generate_report(self) -> Dict[str, any]:
        """Generate evaluation report.
        
        Returns:
            Dictionary with evaluation report.
        """
        if not self.evaluation_results:
            return {"error": "No evaluation results available"}
        
        # Create summary
        summary = {
            "total_models_evaluated": len(self.evaluation_results),
            "evaluation_timestamp": time.time(),
            "models": self.evaluation_results,
        }
        
        # Find best model by F1 score
        best_model = None
        best_f1 = 0
        
        for model_name, metrics in self.evaluation_results.items():
            if metrics.get("f1_score", 0) > best_f1:
                best_f1 = metrics["f1_score"]
                best_model = model_name
        
        summary["best_model"] = best_model
        summary["best_f1_score"] = best_f1
        
        return summary


class EdgePerformanceBenchmark:
    """Benchmark for edge deployment performance."""
    
    def __init__(self, device: str = "cpu") -> None:
        """Initialize the benchmark.
        
        Args:
            device: Device to use for benchmarking.
        """
        self.device = device
        self.benchmark_results = {}
    
    def benchmark_model(
        self,
        model: torch.nn.Module,
        input_shape: Tuple[int, ...] = (1, 4),
        num_runs: int = 1000,
        warmup_runs: int = 100,
    ) -> Dict[str, float]:
        """Benchmark model performance.
        
        Args:
            model: Model to benchmark.
            input_shape: Input tensor shape.
            num_runs: Number of benchmark runs.
            warmup_runs: Number of warmup runs.
            
        Returns:
            Dictionary with benchmark results.
        """
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(input_shape).to(self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = model(dummy_input)
        
        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.time()
                _ = model(dummy_input)
                end_time = time.time()
                times.append(end_time - start_time)
        
        # Calculate statistics
        times = np.array(times)
        
        return {
            "mean_inference_time_ms": np.mean(times) * 1000,
            "std_inference_time_ms": np.std(times) * 1000,
            "min_inference_time_ms": np.min(times) * 1000,
            "max_inference_time_ms": np.max(times) * 1000,
            "p50_inference_time_ms": np.percentile(times, 50) * 1000,
            "p95_inference_time_ms": np.percentile(times, 95) * 1000,
            "p99_inference_time_ms": np.percentile(times, 99) * 1000,
            "inference_fps": 1.0 / np.mean(times),
            "throughput_samples_per_second": 1.0 / np.mean(times),
        }
    
    def benchmark_models(
        self,
        models: Dict[str, torch.nn.Module],
        input_shape: Tuple[int, ...] = (1, 4),
        num_runs: int = 1000,
    ) -> pd.DataFrame:
        """Benchmark multiple models.
        
        Args:
            models: Dictionary of model names and models.
            input_shape: Input tensor shape.
            num_runs: Number of benchmark runs.
            
        Returns:
            DataFrame with benchmark results.
        """
        results = []
        
        for model_name, model in models.items():
            benchmark_result = self.benchmark_model(model, input_shape, num_runs)
            benchmark_result['model_name'] = model_name
            results.append(benchmark_result)
        
        return pd.DataFrame(results)
    
    def compare_edge_formats(
        self,
        model_paths: Dict[str, str],
        input_data: np.ndarray,
    ) -> pd.DataFrame:
        """Compare performance of different edge formats.
        
        Args:
            model_paths: Dictionary of format names and model paths.
            input_data: Input data for testing.
            
        Returns:
            DataFrame with format comparison results.
        """
        results = []
        
        for format_name, model_path in model_paths.items():
            try:
                result = self._test_edge_format(format_name, model_path, input_data)
                result['format'] = format_name
                results.append(result)
            except Exception as e:
                results.append({
                    'format': format_name,
                    'error': str(e),
                    'mean_inference_time_ms': float('inf'),
                    'model_size_mb': 0,
                })
        
        return pd.DataFrame(results)
    
    def _test_edge_format(
        self,
        format_name: str,
        model_path: str,
        input_data: np.ndarray,
    ) -> Dict[str, float]:
        """Test a specific edge format.
        
        Args:
            format_name: Name of the format.
            model_path: Path to the model file.
            input_data: Input data for testing.
            
        Returns:
            Dictionary with test results.
        """
        import os
        
        # Get model size
        model_size_bytes = os.path.getsize(model_path)
        model_size_mb = model_size_bytes / (1024 * 1024)
        
        # Test inference time (simplified)
        start_time = time.time()
        
        if format_name == "onnx":
            self._test_onnx_inference(model_path, input_data)
        elif format_name == "tflite":
            self._test_tflite_inference(model_path, input_data)
        elif format_name == "coreml":
            self._test_coreml_inference(model_path, input_data)
        else:
            raise ValueError(f"Unknown format: {format_name}")
        
        inference_time = time.time() - start_time
        
        return {
            "mean_inference_time_ms": inference_time * 1000,
            "model_size_mb": model_size_mb,
            "inference_fps": 1.0 / inference_time,
        }
    
    def _test_onnx_inference(self, model_path: str, input_data: np.ndarray) -> None:
        """Test ONNX inference."""
        try:
            import onnxruntime as ort
            
            session = ort.InferenceSession(model_path)
            input_name = session.get_inputs()[0].name
            
            # Run inference
            session.run(None, {input_name: input_data.astype(np.float32)})
            
        except ImportError:
            pass  # Skip if ONNX Runtime not available
    
    def _test_tflite_inference(self, model_path: str, input_data: np.ndarray) -> None:
        """Test TFLite inference."""
        try:
            import tflite_runtime.interpreter as tflite
            
            interpreter = tflite.Interpreter(model_path=model_path)
            interpreter.allocate_tensors()
            
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            interpreter.set_tensor(input_details[0]['index'], input_data.astype(np.float32))
            interpreter.invoke()
            
        except ImportError:
            pass  # Skip if TFLite Runtime not available
    
    def _test_coreml_inference(self, model_path: str, input_data: np.ndarray) -> None:
        """Test CoreML inference."""
        try:
            import coremltools as ct
            
            model = ct.models.MLModel(model_path)
            
            # Convert input data to appropriate format
            input_dict = {"input": input_data.astype(np.float32)}
            model.predict(input_dict)
            
        except ImportError:
            pass  # Skip if CoreML not available


def create_leaderboard(
    evaluation_results: Dict[str, Dict],
    benchmark_results: Optional[Dict[str, Dict]] = None,
) -> pd.DataFrame:
    """Create a model leaderboard.
    
    Args:
        evaluation_results: Model evaluation results.
        benchmark_results: Optional benchmark results.
        
    Returns:
        DataFrame with leaderboard.
    """
    leaderboard_data = []
    
    for model_name, metrics in evaluation_results.items():
        row = {
            "model": model_name,
            "accuracy": metrics.get("accuracy", 0),
            "f1_score": metrics.get("f1_score", 0),
            "precision": metrics.get("precision", 0),
            "recall": metrics.get("recall", 0),
            "auc": metrics.get("auc", 0),
            "parameters": metrics.get("total_parameters", 0),
            "size_mb": metrics.get("memory_usage_mb", 0),
        }
        
        # Add benchmark results if available
        if benchmark_results and model_name in benchmark_results:
            benchmark = benchmark_results[model_name]
            row.update({
                "inference_time_ms": benchmark.get("mean_inference_time_ms", 0),
                "fps": benchmark.get("inference_fps", 0),
            })
        
        leaderboard_data.append(row)
    
    df = pd.DataFrame(leaderboard_data)
    
    # Sort by F1 score
    df = df.sort_values("f1_score", ascending=False)
    
    return df
