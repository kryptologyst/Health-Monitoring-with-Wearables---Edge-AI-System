"""Model export utilities for edge deployment."""

import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import onnx
import torch
import torch.onnx
from onnx import version_converter


class ModelExporter:
    """Exporter for converting PyTorch models to edge deployment formats."""
    
    def __init__(self, model: torch.nn.Module, device: str = "cpu") -> None:
        """Initialize the exporter.
        
        Args:
            model: PyTorch model to export.
            device: Device to use for export.
        """
        self.model = model.to(device)
        self.device = device
        
        # Export statistics
        self.export_stats = {
            "exported_formats": [],
            "export_times": {},
            "model_sizes": {},
            "export_errors": {},
        }
    
    def export_to_onnx(
        self,
        output_path: str,
        input_shape: Tuple[int, ...] = (1, 4),
        opset_version: int = 11,
        dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
        verbose: bool = False,
    ) -> bool:
        """Export model to ONNX format.
        
        Args:
            output_path: Output file path.
            input_shape: Input tensor shape.
            opset_version: ONNX opset version.
            dynamic_axes: Dynamic axes configuration.
            verbose: Whether to print export details.
            
        Returns:
            True if export successful, False otherwise.
        """
        try:
            import time
            start_time = time.time()
            
            # Create dummy input
            dummy_input = torch.randn(input_shape).to(self.device)
            
            # Set model to evaluation mode
            self.model.eval()
            
            # Export to ONNX
            torch.onnx.export(
                self.model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes=dynamic_axes,
                verbose=verbose,
            )
            
            # Verify ONNX model
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            
            # Record statistics
            export_time = time.time() - start_time
            self.export_stats["exported_formats"].append("onnx")
            self.export_stats["export_times"]["onnx"] = export_time
            self.export_stats["model_sizes"]["onnx"] = os.path.getsize(output_path)
            
            return True
            
        except Exception as e:
            self.export_stats["export_errors"]["onnx"] = str(e)
            return False
    
    def export_to_tflite(
        self,
        output_path: str,
        input_shape: Tuple[int, ...] = (1, 4),
        quantize: bool = True,
        target_spec: Optional[Dict] = None,
    ) -> bool:
        """Export model to TensorFlow Lite format.
        
        Args:
            output_path: Output file path.
            input_shape: Input tensor shape.
            quantize: Whether to quantize the model.
            target_spec: Target specification for quantization.
            
        Returns:
            True if export successful, False otherwise.
        """
        try:
            import time
            start_time = time.time()
            
            # First export to ONNX
            onnx_path = output_path.replace('.tflite', '.onnx')
            if not self.export_to_onnx(onnx_path, input_shape):
                return False
            
            # Convert ONNX to TensorFlow
            tf_path = output_path.replace('.tflite', '_tf')
            if not self._onnx_to_tf(onnx_path, tf_path):
                return False
            
            # Convert TensorFlow to TFLite
            if not self._tf_to_tflite(tf_path, output_path, quantize, target_spec):
                return False
            
            # Record statistics
            export_time = time.time() - start_time
            self.export_stats["exported_formats"].append("tflite")
            self.export_stats["export_times"]["tflite"] = export_time
            self.export_stats["model_sizes"]["tflite"] = os.path.getsize(output_path)
            
            return True
            
        except Exception as e:
            self.export_stats["export_errors"]["tflite"] = str(e)
            return False
    
    def export_to_coreml(
        self,
        output_path: str,
        input_shape: Tuple[int, ...] = (1, 4),
        compute_units: str = "cpu",
    ) -> bool:
        """Export model to CoreML format.
        
        Args:
            output_path: Output file path.
            input_shape: Input tensor shape.
            compute_units: CoreML compute units ("cpu", "gpu", "all").
            
        Returns:
            True if export successful, False otherwise.
        """
        try:
            import time
            start_time = time.time()
            
            # First export to ONNX
            onnx_path = output_path.replace('.mlmodel', '.onnx')
            if not self.export_to_onnx(onnx_path, input_shape):
                return False
            
            # Convert ONNX to CoreML
            if not self._onnx_to_coreml(onnx_path, output_path, compute_units):
                return False
            
            # Record statistics
            export_time = time.time() - start_time
            self.export_stats["exported_formats"].append("coreml")
            self.export_stats["export_times"]["coreml"] = export_time
            self.export_stats["model_sizes"]["coreml"] = os.path.getsize(output_path)
            
            return True
            
        except Exception as e:
            self.export_stats["export_errors"]["coreml"] = str(e)
            return False
    
    def export_to_openvino(
        self,
        output_path: str,
        input_shape: Tuple[int, ...] = (1, 4),
        precision: str = "FP32",
    ) -> bool:
        """Export model to OpenVINO format.
        
        Args:
            output_path: Output file path.
            input_shape: Input tensor shape.
            precision: Model precision ("FP32", "FP16", "INT8").
            
        Returns:
            True if export successful, False otherwise.
        """
        try:
            import time
            start_time = time.time()
            
            # First export to ONNX
            onnx_path = output_path.replace('.xml', '.onnx')
            if not self.export_to_onnx(onnx_path, input_shape):
                return False
            
            # Convert ONNX to OpenVINO
            if not self._onnx_to_openvino(onnx_path, output_path, precision):
                return False
            
            # Record statistics
            export_time = time.time() - start_time
            self.export_stats["exported_formats"].append("openvino")
            self.export_stats["export_times"]["openvino"] = export_time
            self.export_stats["model_sizes"]["openvino"] = os.path.getsize(output_path)
            
            return True
            
        except Exception as e:
            self.export_stats["export_errors"]["openvino"] = str(e)
            return False
    
    def export_all_formats(
        self,
        base_path: str,
        input_shape: Tuple[int, ...] = (1, 4),
        formats: Optional[List[str]] = None,
    ) -> Dict[str, bool]:
        """Export model to all supported formats.
        
        Args:
            base_path: Base path for output files.
            input_shape: Input tensor shape.
            formats: List of formats to export (default: all).
            
        Returns:
            Dictionary with export results for each format.
        """
        if formats is None:
            formats = ["onnx", "tflite", "coreml", "openvino"]
        
        results = {}
        
        for format_name in formats:
            if format_name == "onnx":
                output_path = f"{base_path}.onnx"
                results[format_name] = self.export_to_onnx(output_path, input_shape)
            elif format_name == "tflite":
                output_path = f"{base_path}.tflite"
                results[format_name] = self.export_to_tflite(output_path, input_shape)
            elif format_name == "coreml":
                output_path = f"{base_path}.mlmodel"
                results[format_name] = self.export_to_coreml(output_path, input_shape)
            elif format_name == "openvino":
                output_path = f"{base_path}.xml"
                results[format_name] = self.export_to_openvino(output_path, input_shape)
            else:
                results[format_name] = False
        
        return results
    
    def _onnx_to_tf(self, onnx_path: str, tf_path: str) -> bool:
        """Convert ONNX model to TensorFlow format.
        
        Args:
            onnx_path: Path to ONNX model.
            tf_path: Path to save TensorFlow model.
            
        Returns:
            True if conversion successful, False otherwise.
        """
        try:
            import onnx_tf
            from onnx_tf.backend import prepare
            
            # Load ONNX model
            onnx_model = onnx.load(onnx_path)
            
            # Convert to TensorFlow
            tf_rep = prepare(onnx_model)
            tf_rep.export_graph(tf_path)
            
            return True
            
        except Exception as e:
            print(f"Error converting ONNX to TensorFlow: {e}")
            return False
    
    def _tf_to_tflite(
        self,
        tf_path: str,
        tflite_path: str,
        quantize: bool = True,
        target_spec: Optional[Dict] = None,
    ) -> bool:
        """Convert TensorFlow model to TFLite format.
        
        Args:
            tf_path: Path to TensorFlow model.
            tflite_path: Path to save TFLite model.
            quantize: Whether to quantize the model.
            target_spec: Target specification for quantization.
            
        Returns:
            True if conversion successful, False otherwise.
        """
        try:
            import tensorflow as tf
            
            # Load TensorFlow model
            converter = tf.lite.TFLiteConverter.from_saved_model(tf_path)
            
            # Set optimization flags
            if quantize:
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                if target_spec:
                    converter.target_spec = target_spec
            
            # Convert to TFLite
            tflite_model = converter.convert()
            
            # Save TFLite model
            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)
            
            return True
            
        except Exception as e:
            print(f"Error converting TensorFlow to TFLite: {e}")
            return False
    
    def _onnx_to_coreml(
        self,
        onnx_path: str,
        coreml_path: str,
        compute_units: str = "cpu",
    ) -> bool:
        """Convert ONNX model to CoreML format.
        
        Args:
            onnx_path: Path to ONNX model.
            coreml_path: Path to save CoreML model.
            compute_units: CoreML compute units.
            
        Returns:
            True if conversion successful, False otherwise.
        """
        try:
            import coremltools as ct
            
            # Load ONNX model
            onnx_model = onnx.load(onnx_path)
            
            # Convert to CoreML
            coreml_model = ct.convert(onnx_model)
            
            # Set compute units
            if compute_units == "gpu":
                coreml_model.compute_units = ct.ComputeUnit.GPU_ONLY
            elif compute_units == "all":
                coreml_model.compute_units = ct.ComputeUnit.ALL
            else:
                coreml_model.compute_units = ct.ComputeUnit.CPU_ONLY
            
            # Save CoreML model
            coreml_model.save(coreml_path)
            
            return True
            
        except Exception as e:
            print(f"Error converting ONNX to CoreML: {e}")
            return False
    
    def _onnx_to_openvino(
        self,
        onnx_path: str,
        openvino_path: str,
        precision: str = "FP32",
    ) -> bool:
        """Convert ONNX model to OpenVINO format.
        
        Args:
            onnx_path: Path to ONNX model.
            openvino_path: Path to save OpenVINO model.
            precision: Model precision.
            
        Returns:
            True if conversion successful, False otherwise.
        """
        try:
            from openvino.tools.mo import main as mo_main
            import sys
            
            # Prepare arguments for Model Optimizer
            mo_args = [
                '--input_model', onnx_path,
                '--output_dir', os.path.dirname(openvino_path),
                '--model_name', os.path.basename(openvino_path).replace('.xml', ''),
                '--data_type', precision,
            ]
            
            # Run Model Optimizer
            sys.argv = ['mo'] + mo_args
            mo_main()
            
            return True
            
        except Exception as e:
            print(f"Error converting ONNX to OpenVINO: {e}")
            return False
    
    def get_export_report(self) -> Dict[str, any]:
        """Get export report with statistics.
        
        Returns:
            Dictionary with export statistics.
        """
        return self.export_stats.copy()
    
    def cleanup_temp_files(self, base_path: str) -> None:
        """Clean up temporary files created during export.
        
        Args:
            base_path: Base path for temporary files.
        """
        temp_files = [
            f"{base_path}.onnx",
            f"{base_path}_tf",
        ]
        
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                if os.path.isdir(temp_file):
                    import shutil
                    shutil.rmtree(temp_file)
                else:
                    os.remove(temp_file)
