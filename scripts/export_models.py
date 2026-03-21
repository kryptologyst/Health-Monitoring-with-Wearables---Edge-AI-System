"""Export models to edge deployment formats."""

import argparse
import os
from typing import Dict, List

import torch

from src.export import ModelExporter
from src.models import ModelFactory


def export_models(
    input_dir: str = "models",
    output_dir: str = "assets",
    formats: List[str] = None,
) -> Dict[str, bool]:
    """Export trained models to edge deployment formats.
    
    Args:
        input_dir: Directory containing trained models.
        output_dir: Directory to save exported models.
        formats: List of formats to export.
        
    Returns:
        Dictionary with export results.
    """
    if formats is None:
        formats = ["onnx", "tflite", "coreml", "openvino"]
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    export_results = {}
    
    # Find model files
    model_files = []
    for file in os.listdir(input_dir):
        if file.endswith('.pth'):
            model_name = file.replace('.pth', '')
            model_files.append((model_name, os.path.join(input_dir, file)))
    
    if not model_files:
        print("No trained models found in input directory")
        return {}
    
    # Export each model
    for model_name, model_path in model_files:
        print(f"Exporting {model_name}...")
        
        try:
            # Load model
            model_type = model_name.split('_')[0]  # Extract model type from filename
            model = ModelFactory.create_model(model_type, input_size=4)
            
            if torch.cuda.is_available():
                model.load_state_dict(torch.load(model_path))
            else:
                model.load_state_dict(torch.load(model_path, map_location='cpu'))
            
            model.eval()
            
            # Export to different formats
            exporter = ModelExporter(model, device="cpu")
            base_path = os.path.join(output_dir, model_name)
            
            results = exporter.export_all_formats(base_path, formats=formats)
            export_results[model_name] = results
            
            # Print results
            print(f"  Export results for {model_name}:")
            for format_name, success in results.items():
                status = "✓" if success else "✗"
                print(f"    {format_name}: {status}")
            
            # Clean up temporary files
            exporter.cleanup_temp_files(base_path)
            
        except Exception as e:
            print(f"  Error exporting {model_name}: {e}")
            export_results[model_name] = {format_name: False for format_name in formats}
    
    return export_results


def main():
    """Main export function."""
    parser = argparse.ArgumentParser(description="Export models to edge deployment formats")
    parser.add_argument("--input-dir", type=str, default="models",
                       help="Directory containing trained models")
    parser.add_argument("--output-dir", type=str, default="assets",
                       help="Directory to save exported models")
    parser.add_argument("--formats", nargs="+", default=["onnx", "tflite", "coreml", "openvino"],
                       help="Formats to export")
    
    args = parser.parse_args()
    
    print("Starting model export...")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Formats: {args.formats}")
    
    # Export models
    results = export_models(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        formats=args.formats,
    )
    
    # Print summary
    print("\nExport Summary:")
    total_models = len(results)
    successful_exports = sum(1 for model_results in results.values() 
                           if any(model_results.values()))
    
    print(f"Total models processed: {total_models}")
    print(f"Successfully exported: {successful_exports}")
    
    if successful_exports < total_models:
        print("Some models failed to export. Check the error messages above.")


if __name__ == "__main__":
    main()
