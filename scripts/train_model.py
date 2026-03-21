"""Main training script for health monitoring models."""

import argparse
import logging
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split

from src import set_seed, get_device
from src.models import ModelFactory
from src.pipelines import SensorDataGenerator, DataProcessor, FeatureExtractor
from src.export import ModelExporter


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration.
    
    Args:
        log_level: Logging level.
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )


def generate_dataset(
    n_samples: int = 5000,
    device_ids: Optional[List[str]] = None,
    health_distribution: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """Generate synthetic health monitoring dataset.
    
    Args:
        n_samples: Number of samples to generate.
        device_ids: List of device IDs.
        health_distribution: Distribution of health statuses.
        
    Returns:
        Generated dataset.
    """
    generator = SensorDataGenerator(seed=42)
    
    # Generate dataset
    df = generator.generate_dataset(
        n_samples=n_samples,
        device_ids=device_ids,
        health_distribution=health_distribution,
    )
    
    # Create alert labels based on physiological thresholds
    df['alert'] = (
        (df['heart_rate'] > 90) & 
        (df['blood_oxygen'] < 95) & 
        (df['activity_level'] < 0.3)
    ) | (df['skin_temperature'] > 37.5)
    
    return df


def prepare_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.1,
    batch_size: int = 32,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Prepare data for training.
    
    Args:
        df: Input dataset.
        test_size: Test set size ratio.
        val_size: Validation set size ratio.
        batch_size: Batch size for data loaders.
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    # Process data
    processor = DataProcessor()
    df_processed = processor.process_dataframe(df, fit_scaler=True)
    
    # Extract features
    feature_cols = ["heart_rate_scaled", "skin_temperature_scaled", 
                    "activity_level_scaled", "blood_oxygen_scaled"]
    X = df_processed[feature_cols].values
    y = df_processed['alert'].values.astype(np.float32)
    
    # Create dataset
    dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X),
        torch.FloatTensor(y)
    )
    
    # Split dataset
    total_size = len(dataset)
    test_size = int(total_size * test_size)
    val_size = int(total_size * val_size)
    train_size = total_size - test_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


def train_model(
    model_type: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    epochs: int = 50,
    learning_rate: float = 0.001,
    patience: int = 10,
) -> Tuple[torch.nn.Module, Dict[str, List[float]]]:
    """Train a health monitoring model.
    
    Args:
        model_type: Type of model to train.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        device: Device to use for training.
        epochs: Number of training epochs.
        learning_rate: Learning rate.
        patience: Early stopping patience.
        
    Returns:
        Tuple of (trained_model, training_history).
    """
    # Create model
    model = ModelFactory.create_model(model_type, input_size=4)
    
    # Create trainer
    trainer = ModelFactory.create_trainer(
        model,
        trainer_type="qat" if model_type == "quantized" else "standard",
        device=device,
        learning_rate=learning_rate,
    )
    
    # Train model
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        patience=patience,
        verbose=True,
    )
    
    return model, history


def evaluate_model(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: str,
) -> Dict[str, float]:
    """Evaluate model performance.
    
    Args:
        model: Trained model.
        test_loader: Test data loader.
        device: Device to use for evaluation.
        
    Returns:
        Dictionary with evaluation metrics.
    """
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            pred = (torch.sigmoid(output.squeeze()) > 0.5).float()
            
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    # Calculate metrics
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    accuracy = (all_preds == all_targets).mean()
    precision = np.sum((all_preds == 1) & (all_targets == 1)) / np.sum(all_preds == 1) if np.sum(all_preds == 1) > 0 else 0
    recall = np.sum((all_preds == 1) & (all_targets == 1)) / np.sum(all_targets == 1) if np.sum(all_targets == 1) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }


def export_model(
    model: torch.nn.Module,
    model_name: str,
    output_dir: str,
    device: str,
) -> Dict[str, bool]:
    """Export model to edge deployment formats.
    
    Args:
        model: Trained model.
        model_name: Name for the exported model.
        device: Device to use for export.
        
    Returns:
        Dictionary with export results.
    """
    exporter = ModelExporter(model, device=device)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    base_path = os.path.join(output_dir, model_name)
    
    # Export to all formats
    results = exporter.export_all_formats(base_path)
    
    # Clean up temporary files
    exporter.cleanup_temp_files(base_path)
    
    return results


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train health monitoring models")
    parser.add_argument("--model-type", type=str, default="baseline",
                       choices=["baseline", "edge", "tiny", "quantized"],
                       help="Type of model to train")
    parser.add_argument("--n-samples", type=int, default=5000,
                       help="Number of samples to generate")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=0.001,
                       help="Learning rate")
    parser.add_argument("--patience", type=int, default=10,
                       help="Early stopping patience")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use for training")
    parser.add_argument("--output-dir", type=str, default="models",
                       help="Output directory for models")
    parser.add_argument("--log-level", type=str, default="INFO",
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Setup
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Set random seed
    set_seed(42)
    
    # Get device
    device = get_device(args.device)
    logger.info(f"Using device: {device}")
    
    # Generate dataset
    logger.info("Generating dataset...")
    df = generate_dataset(n_samples=args.n_samples)
    logger.info(f"Generated dataset with {len(df)} samples")
    
    # Prepare data
    logger.info("Preparing data...")
    train_loader, val_loader, test_loader = prepare_data(
        df, batch_size=args.batch_size
    )
    logger.info(f"Data split - Train: {len(train_loader.dataset)}, "
                f"Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")
    
    # Train model
    logger.info(f"Training {args.model_type} model...")
    start_time = time.time()
    model, history = train_model(
        model_type=args.model_type,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        patience=args.patience,
    )
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")
    
    # Evaluate model
    logger.info("Evaluating model...")
    metrics = evaluate_model(model, test_loader, device)
    logger.info(f"Test metrics: {metrics}")
    
    # Export model
    logger.info("Exporting model...")
    export_results = export_model(
        model=model,
        model_name=f"{args.model_type}_health_model",
        output_dir=args.output_dir,
        device=device,
    )
    logger.info(f"Export results: {export_results}")
    
    # Save training history
    history_path = os.path.join(args.output_dir, f"{args.model_type}_history.json")
    import json
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()
