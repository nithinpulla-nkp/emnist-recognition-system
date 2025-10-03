#!/usr/bin/env python3
"""
Training script for EMNIST Character Recognition models.

This script provides a command-line interface for training various CNN models
with configurable parameters and advanced optimization techniques.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models import ModelFactory
from src.data import create_data_loaders
from src.training import Trainer, TrainingConfig
from src.utils.config import load_config
from src.utils.logging import setup_logging


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train EMNIST Character Recognition Models"
    )

    # Model configuration
    parser.add_argument(
        "--config",
        type=str,
        default="configs/model_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="base_cnn",
        choices=["base_cnn", "vgg13"],
        help="Model architecture to train"
    )

    # Training parameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate"
    )

    # Advanced options
    parser.add_argument(
        "--k-fold",
        type=int,
        default=None,
        help="Number of folds for K-fold cross-validation"
    )
    parser.add_argument(
        "--early-stopping",
        action="store_true",
        help="Enable early stopping"
    )
    parser.add_argument(
        "--scheduler",
        action="store_true",
        help="Enable learning rate scheduling"
    )

    # Data and output
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/emnist",
        help="Path to dataset directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments",
        help="Output directory for models and logs"
    )

    # Compute options
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cpu, cuda, mps)"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loader workers"
    )

    # Logging
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Experiment name for tracking"
    )

    return parser.parse_args()


def get_device(device_arg: str) -> torch.device:
    """Get the appropriate device for training."""
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    else:
        return torch.device(device_arg)


def main():
    """Main training function."""
    args = parse_arguments()

    # Setup logging
    setup_logging(level=args.log_level)
    logger = logging.getLogger(__name__)

    logger.info("Starting EMNIST Character Recognition Training")
    logger.info(f"Arguments: {vars(args)}")

    try:
        # Load configuration
        config = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")

        # Override config with command line arguments
        config['training']['epochs'] = args.epochs
        config['training']['batch_size'] = args.batch_size
        config['training']['learning_rate'] = args.learning_rate

        if args.early_stopping:
            config['training']['early_stopping']['enabled'] = True

        if args.scheduler:
            config['training']['scheduler']['type'] = 'step'

        if args.k_fold:
            config['training']['k_fold']['enabled'] = True
            config['training']['k_fold']['folds'] = args.k_fold

        # Get device
        device = get_device(args.device)
        logger.info(f"Using device: {device}")

        # Create data loaders
        logger.info("Creating data loaders...")
        train_loader, val_loader, test_loader = create_data_loaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            config=config['dataset']
        )

        logger.info(f"Dataset sizes - Train: {len(train_loader.dataset)}, "
                   f"Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")

        # Create model
        logger.info(f"Creating {args.model} model...")
        model = ModelFactory.create_model(
            model_name=args.model,
            config=config['models'][args.model],
            device=device
        )

        logger.info(f"Model created: {model}")

        # Create training configuration
        training_config = TrainingConfig(
            model_name=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=device,
            output_dir=args.output_dir,
            experiment_name=args.experiment_name or f"{args.model}_training",
            **config['training']
        )

        # Initialize trainer
        trainer = Trainer(training_config)

        # Train model
        logger.info("Starting training...")
        if args.k_fold:
            # K-fold cross-validation
            results = trainer.train_k_fold(
                dataset=train_loader.dataset,
                k_folds=args.k_fold
            )
            logger.info(f"K-fold CV results: {results}")
        else:
            # Standard training
            model, history = trainer.train(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader
            )

            # Final evaluation
            logger.info("Evaluating on test set...")
            test_metrics = trainer.evaluate(model, test_loader)
            logger.info(f"Test metrics: {test_metrics}")

            # Save final model
            model_path = Path(args.output_dir) / "models" / f"final_{args.model}.pth"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': config,
                'test_metrics': test_metrics,
                'training_history': history
            }, model_path)

            logger.info(f"Model saved to {model_path}")

        logger.info("Training completed successfully!")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()