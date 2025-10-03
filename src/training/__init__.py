"""
Training utilities and experiment management for EMNIST character recognition.

This module provides comprehensive training capabilities including:
- Training loops with advanced optimization techniques
- Experiment tracking and logging
- Model evaluation and validation
- Hyperparameter optimization
"""

from .trainer import Trainer, TrainingConfig
from .optimizer import OptimizerFactory, SchedulerFactory
from .metrics import MetricsCalculator, PerformanceTracker
from .experiment import ExperimentManager

__all__ = [
    'Trainer',
    'TrainingConfig',
    'OptimizerFactory',
    'SchedulerFactory',
    'MetricsCalculator',
    'PerformanceTracker',
    'ExperimentManager'
]