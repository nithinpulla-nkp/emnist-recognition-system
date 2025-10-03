"""
EMNIST Character Recognition System

A production-ready deep learning system for handwritten character recognition
using Convolutional Neural Networks on the EMNIST dataset.

This package provides:
- Multiple CNN architectures (Base CNN, VGG-13, custom models)
- Professional data processing pipelines
- Training utilities with advanced optimization techniques
- REST API for real-time inference
- Model monitoring and evaluation tools
"""

__version__ = "1.0.0"
__author__ = "Nithin Pulla, Sai Venkata Sathwik Golla"
__email__ = "npulla2@buffalo.edu, sgolla2@buffalo.edu"

from .models import *
from .data import *
from .training import *
from .inference import *
from .utils import *