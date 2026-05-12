"""
Age and Gender Detection Project
Package initialization for the main source code
"""

__version__ = "1.0.0"
__author__ = "Ritik S"
__email__ = "ritik@example.com"

from . import data_loader
from . import model
from . import utils

__all__ = ['data_loader', 'model', 'utils']
