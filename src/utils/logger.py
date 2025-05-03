"""
Logging utilities for the Luminara CRF model.

This module provides a customized logger setup for the Luminara framework.
"""

import logging
import os
import sys
from datetime import datetime


def setup_logger(name, log_file=None, level=logging.INFO, 
                 format_string=None, stream=True):
    """
    Set up a logger with file and/or stream handlers.
    
    Parameters
    ----------
    name : str
        Logger name
    log_file : str, optional
        Path to log file
    level : int, optional
        Logging level
    format_string : str, optional
        Format string for log messages
    stream : bool, optional
        Whether to include a stream handler
        
    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Set default format if not provided
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    formatter = logging.Formatter(format_string)
    
    # Add file handler if log file is specified
    if log_file:
        # Create directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Add stream handler if requested
    if stream:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    
    return logger


def get_run_logger(run_name=None, output_dir=None):
    """
    Get a logger for a specific training/inference run.
    
    Parameters
    ----------
    run_name : str, optional
        Name for the run (default: timestamp)
    output_dir : str, optional
        Directory to store log files
        
    Returns
    -------
    logging.Logger
        Configured logger for the run
    """
    # Generate timestamp for run if not provided
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"run_{timestamp}"
    
    # Set up log file if output directory is provided
    log_file = None
    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        log_file = os.path.join(output_dir, f"{run_name}.log")
    
    # Create and return logger
    return setup_logger(
        name=f"luminara.{run_name}",
        log_file=log_file,
        level=logging.INFO
    )