"""Logging configuration for ome_zarr_pyramid."""

import logging


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for the given module name.
    
    Args:
        name: Module name (typically __name__)
    
    Returns:
        logging.Logger: The logger; ``logging.basicConfig(level=logging.INFO)`` shows its lines
    """
    # a library leaves output to the application: no handler of its own (with one, an
    # application that logs printed every line twice, and plain Python printed INFO)
    logger = logging.getLogger(name)
    logger.addHandler(logging.NullHandler())
    return logger
