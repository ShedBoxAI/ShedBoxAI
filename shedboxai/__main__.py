"""
ShedBoxAI CLI entry point.

This module is invoked when running `python -m shedboxai`.
It suppresses warnings before loading any other modules.
"""

# Disable warnings before any other imports
import warnings

warnings.filterwarnings("ignore")

# Now import and run the CLI
from .cli import main

if __name__ == "__main__":
    main()
