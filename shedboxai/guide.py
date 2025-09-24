"""
Utility functions for accessing the AI Assistant Guide.

This module provides functions to access the bundled AI_ASSISTANT_GUIDE.md
file that contains comprehensive ShedBoxAI configuration reference.
"""

import importlib.resources
from pathlib import Path
from typing import Optional


def get_guide_content() -> str:
    """
    Get the content of the AI Assistant Guide.

    Returns:
        str: The full content of the AI_ASSISTANT_GUIDE.md file

    Raises:
        FileNotFoundError: If the guide file cannot be found
        IOError: If the guide file cannot be read
    """
    try:
        # Try to read from package data first using modern approach
        try:
            files = importlib.resources.files('shedboxai.data')
            guide_file = files / 'AI_ASSISTANT_GUIDE.md'
            content = guide_file.read_text(encoding='utf-8')
            return content
        except (ImportError, AttributeError):
            # Fallback for older Python versions
            with importlib.resources.open_text('shedboxai.data', 'AI_ASSISTANT_GUIDE.md', encoding='utf-8') as f:
                return f.read()
    except Exception:
        # Fallback: try to read from relative path (development mode)
        try:
            guide_path = Path(__file__).parent / 'data' / 'AI_ASSISTANT_GUIDE.md'
            with open(guide_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            raise FileNotFoundError(
                "Could not find AI_ASSISTANT_GUIDE.md. "
                "This file should be bundled with the package."
            ) from e


def save_guide_to_file(file_path: str) -> None:
    """
    Save the AI Assistant Guide to a specified file.

    Args:
        file_path: Path where the guide should be saved

    Raises:
        FileNotFoundError: If the guide content cannot be retrieved
        IOError: If the file cannot be written
    """
    content = get_guide_content()

    # Ensure parent directory exists
    output_path = Path(file_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)




def print_guide_info() -> None:
    """Print information about the AI Assistant Guide."""
    print("📖 ShedBoxAI AI Assistant Guide")
    print("=" * 50)
    print("This guide transforms any LLM into a ShedBoxAI configuration expert.")
    print()
    print("Available commands:")
    print("  shedboxai guide              # Display the full guide")
    print("  shedboxai guide --save FILE  # Save guide to a file")
    print()
    print("The guide includes:")
    print("• Complete YAML configuration reference")
    print("• All 6 operation types with 80+ functions")
    print("• Data source configuration examples")
    print("• AI/LLM integration patterns")
    print("• Best practices and error handling")