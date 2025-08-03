"""Setup configuration for shedboxai package."""

from setuptools import find_packages, setup

setup(
    name="shedboxai",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "pyyaml>=6.0.1",
        "pandas>=2.0.0",
        "requests>=2.31.0",
        "python-dotenv>=1.0.0",
        "pydantic>=2.0.0",
        "openai>=1.75.0",
        "networkx>=3.2.1",
        "jsonpath-ng>=1.7.0",
        "tenacity>=8.0.0",
        # Introspection feature dependencies
        "genson>=1.2.2",  # JSON schema generation
        "chardet>=5.0.0",  # Character encoding detection
    ],
    extras_require={
        "test": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "responses>=0.25.0",
        ],
        "dev": [
            # Testing
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "responses>=0.25.0",
            # Code formatting
            "black>=23.0.0",
            "isort>=5.12.0",
            # Linting
            "flake8>=6.0.0",
            # Security scanning
            "bandit>=1.7.5",
            "safety>=2.3.0",
            # Pre-commit hooks
            "pre-commit>=3.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "shedboxai=shedboxai.cli:main",
        ],
    },
    author="ShedBoxAI",
    description="A lightweight framework for building AI-powered applications through configuration",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/shedboxai/shedboxai",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
    python_requires=">=3.9",
)
