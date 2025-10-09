"""Setup configuration for shedboxai package."""

from setuptools import find_packages, setup

setup(
    name="shedboxai",
    version="1.0.1",
    packages=find_packages(),
    package_data={
        "shedboxai": ["data/AI_ASSISTANT_GUIDE.md"],
    },
    include_package_data=True,
    install_requires=[
        "pyyaml>=6.0.1",
        "pandas>=2.0.0",
        "requests>=2.31.0",
        "python-dotenv>=1.0.1",
        "pydantic>=2.0.0",
        "openai>=1.75.0",
        "networkx>=3.2.1",
        "jsonpath-ng>=1.7.0",
        "tenacity>=8.0.0",
        "jinja2>=3.0.0",
        "rich>=13.0.0",  # For CLI interface
        # Introspection feature dependencies
        "genson>=1.2.2",  # JSON schema generation
        "chardet>=5.0.0",  # Character encoding detection
    ],
    extras_require={
        "test": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "responses>=0.25.0",
            "flask>=3.0.0",  # For E2E test server
        ],
        "dev": [
            # Testing
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "responses>=0.25.0",
            "flask>=3.0.0",  # For E2E test server
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
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
)
