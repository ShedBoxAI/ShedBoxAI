# Contributing to ShedBoxAI

Thank you for your interest in contributing to ShedBoxAI! This document provides guidelines and instructions for setting up your development environment and contributing to the project.

## Development Setup

### Prerequisites

- Python 3.9 or higher
- Git

### Setting Up Your Development Environment

1. **Clone the repository**
   ```bash
   git clone https://github.com/shedboxai/shedboxai.git
   cd shedboxai
   ```

2. **Create and activate a virtual environment**
   ```bash
   # Create virtual environment
   python -m venv ../shedgpt-python-env

   # Activate virtual environment
   source ../shedgpt-python-env/bin/activate  # Linux/Mac
   # or
   ..\shedgpt-python-env\Scripts\activate     # Windows
   ```

3. **Install the package in development mode**
   ```bash
   # Install with all development dependencies
   pip install -e ".[dev]"
   ```

4. **Set up pre-commit hooks** (recommended)
   ```bash
   pre-commit install
   ```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test categories
pytest tests/ -m unit          # Unit tests only
pytest tests/ -m integration   # Integration tests only
pytest tests/ -m slow          # Slow running tests

# Run tests with coverage report
pytest --cov=shedboxai tests/

# Run single test function
pytest tests/unit/test_pipeline.py::test_pipeline_run
```

### Code Quality and Formatting

We use several tools to maintain code quality:

- **Black** for code formatting
- **isort** for import sorting
- **flake8** for linting
- **bandit** for security scanning

#### Manual Code Quality Checks

```bash
# Format code with black
black .

# Sort imports with isort
isort .

# Run linting with flake8
flake8 .

# Run security scan with bandit
bandit -r shedboxai/
```

#### Pre-commit Hooks

Pre-commit hooks will automatically run these checks before each commit:

```bash
# Install pre-commit hooks (one-time setup)
pre-commit install

# Run pre-commit hooks manually on all files
pre-commit run --all-files
```

### Running the Application

```bash
# Run with a configuration file
shedboxai run examples/sample_config.yaml

# Run with verbose logging
shedboxai run examples/sample_config.yaml --verbose

# Run with custom output file
shedboxai run examples/sample_config.yaml --output results.json
```

## Contributing Guidelines

### Branch Naming Convention

- **feature/**: New features (`feature/add-csv-support`)
- **bugfix/**: Bug fixes (`bugfix/fix-json-parsing`)
- **hotfix/**: Critical fixes (`hotfix/security-patch`)
- **docs/**: Documentation updates (`docs/update-readme`)

### Commit Message Format

Use clear, descriptive commit messages:

```
type(scope): brief description

Longer description if needed, explaining what and why.

- List specific changes if helpful
- Reference issues: Fixes #123
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

### Pull Request Process

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write tests for new functionality
   - Ensure all tests pass
   - Follow code style guidelines

3. **Run quality checks**
   ```bash
   # Run all tests
   pytest tests/

   # Check code formatting
   black --check .
   isort --check-only .
   flake8 .

   # Run security scan
   bandit -r shedboxai/
   ```

4. **Create a pull request**
   - Provide a clear description of changes
   - Reference any related issues
   - Ensure CI checks pass

### Testing Guidelines

- Write unit tests for new functionality
- Use appropriate test markers (`@pytest.mark.unit`, `@pytest.mark.integration`)
- Mock external dependencies in unit tests
- Ensure test coverage remains above 80%

### Code Style Guidelines

- Follow PEP 8 style guide
- Use type hints where appropriate
- Write docstrings for public functions and classes
- Keep functions small and focused
- Use meaningful variable and function names

### Security Considerations

- Never commit secrets, API keys, or sensitive data
- Use environment variables for configuration
- Run security scans before submitting PRs
- Follow secure coding practices

## Development Workflow

### Adding New Operations

1. Create a new handler in `shedboxai/core/operations/`
2. Extend `OperationHandler` base class
3. Add configuration model to `shedboxai/core/config/models.py`
4. Add normalizer function to `shedboxai/core/config/normalizers.py`
5. Write comprehensive tests in `tests/unit/core/test_operations/`

### Adding New Data Source Types

1. Extend connector logic in `shedboxai/connector.py`
2. Add configuration validation
3. Write tests for the new connector
4. Update documentation

## Environment Variables

Set these environment variables for testing:

```bash
# Enable test mode (mocks AI operations)
export SHEDBOXAI_TEST_MODE=1

# AI API keys (for integration tests)
export OPENAI_API_KEY=your_openai_key
```

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure you've installed the package with `pip install -e ".[dev]"`
2. **Test failures**: Check that `SHEDBOXAI_TEST_MODE=1` is set
3. **Pre-commit hook failures**: Run `pre-commit run --all-files` to fix formatting issues

### Getting Help

- Check existing [Issues](https://github.com/shedboxai/shedboxai/issues)
- Review the [Documentation](docs/)
- Ask questions in [Discussions](https://github.com/shedboxai/shedboxai/discussions)

## License

By contributing to ShedBoxAI, you agree that your contributions will be licensed under the project's BSD 3-Clause License.

Thank you for contributing to ShedBoxAI! 🚀
