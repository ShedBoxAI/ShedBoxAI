# ShedBoxAI

[![CI](https://github.com/ShedBoxAI/ShedBoxAI/workflows/🧪%20Tests%20%26%20Quality/badge.svg)](https://github.com/ShedBoxAI/ShedBoxAI/actions/workflows/ci.yml)
[![Security](https://github.com/ShedBoxAI/ShedBoxAI/workflows/🔒%20Security%20Scan/badge.svg)](https://github.com/ShedBoxAI/ShedBoxAI/actions/workflows/security.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A powerful, enterprise-grade framework for building sophisticated AI-powered applications through pure configuration. No coding required!

## Overview

ShedBoxAI transforms complex data processing and AI workflows into simple YAML configurations. Build data analysis pipelines, AI-powered content generation, and business automation workflows without writing code.

**Key Features:**
- 🔧 **Configuration-First**: Define complex workflows in simple YAML
- 🤖 **AI Integration**: Built-in support for LLMs with batch processing
- 📊 **Data Processing**: 80+ built-in functions across 6 operation types
- 🔐 **Enterprise Ready**: Authentication, error handling, and retry logic
- ⚡ **High Performance**: Parallel execution and dependency-aware workflows

## Quick Start

1. **Install ShedBoxAI**
   ```bash
   pip install shedboxai
   ```

2. **Create a configuration file** (`config.yaml`)
   ```yaml
   data_sources:
     users:
       type: csv
       path: "users.csv"

   operations:
     - name: "process_users"
       type: transform
       source: "users"
       transformations:
         - field: "email"
           function: "lowercase"
   ```

3. **Run your workflow**
   ```bash
   shedboxai run config.yaml
   ```

## Documentation

📖 **[Full Documentation](https://shedboxai.com/)** - Complete guides, API reference, and examples

- [Getting Started](https://shedboxai.com/docs/getting-started/installation)
- [Configuration Guide](https://shedboxai.com/docs/configuration/data-sources)
- [Operations Reference](https://shedboxai.com/docs/operations/)
- [Examples](https://shedboxai.com/docs/examples)
- [CLI Reference](https://shedboxai.com/docs/cli-reference/run-command)

## Data Introspection

Analyze your data sources automatically:

```bash
# Generate comprehensive data documentation
shedboxai introspect sources.yaml

# Include sample data in the analysis
shedboxai introspect sources.yaml --include-samples
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License

ShedBoxAI is released under the MIT License. See [LICENSE](LICENSE) for details.

## Support

- [Documentation](https://shedboxai.github.io)
- [Issue Tracker](https://github.com/ShedBoxAI/ShedBoxAI/issues)
- [Discussions](https://github.com/ShedBoxAI/ShedBoxAI/discussions)
