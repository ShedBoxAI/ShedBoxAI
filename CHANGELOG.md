# Changelog

All notable changes to ShedBoxAI will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- LLM-friendly introspection output with ShedBoxAI YAML syntax reference section
- Actionable `link_fields` YAML in relationship detection output
- Per-source field access patterns showing direct and joined field access
- Comprehensive documentation for `item.{target}_info.{field}` pattern in AI_ASSISTANT_GUIDE.md
- New test suite for markdown generator (19 tests)

### Fixed
- Field access patterns now correctly extract column names from schema_info
- Duplicate "After joining" lines in introspection output
- Pandas deprecation warnings in CSV analyzer (infer_datetime_format, to_datetime)

## [1.0.3] - 2025-11-30

### Fixed
- Error collection with severity levels (Issues #7.3, #7.4)

## [1.0.2] - 2025-11-29

### Fixed
- Derived fields and nested group_by issues

### Changed
- Improved AI_ASSISTANT_GUIDE.md for better LLM config generation

## [1.0.1] - 2025-10-09

### Fixed
- Minor bug fixes and improvements

## [1.0.0] - 2025-08-31

### Added
- Initial release of ShedBoxAI
- YAML-based configuration for data processing pipelines
- Support for CSV, JSON, YAML, REST API, and text data sources
- Six operation types: contextual_filtering, format_conversion, content_summarization, relationship_highlighting, advanced_operations, template_matching
- Data introspection system with automatic relationship detection
- AI integration for LLM-powered analysis
- Graph-based execution engine for complex workflows
- CLI interface with `run` and `introspect` commands

[Unreleased]: https://github.com/ShedBoxAI/ShedBoxAI/compare/v1.0.3...HEAD
[1.0.3]: https://github.com/ShedBoxAI/ShedBoxAI/compare/v1.0.2...v1.0.3
[1.0.2]: https://github.com/ShedBoxAI/ShedBoxAI/compare/v1.0.1...v1.0.2
[1.0.1]: https://github.com/ShedBoxAI/ShedBoxAI/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/ShedBoxAI/ShedBoxAI/releases/tag/v1.0.0
