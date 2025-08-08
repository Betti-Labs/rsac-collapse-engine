# Contributing to RSAC

Thank you for your interest in contributing to RSAC! This document provides guidelines for contributing to the project.

## 🚀 Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/rsac-collapse-engine.git
   cd rsac-collapse-engine
   ```
3. **Set up development environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   pip install -e .  # Install in development mode
   ```

## 🛠️ Development Workflow

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following our coding standards

3. **Run tests** to ensure everything works:
   ```bash
   python -m pytest tests/
   python stress_test.py --correctness --cases 10
   ```

4. **Format your code**:
   ```bash
   black src/ tests/
   flake8 src/ tests/
   ```

5. **Commit your changes**:
   ```bash
   git add .
   git commit -m "Add: brief description of your changes"
   ```

6. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Create a Pull Request** on GitHub

## 📝 Coding Standards

### Python Style
- Follow [PEP 8](https://pep8.org/) style guidelines
- Use [Black](https://black.readthedocs.io/) for code formatting
- Maximum line length: 88 characters
- Use type hints where appropriate

### Documentation
- Document all public functions and classes
- Include docstrings with parameter descriptions
- Add inline comments for complex algorithms
- Update README.md if adding new features

### Testing
- Write tests for new functionality
- Maintain >90% test coverage
- Include both unit tests and integration tests
- Test edge cases and error conditions

## 🎯 Types of Contributions

### 🐛 Bug Reports
When reporting bugs, please include:
- Python version and operating system
- Steps to reproduce the issue
- Expected vs actual behavior
- Error messages and stack traces
- Minimal code example if possible

### ✨ Feature Requests
For new features, please:
- Check if the feature already exists
- Describe the use case and motivation
- Provide examples of how it would be used
- Consider backward compatibility

### 🔧 Code Contributions

#### High-Priority Areas
- **Performance optimizations**: Faster signature generation, GPU acceleration
- **New signature methods**: Alternative reduction rules, hybrid approaches
- **SAT solver integration**: CDCL integration, preprocessing techniques
- **Benchmarking**: More test cases, real-world instances
- **Documentation**: Tutorials, examples, API documentation

#### Algorithm Improvements
- New signature generation methods
- Improved bucket ordering strategies
- Hybrid search techniques
- Parallel processing optimizations

#### Infrastructure
- Better error handling and logging
- Configuration management
- Continuous integration setup
- Performance profiling tools

## 🧪 Testing Guidelines

### Running Tests
```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=src/rsac --cov-report=html

# Run stress tests
python stress_test.py --all

# Run benchmarks
python src/benchmarks.py --series sat --ns 8 10 --instances 5
```

### Writing Tests
- Place tests in the `tests/` directory
- Use descriptive test names: `test_signature_generation_with_empty_input`
- Test both success and failure cases
- Mock external dependencies when appropriate

## 📊 Benchmarking

When adding new features, please:
- Run existing benchmarks to ensure no regressions
- Add new benchmarks for new functionality
- Include performance comparisons in pull requests
- Document any significant performance changes

## 🔍 Code Review Process

1. **Automated checks** must pass (tests, linting, formatting)
2. **Manual review** by maintainers
3. **Performance validation** for algorithm changes
4. **Documentation review** for user-facing changes

### Review Criteria
- Code quality and readability
- Test coverage and quality
- Performance impact
- Documentation completeness
- Backward compatibility

## 🏷️ Commit Message Guidelines

Use clear, descriptive commit messages:

```
Type: Brief description (50 chars max)

Longer explanation if needed (wrap at 72 chars)

- Bullet points for multiple changes
- Reference issues: Fixes #123
```

**Types:**
- `Add:` New features
- `Fix:` Bug fixes
- `Update:` Changes to existing features
- `Remove:` Removing features
- `Docs:` Documentation changes
- `Test:` Adding or updating tests
- `Refactor:` Code restructuring

## 🚀 Release Process

1. **Version bumping** follows [Semantic Versioning](https://semver.org/)
2. **Changelog** updated with all changes
3. **Tests** must pass on all supported Python versions
4. **Documentation** updated as needed
5. **Performance benchmarks** validated

## 📞 Getting Help

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Email**: gregory@betti-labs.com for direct contact

## 🙏 Recognition

Contributors will be:
- Listed in the project's contributors section
- Acknowledged in release notes
- Credited in academic papers (for significant contributions)

## 📜 License

By contributing to RSAC, you agree that your contributions will be licensed under the Apache License 2.0.

---

Thank you for helping make RSAC better! 🎉