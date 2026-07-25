# Contributing to DeepRAG

Thank you for your interest in contributing to DeepRAG! This document provides guidelines and instructions for contributing.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)

---

## 📜 Code of Conduct

This project adheres to a Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11 or higher
- Git
- Docker (optional, for containerized development)

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/deep-rag.git
   cd deep-rag
   ```

3. Add upstream remote:
   ```bash
   git remote add upstream https://github.com/Alelasi/deep-rag.git
   ```

---

## 💻 Development Setup

### 1. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
# Install all dependencies including dev tools
pip install -e ".[dev,llm,api,qdrant,pgvector,reranker]"
```

### 3. Set Up Environment Variables

```bash
cp .env.example .env
# Edit .env and add your API keys
```

### 4. Run Tests

```bash
pytest tests/ -v
```

---

## 🤝 How to Contribute

### Types of Contributions

We welcome the following types of contributions:

1. **Bug Fixes** - Fix issues reported in GitHub Issues
2. **New Features** - Add new functionality (discuss first in an issue)
3. **Documentation** - Improve or add documentation
4. **Tests** - Add or improve test coverage
5. **Performance** - Optimize existing code
6. **Refactoring** - Improve code quality without changing functionality

### Before You Start

1. **Check existing issues** - Someone might already be working on it
2. **Create an issue** - For new features, create an issue to discuss first
3. **Get assignment** - Wait for maintainer approval before starting work

---

## 📝 Coding Standards

### Python Style Guide

We follow [PEP 8](https://pep8.org/) with some modifications:

- **Line length**: 120 characters (not 79)
- **Imports**: Use `isort` for import sorting
- **Formatting**: Use `black` for code formatting
- **Type hints**: Use type annotations for all functions

### Code Formatting

Before committing, run:

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Check linting
flake8 src/ tests/
```

### Naming Conventions

- **Functions/Variables**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private methods**: `_leading_underscore`

### Documentation

- **Docstrings**: Use Google-style docstrings
- **Comments**: Explain "why", not "what"
- **Type hints**: Required for all public functions

Example:

```python
def search_documents(query: str, top_k: int = 10) -> List[Dict[str, Any]]:
    """Search for relevant documents using hybrid retrieval.

    Args:
        query: The search query string
        top_k: Number of results to return (default: 10)

    Returns:
        List of document dictionaries with scores

    Raises:
        ValueError: If query is empty
    """
    if not query:
        raise ValueError("Query cannot be empty")
    # Implementation...
```

---

## 🧪 Testing Guidelines

### Test Structure

- **Unit tests**: Test individual functions/classes
- **Integration tests**: Test component interactions
- **End-to-end tests**: Test complete workflows

### Writing Tests

1. **Location**: Place tests in `tests/` directory
2. **Naming**: Test files should start with `test_`
3. **Coverage**: Aim for >80% code coverage
4. **Fixtures**: Use pytest fixtures for common setup

Example:

```python
import pytest
from src.retrieval.hybrid import HybridRetriever

@pytest.fixture
def retriever():
    """Create a test retriever instance."""
    return HybridRetriever(collection_name="test")

def test_search_returns_results(retriever):
    """Test that search returns non-empty results."""
    results = retriever.search("test query", top_k=5)
    assert len(results) > 0
    assert all("score" in r for r in results)
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_hybrid.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run only fast tests (skip slow integration tests)
pytest tests/ -m "not slow"
```

---

## 🔄 Pull Request Process

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-number-description
```

Branch naming:
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation changes
- `refactor/` - Code refactoring
- `test/` - Test additions/changes

### 2. Make Changes

- Write clean, well-documented code
- Add tests for new functionality
- Update documentation if needed
- Follow coding standards

### 3. Commit Changes

Use clear, descriptive commit messages:

```bash
git commit -m "feat: add pgvector support for enterprise deployments

- Implement PgvectorRetriever class
- Add HNSW index optimization
- Include 12 unit tests
- Update documentation

Closes #123"
```

Commit message format:
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `test:` - Test changes
- `refactor:` - Code refactoring
- `perf:` - Performance improvements
- `chore:` - Maintenance tasks

### 4. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub with:
- **Clear title** - Summarize the change
- **Description** - Explain what and why
- **Related issues** - Link to related issues
- **Screenshots** - If UI changes
- **Checklist** - Complete the PR template checklist

### 5. Code Review

- Address reviewer feedback promptly
- Keep discussions professional and constructive
- Update your PR based on feedback
- Request re-review after changes

### 6. Merge

Once approved:
- Maintainer will merge your PR
- Your branch will be deleted automatically
- Celebrate! 🎉

---

## 🐛 Issue Reporting

### Before Creating an Issue

1. **Search existing issues** - Check if already reported
2. **Check documentation** - Might be a usage question
3. **Try latest version** - Bug might be fixed

### Creating a Good Issue

Use the appropriate issue template:

**Bug Report:**
- Clear title
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, etc.)
- Error messages/logs
- Minimal reproducible example

**Feature Request:**
- Clear description of the feature
- Use case / motivation
- Proposed solution (optional)
- Alternatives considered (optional)

---

## 📚 Additional Resources

- [Project README](README.md)
- [Architecture Documentation](docs/architecture.md)
- [API Documentation](http://localhost:8000/docs)
- [GitHub Issues](https://github.com/Alelasi/deep-rag/issues)

---

## 🙏 Thank You!

Your contributions make DeepRAG better for everyone. We appreciate your time and effort!

---

**Questions?** Feel free to ask in GitHub Discussions or create an issue.
