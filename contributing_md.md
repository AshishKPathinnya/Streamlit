# Contributing to ML Model Deployment Hub

We love your input! We want to make contributing to this project as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## Development Process

We use GitHub to host code, to track issues and feature requests, as well as accept pull requests.

### Pull Request Process

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. Issue that pull request!

## Development Setup

### Prerequisites
- Python 3.8 or higher
- Git

### Local Development

1. **Clone your fork locally**:
```bash
git clone https://github.com/yourusername/ml-model-deployment-hub.git
cd ml-model-deployment-hub
```

2. **Create a virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
pip install -e .  # Install in development mode
```

4. **Create a feature branch**:
```bash
git checkout -b feature/amazing-feature
```

5. **Make your changes and test**:
```bash
streamlit run app.py
```

## Code Style

We use Python's PEP 8 style guide. Key points:

- Use 4 spaces for indentation
- Line length should be under 88 characters
- Use meaningful variable and function names
- Add docstrings to functions and classes

### Code Formatting
We recommend using `black` for code formatting:

```bash
pip install black
black app.py
```

## Testing

Before submitting a PR, make sure:

1. **The app runs without errors**:
```bash
streamlit run app.py
```

2. **All features work as expected**:
   - Test prediction functionality
   - Verify all visualizations render
   - Check model performance metrics
   - Test data export features

3. **Code follows style guidelines**

## Bug Reports

We use GitHub issues to track public bugs. Report a bug by opening a new issue.

**Great Bug Reports** tend to have:

- A quick summary and/or background
- Steps to reproduce
  - Be specific!
  - Give sample code if you can
- What you expected would happen
- What actually happens
- Notes (possibly including why you think this might be happening, or stuff you tried that didn't work)

### Bug Report Template

```markdown
## Bug Description
A clear and concise description of what the bug is.

## Steps to Reproduce
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

## Expected Behavior
A clear description of what you expected to happen.

## Actual Behavior
A clear description of what actually happened.

## Screenshots
If applicable, add screenshots to help explain your problem.

## Environment
- OS: [e.g. Windows 10, macOS 11.0, Ubuntu 20.04]
- Python Version: [e.g. 3.9.0]
- Streamlit Version: [e.g. 1.47.0]
- Browser: [e.g. Chrome 96, Safari 15]

## Additional Context
Add any other context about the problem here.
```

## Feature Requests

We also use GitHub issues to track feature requests.

**Great Feature Requests** include:

- A clear description of the problem this feature would solve
- A description of the desired solution
- Any alternative solutions considered
- Additional context or screenshots

### Feature Request Template

```markdown
## Feature Description
A clear and concise description of the feature you'd like to see.

## Problem Statement
What problem does this feature solve? Why is it needed?

## Proposed Solution
A clear description of how you envision this feature working.

## Alternatives Considered
Any alternative solutions or features you've considered.

## Additional Context
Add any other context, mockups, or examples about the feature request.
```

## Types of Contributions

### 🐛 Bug Fixes
- Fix broken functionality
- Resolve UI/UX issues
- Address performance problems

### ✨ New Features
- Add new model types
- Implement additional visualizations
- Create new dataset support
- Enhance user interface

### 📚 Documentation
- Improve README
- Add code comments
- Create tutorials or examples
- Update API documentation

### 🎨 UI/UX Improvements
- Enhance visual design
- Improve user experience
- Add accessibility features
- Optimize responsive design

### ⚡ Performance
- Optimize loading times
- Improve memory usage
- Enhance caching strategies
- Reduce computational complexity

## Code Organization

```
ml-model-deployment-hub/
├── app.py              # Main application file
├── utils/              # Utility functions (if created)
├── models/             # Model definitions (if created)
├── data/               # Data handling (if created)
├── tests/              # Test files (if created)
└── docs/               # Documentation
```

## Commit Message Guidelines

Use clear, descriptive commit messages:

- Use the present tense ("Add feature" not "Added feature")
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit the first line to 72 characters or less
- Reference issues and pull requests liberally after the first line

### Examples:
```
Add confusion matrix visualization

- Implement interactive plotly confusion matrix
- Add hover tooltips showing actual vs predicted
- Fix issue #123 with matrix display
```

## Review Process

1. **Automated Checks**: PRs must pass all automated checks
2. **Code Review**: At least one maintainer will review your code
3. **Testing**: Ensure all functionality works as expected
4. **Documentation**: Update relevant documentation

## Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes for significant contributions
- GitHub contributor insights

## Questions?

Feel free to open an issue with the "question" label or reach out to the maintainers directly.

## License

By contributing, you agree that your contributions will be licensed under the same MIT License that covers the project.