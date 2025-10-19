# Cirq-RAG-Code-Assistant

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)](https://tensorflow.org)
[![Cirq](https://img.shields.io/badge/Cirq-1.2+-green.svg)](https://quantumai.google/cirq)
[![License](https://img.shields.io/badge/License-Academic%20Use-red.svg)](LICENSE)

**A research-grade system for generating, optimizing, explaining, and validating Cirq quantum computing code using hybrid RAG + Multi-Agent architecture with TensorFlow GPU optimization.**

</div>

## 🎯 Overview

The Cirq-RAG-Code-Assistant is a cutting-edge research project that combines **Retrieval-Augmented Generation (RAG)** with **Multi-Agent Systems** to provide intelligent assistance for Google's Cirq quantum computing framework. Our system generates syntactically correct, executable Cirq code from natural language descriptions while providing comprehensive educational explanations.

### 🚀 Key Features

- **🧠 Hybrid RAG + Multi-Agent Architecture** - Combines knowledge retrieval with specialized agents
- **⚡ TensorFlow GPU Optimization** - Leverages GPU acceleration for performance
- **🎓 Educational Focus** - Provides step-by-step explanations alongside generated code
- **🔧 Tool-Augmented Reasoning** - Uses compile/simulate loops for code validation
- **🤖 Agentic Reinforcement Learning** - Iterative refinement using RL techniques
- **📊 Comprehensive Evaluation** - Rigorous testing and benchmarking framework

### 🏗️ System Architecture

Our system employs four specialized agents working in coordination:

- **🎨 Designer Agent** - Creates quantum circuits from natural language descriptions
- **⚡ Optimizer Agent** - Optimizes circuits for depth, gate count, and performance
- **✅ Validator Agent** - Validates code syntax, logic, and quantum principles
- **📚 Educational Agent** - Provides explanations and learning content

## 🛠️ Technology Stack

### Core Technologies
- **Python 3.11+** - Primary development language
- **TensorFlow 2.13+** - Deep learning and GPU optimization
- **Cirq 1.2+** - Quantum computing framework
- **Sentence Transformers** - Text embeddings and similarity search
- **FAISS/ChromaDB** - Vector database for knowledge retrieval

### Development Tools
- **pytest** - Testing framework
- **Black + isort** - Code formatting
- **mypy** - Type checking
- **pre-commit** - Code quality hooks
- **Make** - Development automation

## 📁 Repository Structure

```
.
├─ data/                                    # Data storage (git-ignored)
│  ├─ datasets/                            # Training and evaluation datasets
│  ├─ knowledge_base/                      # Curated Cirq code snippets
│  └─ models/                              # Pre-trained models and embeddings
│
├─ docs/                                   # Project documentation
│  ├─ agents/                              # Agent documentation
│  │  ├─ designer.md                       # Designer agent details
│  │  └─ README.md                         # Agent system overview
│  ├─ api/                                 # API documentation
│  │  └─ README.md                         # REST API reference
│  ├─ Proposal/                            # Research proposal
│  │  └─ LaTeX Files/                      # LaTeX source files
│  ├─ architecture.md                      # System architecture
│  ├─ installation.md                      # Setup instructions
│  ├─ integration.md                       # QCanvas integration guide
│  ├─ overview.md                          # Project overview
│  ├─ quickstart.md                        # Quick start guide
│  ├─ README.md                            # Documentation index
│  └─ tech-stack.md                        # Technology details
│
├─ memory-bank/                            # Project memory system
│  ├─ activeContext.md                     # Current focus and next steps
│  ├─ productContext.md                    # Product vision and UX goals
│  ├─ progress.md                          # Status and known issues
│  ├─ projectbrief.md                      # Scope and objectives
│  ├─ systemPatterns.md                    # Architecture patterns
│  └─ techContext.md                       # Technical context
│
├─ outputs/                                # Generated outputs (git-ignored)
│  ├─ artifacts/                           # Generated code and visualizations
│  ├─ logs/                                # System and application logs
│  └─ reports/                             # Evaluation reports and metrics
│
├─ src/                                    # Python source code
│  ├─ agents/                              # Multi-agent system
│  ├─ cirq_rag_code_assistant/             # Main package
│  │  └─ config/                           # Configuration modules
│  ├─ cli/                                 # Command-line interface
│  ├─ evaluation/                          # Evaluation framework
│  ├─ orchestration/                       # Agent coordination
│  └─ rag/                                 # RAG system implementation
│
├─ tests/                                  # Test suite
│  ├─ e2e/                                 # End-to-end tests
│  ├─ integration/                         # Integration tests
│  └─ unit/                                # Unit tests
│
├─ .cursorrules                            # Project intelligence
├─ .gitignore                              # Git ignore patterns
├─ .pre-commit-config.yaml                 # Pre-commit hooks
├─ CHANGELOG.md                            # Project changelog
├─ LICENSE                                 # Academic Use License
├─ Makefile                                # Development automation
├─ README.md                               # This file
├─ env.template                            # Environment variables template
├─ pyproject.toml                          # Modern Python packaging
├─ requirements.txt                        # Python dependencies
├─ setup-dev.bat                           # Windows batch setup script
├─ setup-dev.ps1                           # Windows PowerShell setup script
├─ setup-dev.sh                            # Linux setup script
└─ setup.py                                # Python package setup
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.11+** (Linux Ubuntu recommended for TensorFlow GPU)
- **NVIDIA GPU** (optional, for TensorFlow GPU acceleration)
- **CUDA 12.0+** (if using GPU)

### Installation

#### Option 1: Automated Setup (Recommended)

**Windows PowerShell:**
```powershell
.\setup-dev.ps1
```

**Windows Command Prompt:**
```cmd
setup-dev.bat
```

**Linux/Unix:**
```bash
chmod +x setup-dev.sh
./setup-dev.sh
```

#### Option 2: Manual Setup

1. **Create virtual environment:**
   ```bash
   python3.11 -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate     # Windows
   ```

2. **Install dependencies:**
   ```bash
   # Basic installation
   pip install -e .
   
   # Development installation with TensorFlow GPU
   pip install -e ".[dev,gpu,quantum]"
   ```

3. **Setup pre-commit hooks:**
   ```bash
   pre-commit install
   pre-commit install --hook-type pre-push
   ```

4. **Run tests:**
   ```bash
   pytest -q
   ```

### Environment Configuration

1. **Copy environment template:**
   ```bash
   cp env.template .env
   ```

2. **Edit `.env` file** with your configuration:
   ```env
   DEBUG=true
   ENVIRONMENT=development
   LOG_LEVEL=INFO
   DATABASE_URL=sqlite:///data/cirq_rag.db
   ```

## 🧪 Development

### Available Commands

```bash
# Development automation
make help              # Show all available commands
make test              # Run all tests
make lint              # Run linting and formatting
make format            # Format code with Black and isort
make clean             # Clean build artifacts

# Installation options
make install           # Basic installation
make install-dev       # Development installation
make install-gpu       # With TensorFlow GPU support
make install-quantum   # With quantum computing extensions
```

### Testing

```bash
# Run all tests
pytest

# Run specific test types
pytest tests/unit/                    # Unit tests only
pytest tests/integration/             # Integration tests only
pytest tests/e2e/                     # End-to-end tests only

# Run with coverage
pytest --cov=src --cov-report=html
```

### Code Quality

```bash
# Format code
black src/ tests/
isort src/ tests/

# Type checking
mypy src/

# Linting
flake8 src/ tests/

# Security checks
bandit -r src/
safety check
```

## 📊 Research Goals

### Primary Objectives

1. **Code Generation Accuracy** - Achieve >90% syntactically correct Cirq code
2. **Educational Value** - Provide comprehensive explanations and learning content
3. **Performance Optimization** - Leverage TensorFlow GPU for neural network acceleration
4. **Validation & Testing** - Ensure generated code works through simulation
5. **Reproducible Evaluation** - Establish benchmarks and evaluation metrics

### Target Algorithms

- **VQE (Variational Quantum Eigensolver)** - Quantum chemistry applications
- **QAOA (Quantum Approximate Optimization Algorithm)** - Combinatorial optimization
- **Quantum Teleportation** - Quantum communication protocols
- **Grover's Algorithm** - Quantum search algorithms
- **Quantum Fourier Transform** - Quantum signal processing

## 🔮 Future Enhancements

### Post-Project Development

- **QCanvas Integration** - Real-time circuit visualization and execution
- **Multi-Framework Support** - Extend to Qiskit, PennyLane, and other frameworks
- **Interactive Learning** - Personalized learning experiences
- **Cloud Deployment** - Scalable cloud-based quantum computing assistance

## 📚 Documentation

- **[Project Overview](docs/overview.md)** - Detailed project description and goals
- **[Architecture Guide](docs/architecture.md)** - System design and components
- **[Installation Guide](docs/installation.md)** - Comprehensive setup instructions
- **[Quick Start](docs/quickstart.md)** - Get up and running quickly
- **[Technology Stack](docs/tech-stack.md)** - Complete technology overview
- **[API Documentation](docs/api/README.md)** - REST API reference
- **[Agent Documentation](docs/agents/README.md)** - Multi-agent system details

## 🤝 Contributing

We welcome contributions! Please see our development guidelines:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Make your changes** and ensure tests pass
4. **Commit your changes** (`git commit -m 'Add amazing feature'`)
5. **Push to the branch** (`git push origin feature/amazing-feature`)
6. **Open a Pull Request**

### Development Standards

- Follow PEP 8 style guidelines
- Write comprehensive tests for new features
- Update documentation for any changes
- Ensure all pre-commit hooks pass
- Use conventional commit messages

## 📄 License

This project is licensed under the **Academic Use License** - see the [LICENSE](LICENSE) file for details.

**Key Points:**
- ✅ **Academic use** - Free for educational and research purposes
- ✅ **Open source** - Source code available for study and modification
- ❌ **Commercial use** - Requires explicit written permission
- 📧 **Contact** - umerfarooqcs0891@gmail.com for licensing inquiries

## 👥 Authors

- **Umer Farooq** (Team Lead) - umerfarooqcs0891@gmail.com
- **Hussain Waseem Syed** - i220893@nu.edu.pk  
- **Muhammad Irtaza Khan** - i220911@nu.edu.pk

## 📞 Support

- **Issues** - [GitHub Issues](https://github.com/Umer-Farooq-CS/Cirq-RAG-Code-Assistant/issues)
- **Email** - umerfarooqcs0891@gmail.com
- **Documentation** - [docs/](docs/) directory

## 🙏 Acknowledgments

- **Google Cirq Team** - For the excellent quantum computing framework
- **TensorFlow Team** - For GPU optimization capabilities
- **Open Source Community** - For the tools and libraries that made this possible

---

<div align="center">

**Made with ❤️ for the quantum computing community**

[⭐ Star this repo](https://github.com/Umer-Farooq-CS/Cirq-RAG-Code-Assistant) | [🐛 Report Bug](https://github.com/Umer-Farooq-CS/Cirq-RAG-Code-Assistant/issues) | [💡 Request Feature](https://github.com/Umer-Farooq-CS/Cirq-RAG-Code-Assistant/issues)

</div>