# Installation Guide

## 🚀 Quick Start

The fastest way to get started with Cirq-RAG-Code-Assistant is to install it using pip:

```bash
pip install cirq-rag-code-assistant
```

## 📋 Prerequisites

### System Requirements
- **Operating System**: Linux Ubuntu 20.04+ (recommended)
- **Python**: 3.11 or higher
- **Memory**: 8GB RAM minimum, 16GB recommended
- **Storage**: 10GB free space
- **CPU**: Multi-core processor recommended
- **GPU**: NVIDIA GPU with CUDA support (for TensorFlow GPU optimization)

### Python Installation
If you don't have Python 3.11+ installed:

#### Linux Ubuntu
```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-pip
```

### GPU Setup (Optional but Recommended)
For TensorFlow GPU optimization:

#### Install NVIDIA Drivers
```bash
# Check if NVIDIA GPU is available
nvidia-smi

# Install NVIDIA drivers (if not already installed)
sudo apt update
sudo apt install nvidia-driver-525
```

#### Install CUDA Toolkit
```bash
# Install CUDA toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.0.0/local_installers/cuda-repo-ubuntu2004-12-0-local_12.0.0-525.60.13-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-12-0-local_12.0.0-525.60.13-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2004-12-0-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda
```

## 🔧 Installation Methods

### 1. Production Installation

#### From PyPI (Recommended)
```bash
pip install cirq-rag-code-assistant
```

#### From Source
```bash
git clone https://github.com/umerfarooq/cirq-rag-code-assistant.git
cd cirq-rag-code-assistant
pip install -e .
```

### 2. Development Installation

For development work, install with all development dependencies:

```bash
git clone https://github.com/umerfarooq/cirq-rag-code-assistant.git
cd cirq-rag-code-assistant
pip install -e ".[dev,gpu,quantum,qcanvas]"
```

### 3. Virtual Environment (Recommended)

#### Create Virtual Environment
```bash
# Create virtual environment
python -m venv cirq-rag-env

# Activate (Linux/macOS)
source cirq-rag-env/bin/activate

# Activate (Windows)
cirq-rag-env\Scripts\activate
```

#### Install in Virtual Environment
```bash
pip install cirq-rag-code-assistant
```

### 4. Using Poetry (Alternative)

If you prefer Poetry for dependency management:

```bash
# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install

# Activate environment
poetry shell
```

## 🎯 Installation Options

### Basic Installation
```bash
pip install cirq-rag-code-assistant
```

### With TensorFlow GPU Support
```bash
pip install cirq-rag-code-assistant[gpu]
```

### With Quantum Computing Extensions
```bash
pip install cirq-rag-code-assistant[quantum]
```

### With QCanvas Integration
```bash
pip install cirq-rag-code-assistant[qcanvas]
```

### Complete Development Installation
```bash
pip install cirq-rag-code-assistant[dev,gpu,quantum,qcanvas]
```

## 🔍 Verification

### Check Installation
```bash
# Check version
cirq-rag --version

# Test CLI
cirq-rag --help

# Test Python import
python -c "import cirq_rag_code_assistant; print('Installation successful!')"
```

### Run Basic Test
```bash
# Test basic functionality
python -c "
from cirq_rag_code_assistant import DesignerAgent
agent = DesignerAgent()
print('Designer Agent initialized successfully!')
"
```

## 🔧 QCanvas Integration Setup

### Verify QCanvas Integration
```bash
# Test QCanvas integration
python -c "from cirq_rag_code_assistant.integration import QCanvasClient; print('QCanvas integration ready!')"
```

### Start Development Server for QCanvas
```bash
# Start server for QCanvas integration
cirq-rag server --host 0.0.0.0 --port 8000
```

## 🔧 Configuration

### Environment Variables
Create a `.env` file in your project directory:

```bash
# API Configuration
OPENAI_API_KEY=your_openai_api_key_here
CIRQ_RAG_LOG_LEVEL=INFO
CIRQ_RAG_DEBUG=false

# Database Configuration (SQLite for development)
DATABASE_URL=sqlite:///./cirq_rag.db

# Vector Database
VECTOR_DB_PATH=./vector_db
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# QCanvas Integration
QCANVAS_HOST=localhost
QCANVAS_PORT=3000
QCANVAS_API_KEY=your_qcanvas_api_key

# GPU Configuration
TF_GPU_MEMORY_GROWTH=true
CUDA_VISIBLE_DEVICES=0

# Server Configuration
HOST=0.0.0.0
PORT=8000
```

### Configuration File
Create `config.yaml`:

```yaml
# Cirq-RAG-Code-Assistant Configuration
system:
  log_level: INFO
  debug: false
  workers: 4

api:
  host: "0.0.0.0"
  port: 8000
  cors_origins: ["*"]

agents:
  designer:
    max_retries: 3
    timeout: 30
  optimizer:
    optimization_level: "balanced"
  validator:
    simulation_timeout: 60
  educational:
    explanation_depth: "intermediate"

rag:
  vector_store:
    index_type: "hnsw"
    similarity_threshold: 0.7
  knowledge_base:
    update_interval: 3600

database:
  url: "sqlite:///./cirq_rag.db"
  echo: false

cache:
  redis_url: "redis://localhost:6379"
  ttl: 3600
```

## 🚀 First Run

### Initialize the System
```bash
# Initialize knowledge base
cirq-rag init

# Start the server
cirq-rag server

# Or use the CLI
cirq-rag generate "Create a simple VQE circuit"
```

### Web Interface
1. Start the server: `cirq-rag server`
2. Open browser: `http://localhost:8000`
3. Access API docs: `http://localhost:8000/docs`

## 🔧 Development Setup

### Clone Repository
```bash
git clone https://github.com/umerfarooq/cirq-rag-code-assistant.git
cd cirq-rag-code-assistant
```

### Install Development Dependencies
```bash
# Install with all development tools
pip install -e ".[dev,docs,gpu,quantum]"

# Install pre-commit hooks
pre-commit install
```

### Run Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=cirq_rag_code_assistant

# Run specific test categories
pytest -m "not slow"
pytest -m "gpu"
pytest -m "quantum"
```

### Code Quality
```bash
# Format code
black src/ tests/
isort src/ tests/

# Run linting
flake8 src/ tests/
mypy src/

# Run all quality checks
pre-commit run --all-files
```

## 🐛 Troubleshooting

### Common Issues

#### 1. Python Version Error
```
ERROR: Package requires Python >=3.11
```
**Solution**: Install Python 3.11 or higher

#### 2. Memory Issues
```
ERROR: Out of memory during installation
```
**Solution**: 
- Close other applications
- Use `--no-cache-dir` flag: `pip install --no-cache-dir cirq-rag-code-assistant`

#### 3. CUDA/GPU Issues
```
ERROR: CUDA not found
```
**Solution**: 
- Install CPU-only version: `pip install cirq-rag-code-assistant`
- Or install CUDA toolkit for GPU support

#### 4. Import Errors
```
ModuleNotFoundError: No module named 'cirq_rag_code_assistant'
```
**Solution**:
- Ensure virtual environment is activated
- Reinstall the package: `pip install -e .`

#### 5. Permission Errors
```
ERROR: Permission denied
```
**Solution**:
- Use virtual environment
- Or use `--user` flag: `pip install --user cirq-rag-code-assistant`

### Getting Help

#### Check Installation
```bash
# Verify Python version
python --version

# Check installed packages
pip list | grep cirq-rag

# Test import
python -c "import cirq_rag_code_assistant; print('OK')"
```

#### Debug Mode
```bash
# Enable debug logging
export CIRQ_RAG_DEBUG=true
export CIRQ_RAG_LOG_LEVEL=DEBUG

# Run with verbose output
cirq-rag --verbose generate "test"
```

#### Log Files
Check log files for detailed error information:
```bash
# View logs
tail -f logs/app.log
tail -f logs/error.log
```

## 📚 Next Steps

After successful installation:

1. **Read the Documentation**: Start with [Quick Start Guide](quickstart.md)
2. **Explore Examples**: Check [Usage Examples](examples/README.md)
3. **API Reference**: See [API Documentation](api/README.md)
4. **Join Community**: Visit our [GitHub Discussions](https://github.com/umerfarooq/cirq-rag-code-assistant/discussions)

## 🔄 Updates

### Update Installation
```bash
# Update to latest version
pip install --upgrade cirq-rag-code-assistant

# Update from source
git pull origin main
pip install -e .
```

### Uninstall
```bash
# Remove package
pip uninstall cirq-rag-code-assistant

# Remove virtual environment
rm -rf cirq-rag-env
```

---

*For more detailed information, see the [Architecture Guide](architecture.md) and [API Documentation](api/README.md).*
