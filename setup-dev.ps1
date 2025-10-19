# Cirq-RAG-Code-Assistant Development Environment Setup Script
# For Windows with TensorFlow GPU support

param(
    [string]$PythonVersion = "3.11",
    [switch]$SkipGPUCheck = $false,
    [switch]$SkipTests = $false
)

# Set error action preference
$ErrorActionPreference = "Stop"

Write-Host "🚀 Setting up Cirq-RAG-Code-Assistant development environment..." -ForegroundColor Blue

# Function to print colored output
function Write-Status {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Blue
}

function Write-Success {
    param([string]$Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

# Check Python version
Write-Status "Checking Python version..."
try {
    $pythonVersion = python --version 2>&1
    if ($pythonVersion -match "Python (\d+\.\d+)") {
        $version = $matches[1]
        $requiredVersion = [version]$PythonVersion
        $currentVersion = [version]$version
        
        if ($currentVersion -lt $requiredVersion) {
            Write-Error "Python $PythonVersion+ is required. Found: $version"
            Write-Status "Please install Python $PythonVersion+ and try again."
            exit 1
        }
        Write-Success "Python version check passed: $version"
    } else {
        Write-Error "Could not determine Python version"
        exit 1
    }
} catch {
    Write-Error "Python not found. Please install Python $PythonVersion+ and try again."
    exit 1
}

# Check for NVIDIA GPU and CUDA
if (-not $SkipGPUCheck) {
    Write-Status "Checking for NVIDIA GPU and CUDA..."
    try {
        $nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
        if ($nvidiaSmi) {
            Write-Success "NVIDIA GPU detected:"
            nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader,nounits
        } else {
            Write-Warning "NVIDIA GPU not detected. TensorFlow will run on CPU."
        }
    } catch {
        Write-Warning "Could not check for NVIDIA GPU. TensorFlow will run on CPU."
    }
}

# Create virtual environment
Write-Status "Creating virtual environment..."
if (-not (Test-Path "venv")) {
    python -m venv venv
    Write-Success "Virtual environment created"
} else {
    Write-Warning "Virtual environment already exists"
}

# Activate virtual environment
Write-Status "Activating virtual environment..."
& ".\venv\Scripts\Activate.ps1"

# Upgrade pip
Write-Status "Upgrading pip..."
python -m pip install --upgrade pip

# Install development dependencies
Write-Status "Installing development dependencies..."
pip install -e ".[dev,gpu,quantum]"

# Install pre-commit hooks
Write-Status "Installing pre-commit hooks..."
pre-commit install
pre-commit install --hook-type pre-push

# Create necessary directories
Write-Status "Creating project directories..."
$directories = @(
    "data\knowledge_base",
    "data\datasets", 
    "data\models",
    "outputs\logs",
    "outputs\reports",
    "outputs\artifacts",
    "tests\unit",
    "tests\integration", 
    "tests\e2e",
    "src\rag",
    "src\agents",
    "src\orchestration",
    "src\evaluation",
    "src\cli"
)

foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
}

# Create initial configuration files
Write-Status "Creating initial configuration files..."

# Create .env file from template
if (-not (Test-Path ".env")) {
    if (Test-Path "env.template") {
        Copy-Item "env.template" ".env"
        Write-Success "Created .env file from template"
    } else {
        Write-Warning "env.template not found, creating basic .env file"
        @"
# Cirq-RAG-Code-Assistant Environment Configuration
DEBUG=true
ENVIRONMENT=development
LOG_LEVEL=INFO
LOG_FILE=outputs/logs/cirq_rag.log
DATABASE_URL=sqlite:///data/cirq_rag.db
"@ | Out-File -FilePath ".env" -Encoding UTF8
    }
} else {
    Write-Warning ".env file already exists"
}

# Test TensorFlow GPU installation
Write-Status "Testing TensorFlow GPU installation..."
$tensorflowTest = @"
import tensorflow as tf
print(f'TensorFlow version: {tf.__version__}')
print(f'GPU available: {tf.config.list_physical_devices("GPU")}')
if tf.config.list_physical_devices('GPU'):
    print('✅ TensorFlow GPU support is working!')
else:
    print('⚠️  TensorFlow GPU support not available - will use CPU')
"@

$tensorflowTest | python

# Create basic test file if it doesn't exist
if (-not (Test-Path "tests\unit\test_basic.py")) {
    Write-Status "Creating basic test file..."
    $testContent = @'
"""Basic tests for Cirq-RAG-Code-Assistant."""

import pytest
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


class TestBasicFunctionality:
    """Test basic functionality and imports."""

    def test_python_version(self):
        """Test that we're using Python 3.11+."""
        assert sys.version_info >= (3, 11)

    def test_imports(self):
        """Test that basic imports work."""
        try:
            import tensorflow as tf
            import cirq
            import numpy as np
            assert True
        except ImportError as e:
            pytest.fail(f"Failed to import required packages: {e}")

    def test_tensorflow_gpu(self):
        """Test TensorFlow GPU availability."""
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        # This test passes whether GPU is available or not
        assert isinstance(gpus, list)

    def test_cirq_basic(self):
        """Test basic Cirq functionality."""
        import cirq
        
        # Create a simple circuit
        qubits = cirq.LineQubit.range(2)
        circuit = cirq.Circuit()
        circuit.append(cirq.H(qubits[0]))
        circuit.append(cirq.CNOT(qubits[0], qubits[1]))
        
        # Verify circuit has gates
        assert len(circuit) == 2
        assert len(list(circuit.all_qubits())) == 2

    def test_numpy_basic(self):
        """Test basic NumPy functionality."""
        import numpy as np
        
        # Test basic operations
        arr = np.array([1, 2, 3, 4])
        assert arr.sum() == 10
        assert arr.mean() == 2.5

    @pytest.mark.gpu
    def test_tensorflow_gpu_operations(self):
        """Test TensorFlow GPU operations if available."""
        import tensorflow as tf
        
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            # Test basic GPU operation
            with tf.device('/GPU:0'):
                a = tf.constant([1.0, 2.0, 3.0])
                b = tf.constant([4.0, 5.0, 6.0])
                c = tf.add(a, b)
                assert tf.reduce_sum(c).numpy() == 21.0
        else:
            pytest.skip("GPU not available")

    def test_project_structure(self):
        """Test that project structure is correct."""
        project_root = Path(__file__).parent.parent.parent
        
        # Check main directories exist
        assert (project_root / "src").exists()
        assert (project_root / "tests").exists()
        assert (project_root / "docs").exists()
        assert (project_root / "memory-bank").exists()
        
        # Check main files exist
        assert (project_root / "README.md").exists()
        assert (project_root / "requirements.txt").exists()
        assert (project_root / "pyproject.toml").exists()
        assert (project_root / "LICENSE").exists()

    def test_src_structure(self):
        """Test that src directory structure is correct."""
        project_root = Path(__file__).parent.parent.parent
        src_path = project_root / "src"
        
        # Check main source directories exist
        expected_dirs = ["rag", "agents", "orchestration", "evaluation", "cli"]
        for dir_name in expected_dirs:
            assert (src_path / dir_name).exists(), f"Missing directory: {dir_name}"


if __name__ == "__main__":
    pytest.main([__file__])
'@
    
    $testContent | Out-File -FilePath "tests\unit\test_basic.py" -Encoding UTF8
    Write-Success "Created basic test file"
}

# Run the basic tests
if (-not $SkipTests) {
    Write-Status "Running basic tests..."
    python -m pytest tests\unit\test_basic.py -v
}

# Final setup summary
Write-Success "🎉 Development environment setup complete!"
Write-Host ""
Write-Status "Next steps:"
Write-Host "  1. Activate the virtual environment: .\venv\Scripts\Activate.ps1"
Write-Host "  2. Review and update .env file with your configuration"
Write-Host "  3. Start implementing the RAG system: make dev-start"
Write-Host "  4. Run tests: make test"
Write-Host "  5. Check code quality: make lint"
Write-Host ""
Write-Status "Useful commands:"
Write-Host "  - make help          # Show all available commands"
Write-Host "  - make test          # Run all tests"
Write-Host "  - make lint          # Run linting"
Write-Host "  - make format        # Format code"
Write-Host "  - make clean         # Clean build artifacts"
Write-Host ""
Write-Success "Happy coding! 🚀"
