"""Basic unit tests for Cirq-RAG-Code-Assistant."""

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

    def test_project_structure(self, project_root):
        """Test that project structure is correct."""
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

    def test_src_structure(self, src_path):
        """Test that src directory structure is correct."""
        # Check main source directories exist
        expected_dirs = ["rag", "agents", "orchestration", "evaluation", "cli"]
        for dir_name in expected_dirs:
            assert (src_path / dir_name).exists(), f"Missing directory: {dir_name}"


if __name__ == "__main__":
    pytest.main([__file__])
