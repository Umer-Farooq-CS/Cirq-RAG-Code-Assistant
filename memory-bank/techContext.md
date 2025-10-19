# Tech Context

Languages & Frameworks:
- Python 3.11+
- Cirq
- TensorFlow 2.13+ (primary ML framework)
- TensorFlow GPU for optimization
- Sentence Transformers for embeddings
- FAISS (or equivalent) for vector search

Dev setup:
- Linux Ubuntu 20.04+ (recommended)
- pip + venv
- Pre-commit hooks (black, isort, flake8, mypy optional)
- CUDA support for TensorFlow GPU

Constraints:
- Reproducibility; deterministic seeds for simulation
- GPU memory management for TensorFlow

Future enhancement (post-project):
- FastAPI for QCanvas integration
- QCanvas integration requirements
