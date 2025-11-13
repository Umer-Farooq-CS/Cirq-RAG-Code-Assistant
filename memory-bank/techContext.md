# Tech Context

Languages & Frameworks:
- Python 3.11+
- Cirq
- PyTorch 2.1+ (primary ML framework)
- PyTorch CUDA for GPU optimization
- Sentence Transformers for embeddings
- FAISS (or equivalent) for vector search

Dev setup:
- Linux Ubuntu 20.04+ (recommended)
- pip + venv
- Pre-commit hooks (black, isort, flake8, mypy optional)
- CUDA support for PyTorch CUDA

Constraints:
- Reproducibility; deterministic seeds for simulation
- GPU memory management for PyTorch

Future enhancement (post-project):
- FastAPI for QCanvas integration
- QCanvas integration requirements
