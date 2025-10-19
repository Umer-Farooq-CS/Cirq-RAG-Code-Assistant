# Cirq-RAG-Code-Assistant

Research-oriented assistant for generating, optimizing, explaining, and validating Cirq quantum computing code using a hybrid RAG + Multi-Agent architecture with tool augmentation and Agentic RL.

## Highlights
- RAG over a curated Cirq knowledge base
- Multi-agent orchestration: Designer, Optimizer, Validator, Educational
- Tool-augmented compile/simulate loop, agentic RL for refinement
- TensorFlow GPU optimization for performance
- Reproducible evaluation and educational focus

## Future Enhancement (Post-Project)
- QCanvas integration for real-time circuit visualization and execution

## Repository Structure

```
.
├─ data/                         # Local datasets/knowledge base (ignored)
│  ├─ datasets/                  # Training and evaluation datasets
│  ├─ knowledge_base/            # Curated Cirq code snippets
│  └─ models/                    # Pre-trained models and embeddings
├─ docs/                         # Project documentation
│  ├─ agents/                    # Agent documentation
│  │  ├─ designer.md             # Designer agent documentation
│  │  └─ README.md               # Agent overview
│  ├─ api/                       # API documentation
│  │  └─ README.md               # API reference
│  ├─ Proposal/                  # Research proposal (LaTeX + PDF)
│  │  └─ LaTeX Files/            # LaTeX source files
│  ├─ architecture.md            # System architecture
│  ├─ installation.md            # Installation guide
│  ├─ integration.md             # QCanvas integration guide
│  ├─ overview.md                # Project overview
│  ├─ quickstart.md              # Quick start guide
│  ├─ README.md                  # Documentation index
│  └─ tech-stack.md              # Technology stack
├─ memory-bank/                  # Project memory (read first on each task)
│  ├─ activeContext.md           # Current focus and next steps
│  ├─ productContext.md          # Why it exists, UX goals
│  ├─ progress.md                # Status and known issues
│  ├─ projectbrief.md            # Scope and goals
│  ├─ systemPatterns.md          # Architecture, patterns, decisions
│  └─ techContext.md             # Tech stack, setup, constraints
├─ outputs/                      # Artifacts, logs, reports (ignored)
│  ├─ artifacts/                 # Generated code and visualizations
│  ├─ logs/                      # System and application logs
│  └─ reports/                   # Evaluation reports and metrics
├─ src/                          # Python source (to be implemented)
│  ├─ agents/                    # Designer/Optimizer/Validator/Educational
│  ├─ cirq_rag_code_assistant/   # Main package
│  │  └─ config/                 # Configuration modules
│  ├─ cli/                       # CLI entry points
│  ├─ evaluation/                # Metrics, harness, datasets
│  ├─ orchestration/             # Agent coordination, tools, workflows
│  └─ rag/                       # Retrieval pipeline, embeddings, vector store
├─ tests/                        # Unit/integration tests
│  ├─ e2e/                      # End-to-end tests
│  ├─ integration/              # Integration tests
│  └─ unit/                     # Unit tests
├─ .cursorrules                  # Project intelligence (living doc)
├─ .gitignore                    # Git ignore patterns
├─ .pre-commit-config.yaml       # Pre-commit hooks configuration
├─ CHANGELOG.md                  # Project changelog
├─ LICENSE                       # Academic Use License
├─ Makefile                      # Development automation
├─ README.md                     # This file
├─ env.template                  # Environment variables template
├─ pyproject.toml                # Modern Python packaging
├─ requirements.txt              # Python dependencies
├─ setup-dev.bat                 # Windows batch setup script
├─ setup-dev.ps1                 # Windows PowerShell setup script
├─ setup-dev.sh                  # Linux setup script
└─ setup.py                      # Python package setup
```

Create missing directories with:

**Windows PowerShell:**
```powershell
New-Item -ItemType Directory -Path "src\rag", "src\agents", "src\orchestration", "src\evaluation", "src\cli", "tests", "data", "outputs" -Force
```

**Windows Command Prompt:**
```cmd
mkdir src\rag src\agents src\orchestration src\evaluation src\cli tests data outputs
```

**Linux/Unix:**
```bash
mkdir -p src/{rag,agents,orchestration,evaluation,cli} tests data outputs
```

## Quickstart

1) Python environment (Linux Ubuntu recommended)

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
```

2) Install dependencies

```bash
# Basic installation
pip install -e .

# Development installation with TensorFlow GPU
pip install -e ".[dev,gpu,quantum]"
```

3) Run checks

```bash
pytest -q
```

## Roadmap (High-level)
- RAG scaffolding: embeddings, vector store, retrieval API
- Agent interfaces and basic implementations
- Orchestration flows with compile/simulate tools
- TensorFlow GPU optimization for neural networks
- Evaluation harness and datasets
- CLI for end-to-end prompts → Cirq code + explanations

## Future Enhancement (Post-Project)
- QCanvas integration API and real-time visualization

See `memory-bank/activeContext.md` and `memory-bank/progress.md` for current focus and status.

## Authors

- **Umer Farooq** (Team Lead) - umerfarooqcs0891@gmail.com
- **Hussain Waseem Syed** - i220893@nu.edu.pk  
- **Muhammad Irtaza Khan** - i220911@nu.edu.pk

## License

Academic Use License — see `LICENSE`. Commercial use requires explicit permission.
