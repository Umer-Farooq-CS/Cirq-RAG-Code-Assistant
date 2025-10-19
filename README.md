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
├─ memory-bank/                  # Project memory (read first on each task)
│  ├─ projectbrief.md            # Scope and goals
│  ├─ productContext.md          # Why it exists, UX goals
│  ├─ systemPatterns.md          # Architecture, patterns, decisions
│  ├─ techContext.md             # Tech stack, setup, constraints
│  ├─ activeContext.md           # Current focus and next steps
│  └─ progress.md                # Status and known issues
├─ docs/
│  └─ Proposal/                  # Research proposal (LaTeX + PDF)
├─ src/                          # Python source (to be implemented)
│  ├─ rag/                       # Retrieval pipeline, embeddings, vector store
│  ├─ agents/                    # Designer/Optimizer/Validator/Educational
│  ├─ orchestration/             # Agent coordination, tools, workflows
│  ├─ evaluation/                # Metrics, harness, datasets
│  └─ cli/                       # CLI entry points
├─ data/                         # Local datasets/knowledge base (ignored)
├─ outputs/                      # Artifacts, logs, reports (ignored)
├─ tests/                        # Unit/integration tests
├─ .cursorrules                  # Project intelligence (living doc)
├─ .gitignore
├─ LICENSE
└─ README.md
```

Create missing directories with:

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
