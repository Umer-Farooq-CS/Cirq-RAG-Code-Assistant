# Cirq-RAG-Code-Assistant

Production-ready, research-oriented assistant for generating, optimizing, explaining, and validating Cirq quantum computing code using a hybrid RAG + Multi-Agent architecture with tool augmentation and agentic RL.

## Highlights
- RAG over a curated Cirq knowledge base
- Multi-agent orchestration: Designer, Optimizer, Validator, Educational
- Tool-augmented compile/simulate loop, agentic RL for refinement
- Reproducible evaluation and educational focus

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

1) Python environment

```bash
py -3.11 -m venv .venv
. .venv/Scripts/Activate.ps1   # PowerShell on Windows
pip install -U pip
```

2) Install base deps (placeholder; will be expanded as code lands)

```bash
pip install cirq
```

3) Run checks (placeholder)

```bash
pytest -q
```

## Roadmap (High-level)
- RAG scaffolding: embeddings, vector store, retrieval API
- Agent interfaces and basic implementations
- Orchestration flows with compile/simulate tools
- Evaluation harness and datasets
- CLI for end-to-end prompts → Cirq code + explanations

See `memory-bank/activeContext.md` and `memory-bank/progress.md` for current focus and status.

## License

MIT — see `LICENSE`.
