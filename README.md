# Advancing BERT: Tokenization Strategy Benchmarking

A research repository benchmarking the impact of five distinct tokenization strategies on BERT-based sentiment analysis using the [SST-2](https://huggingface.co/datasets/stanfordnlp/sst2) dataset. The project is governed by a Human-in-the-Loop (HITL) "Room of Agents" autonomous agent framework.

## Results at a Glance

| Tokenizer | Accuracy | F1 Score | Train Time | Parameters |
|---|---|---|---|---|
| Baseline (WordPiece) | **92.55%** | **92.74%** | 1377s | 109,483,778 |
| Dynamic (BPE-Dropout) | 80.28% | 81.30% | 1311s | 93,722,882 |
| BPE | 79.59% | 80.35% | 1331s | 97,562,882 |
| Hybrid (Word + Char) | 78.67% | 79.74% | 1293s | 89,882,882 |
| Character-level | 55.28% | 62.93% | 2432s | 86,097,410 |

> [!NOTE]
> The Baseline dominates because it retains BERT's original pre-trained embedding matrix. Among the custom tokenizers that require full embedding re-initialization, **Dynamic BPE-Dropout achieves the highest F1 (81.30%)** by acting as a structural regularizer across epochs.

See the [full research report](docs/final_report.md) for a complete mathematical analysis of these findings.

## Key Research Findings

- **Character-level tokenization** causes an $O(N^2)$ attention penalty by inflating 99th-percentile sequence lengths from ~46 to 194 tokens, doubling training time and crashing F1 to 62.9%.
- **BPE-Dropout** (p=0.1) stochastically fragments tokens into different subword sequences each epoch, acting as implicit regularization and outperforming all deterministic custom tokenizers.
- **Differential learning rates** (`1e-3` for new embeddings, `2e-5` for the pre-trained encoder) are essential to prevent catastrophic forgetting when resizing `nn.Embedding`.

## Project Structure

```
NLP_2/
├── src/
│   ├── tokenizers/          # Custom tokenizer implementations
│   │   ├── tokenizers_custom.py    # BPE tokenizer
│   │   ├── tokenizers_char.py      # Character-level tokenizer
│   │   ├── tokenizers_hybrid.py    # Hybrid Word+Char tokenizer
│   │   └── tokenizers_dynamic.py   # BPE-Dropout tokenizer
│   ├── training_engine.py   # Centralized training loop (DRY)
│   ├── model_utils.py       # BERT loading & embedding resizing
│   └── data_loader.py       # DataLoader factory
├── scripts/
│   ├── train_baseline.py    # WordPiece baseline runner
│   ├── train_bpe.py         # BPE runner
│   ├── train_char.py        # Character-level runner
│   ├── train_hybrid.py      # Hybrid runner
│   ├── train_dynamic.py     # Dynamic BPE-Dropout runner
│   └── download_data.py     # Dataset verification utility
├── docs/
│   ├── final_report.md      # Full research paper (6 sections)
│   ├── roadmap.md           # ML hyperparameter constraints
│   ├── objective.md         # North Star — project alignment doc
│   └── agent_tools_documentation.md  # Room of Agents catalog
├── results/                 # Training logs & generated plots
├── data/                    # Tokenizer configs (JSON)
└── environment.yml          # Conda environment spec
```

## Setup

**Prerequisites:** Anaconda/Miniconda, CUDA-compatible GPU (tested on NVIDIA RTX 4050 6GB).

```powershell
# Create and activate the environment
conda env create -f environment.yml
conda activate nlp

# Verify the SST-2 dataset is accessible
python scripts\download_data.py
```

> [!NOTE]
> The SST-2 dataset is streamed and cached automatically by Hugging Face at `~/.cache/huggingface/datasets`. The `data/` directory stores only the trained tokenizer JSON configs, not raw data.

## Running Experiments

Each tokenizer has an isolated, independent training script. Run any of them independently:

```powershell
conda activate nlp

# WordPiece Baseline (uses BERT's original tokenizer)
python scripts\train_baseline.py

# Custom tokenizer experiments
python scripts\train_bpe.py
python scripts\train_char.py
python scripts\train_hybrid.py
python scripts\train_dynamic.py
```

All scripts use `torch.amp.autocast` for Automatic Mixed Precision and log results to `results/train_output.txt`.

> [!IMPORTANT]
> Each training run takes approximately **20–40 minutes** on an RTX 4050. The character-level model takes significantly longer (~40 min) due to sequence length inflation.

## Autonomous Agent Framework

This repository is governed by a **Three-Tier Room of Agents** architecture designed for human-in-the-loop AI-assisted research:

| Tier | Agent | Role |
|---|---|---|
| Planning | `lead-architect` | Ingests objectives, generates `roadmap.md` + `active_sprint.md`, halts for human approval |
| Governance | `alignment-sentinel` | Silent middleware that validates all file writes and commands against `docs/objective.md` |
| Execution | `ml-scientist`, `core-engineer`, `data-architect` | Implement tasks only after Sentinel clearance |

> [!TIP]
> See [docs/agent_tools_documentation.md](docs/agent_tools_documentation.md) for a full catalog of agents, skills, and rules. The framework is fully decoupled from this NLP project and can be reused for any ML research repository.

## Documentation

| Document | Description |
|---|---|
| [docs/final_report.md](docs/final_report.md) | Full 6-section research paper with methodology, results, and deep analysis |
| [docs/roadmap.md](docs/roadmap.md) | Immutable ML hyperparameter bounds (LR, vocab sizes, padding limits) |
| [docs/objective.md](docs/objective.md) | North Star alignment document read by all agents before acting |
| [docs/agent_tools_documentation.md](docs/agent_tools_documentation.md) | Room of Agents capability catalog |