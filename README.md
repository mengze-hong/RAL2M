# RAL2M: Retrieval-Augmented Learning-to-Match  
**Against Hallucination in Compliance-Guaranteed Service Systems**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

This repository contains the implementation of **RAL2M**, a novel framework that completely eliminates **generation hallucination** in LLM-driven service systems by using LLMs only for query-response matching in a retrieval-based pipeline. Judgment hallucination is mitigated via a **query-adaptive latent ensemble** that models inter-LLM dependencies and competences.


## Project Structure

```text
RAL2M/
├── data/                       # Raw and processed datasets
├── src/
│   ├── 1_SQG/                  # Similar-question generation & GPT-based labeling
│   │   ├── generate_gpt.py
│   │   └── ...
│   ├── 2_Retrieval/            # Retrieval with BGE embeddings
│   │   ├── bge_embed.py
│   │   ├── retrieve_top1.py
│   │   └── ...
│   ├── 3_LLM/                  # LLM judgment pipelines (5 models + debate + data augmentation)
│   │   ├── gpt_judge.py
│   │   ├── qwen_judge.py
│   │   ├── prompt_template.txt
│   │   ├── prompt_tuning/
│   │   └── ...
│   ├── 4_Ensemble/             # Ensemble methods: latent graphical model, neural model baseline
│   │   ├── graphical_model.py
│   │   ├── neural_model.py
│   │   ├── ablations_*.py
│   │   └── ...
│   └── 5_Analysis/             # Result analysis and figures
│       ├── analysis.py
│       └── figures/
└── README.md
```

## Key Features

- **Zero generation hallucination** — all responses are retrieved from a verified knowledge base
- **Query-adaptive latent ensemble** — captures LLM-specific competences and inter-model dependencies
- **Large-scale benchmark** — 82k+ queries across 5 public QA datasets

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Download models (BGE, Qwen2.5-7B, Llama-3.1-8B, etc.)
./llm_download.sh

# Run full pipeline
python src/4_Ensemble/graphical_model.py
```

## Main Results (from paper)

| Method               | Accuracy | Hallucination Rate | Precision | F1    |
|----------------------|----------|---------------------|-----------|-------|
| BGE + Threshold      | 53.1%    | 38.3%               | 0.45      | 0.43  |
| Majority Voting      | 60.2%    | 49.2%               | 0.53      | 0.61  |
| Neural Aggregator    | 64.2%    | 32.5%               | 0.58      | 0.59  |
| **RAL2M (Ours)**     | **70.7%**| **13.9%**           | **0.73**  | **0.63** |