
# L.A.R.A - LLMs as Aerogel Research Assistants

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.30+-orange.svg)](https://huggingface.co/transformers/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

L.A.R.A (LLMs as Aerogel Research Assistants) is a specialized AI framework for  aerogel research, combining fine-tuned language models with knowledge graphs for hypothesis generation and inverse design of aerogel materials.


This repository contains **Module 2** of a comprehensive AI-driven aerogel research system. This module focuses on fine-tuned language model development and deployment for advanced materials science applications, specifically targeting aerogel synthesis, characterization, and design optimization.

> **Note**: This is part of a larger research framework. **Module 1** (containing RAG queries, simulation tool calling, and other computational modules) is **not included in this repository** as it contains proprietary intellectual property and trade secrets.


## 🚀 Features

- **🧠 Fine-tuned LLaMat Model**: Specialized for aerogel synthesis and materials science
- **🔬 Hypothesis Generation**: AI-powered scientific hypothesis generation for aerogel research
- **🎯 Inverse Design**: Design synthesis routes from target material properties
- **🧪 Molecular Dynamics Integration**: Run moleecular simulations for materials discovery
- **📚 Literature Search**: RAG-based search through scientific papers
- **🖼️ Image Analysis**: Particle segmentation and radius analysis
- **🗃️ Knowledge Graph**: Enhanced MatKG interface for materials science

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Fine-tuning](#fine-tuning)
- [Usage Examples](#usage-examples)
- [Architecture](#architecture)
- [Scripts Overview](#scripts-overview)
- [HPC Usage](#hpc-usage)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

## 🔧 Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- 16GB+ RAM for model inference
- 32GB+ VRAM for fine-tuning

### Environment Setup

```bash
# Clone the repository
git clone https://github.com/username/lara-aerogel-research.git
cd lara-aerogel-research

# Create conda environment
conda create -n lara python=3.9
conda activate lara

# Install PyTorch (CUDA 11.8)
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia

# Install requirements
pip install -r requirements.txt

# Install additional dependencies for tools
pip install ase lammps opencv-python scikit-image matplotlib seaborn
pip install sentence-transformers faiss-cpu nltk rank-bm25
pip install transformers[torch] accelerate bitsandbytes peft trl
```

### Model Setup

```bash
# Set environment variables
export KG_MODEL_PATH="/path/to/your/llamat_finetuned_complete"
export KG_VERBOSE="1"
export CUDA_VISIBLE_DEVICES="0"
```

## 🚀 Quick Start

### Interactive Chat Mode

```bash
python run_chat.py
```

### One-shot Queries

```bash
# Synthesis question
python run_chat.py "How do I synthesize carbon aerogels with high porosity?"

# Hypothesis generation
python run_chat.py "Generate hypotheses for improving aerogel thermal insulation"

# Inverse design
python run_chat.py "Design an aerogel with electrical conductivity > 100 S/m"
```

### Advanced Interface

```bash
python finetuned_llamat_hyp_gen_and_inv_design.py
```

## 🎓 Fine-tuning

### Prepare Your Dataset

Create a JSON file with instruction-output pairs:

```json
[
  {
    "instruction": "How do I synthesize carbon aerogels with high surface area?",
    "output": "To synthesize carbon aerogels with high surface area, use the sol-gel method with resorcinol-formaldehyde precursors..."
  }
]
```

### Run Fine-tuning

```bash
# On local machine
python finetune_llamat.py

# On HPC cluster
sbatch --gres=gpu:1 --time=8:00:00 --wrap="python finetune_llamat.py"
```

### Fine-tuning Parameters

- **Model**: `m3rg-iitd/llamat-2` (base LLaMat model)
- **Method**: LoRA (Low-Rank Adaptation)
- **Batch Size**: 1 per device, 8 gradient accumulation steps
- **Learning Rate**: 2e-5
- **Epochs**: 3
- **Max Sequence Length**: 1024 tokens

## 📖 Usage Examples

### 1. Hypothesis Generation

```python
from finetuned_llamat_hyp_gen_and_inv_design import LARAAdvancedSystem

system = LARAAdvancedSystem()
results = system.generate_hypotheses(
    "How does pyrolysis temperature affect electrical conductivity?",
    num_hypotheses=5
)

for result in results:
    print(f"Hypothesis: {result.hypothesis}")
    print(f"Confidence: {result.confidence:.2f}")
```

### 2. Inverse Design

```python
result = system.perform_inverse_design(
    "high porosity > 0.95 and electrical conductivity > 50 S/m"
)

print(f"Synthesis Routes: {len(result.synthesis_routes)}")
print(f"L.A.R.A Insights: {result.design_rationale}")
```

### 3. Interactive Tools

```python
from src.langchain_adapter import run_with_langchain_agent

response = run_with_langchain_agent(
    prompt="Analyze this microscopy image for particle sizes",
    model="local:microsoft/Phi-3-mini-4k-instruct",
    max_rounds=3
)
```

## 🏗️ Architecture

```
L.A.R.A Framework
├── Fine-tuned LLaMat Model (Aerogel Specialist)
├── Knowledge Graph Interface (MatKG)
├── Hypothesis Generation Engine
├── Inverse Design Engine
├── Multi-Agent Tool System
│   ├── Literature Search (RAG)
│   ├── Image Analysis
│   ├── Molecular Dynamics
│   └── Knowledge Generation
└── Interactive Interface
```

### Core Components

1. **Fine-tuned LLaMat**: Specialized language model for aerogel research
2. **MatKG Interface**: Enhanced materials knowledge graph
3. **Hypothesis Generator**: Template-based and LLM-guided hypothesis generation
4. **Inverse Design Engine**: Property-to-synthesis route mapping
5. **Multi-Agent System**: Specialized tools for different research tasks

## 📁 Scripts Overview

| Script | Description | Usage |
|--------|-------------|--------|
| `run_chat.py` | Interactive chat interface | `python run_chat.py` |
| `finetune_llamat.py` | Fine-tune LLaMat for aerogels | `python finetune_llamat.py` |
| `finetuned_llamat_hyp_gen_and_inv_design.py` | Advanced research interface | `python finetuned_llamat_hyp_gen_and_inv_design.py` |
| `src/langchain_adapter.py` | Multi-agent orchestration | Internal |
| `tools/knowledge_generation.py` | L.A.R.A model inference | Tool |
| `tools/rag_search.py` | Literature search | Tool |
| `tools/radius_segmentation.py` | Image analysis | Tool |
| `tools/simulation_tool.py` | simulation runner | Tool |

### Tool System

L.A.R.A includes specialized tools for different research tasks:

- **Knowledge Generation**: Use fine-tuned L.A.R.A model for synthesis advice
- **RAG Search**: Search scientific literature (arXiv, local papers)
- **Image Segmentation**: Analyze microscopy images for particle sizing
- **Simulation**: Run molecular dynamics with LAMMPS/AMS/GULP
- **Registry System**: Automatic tool discovery and orchestration

## 🖥️ HPC Usage

### SLURM Job Scripts

```bash
#!/bin/bash
#SBATCH --job-name=lara-finetune
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

module load python/3.9 pytorch cuda/11.8

export HF_HOME=/scratch/username/huggingface_cache
export KG_MODEL_PATH="/path/to/llamat_finetuned_complete"

python finetune_llamat.py
```

### Memory Optimization

- Use 4-bit quantization with BitsAndBytesConfig
- Gradient accumulation for large effective batch sizes
- LoRA for parameter-efficient fine-tuning
- Automatic mixed precision (bf16)

## 🔬 Research Applications

### Supported Research Areas

- ** Aerogel Synthesis**: Sol-gel processes, pyrolysis optimization
- **Property Optimization**: Porosity, conductivity, surface area
- **Inverse Design**: Target property → synthesis route mapping
- **Hypothesis Generation**: Novel research directions and mechanisms
- **Literature Analysis**: Automated paper search and summarization
- **Materials Characterization**: Image analysis and property prediction

### Example Research Workflows

1. **Literature Review**: Use RAG search to find relevant papers
2. **Hypothesis Generation**: Generate research questions using L.A.R.A
3. **Synthesis Planning**: Design routes using inverse design engine
4. **Simulation**: Run simulations to validate approaches
5. **Analysis**: Analyze experimental results with image tools

## 📊 Performance Metrics

### Model Performance
- **Fine-tuning Dataset**: 1,000+  aerogel research examples
- **Training Time**: ~4-8 hours on V100 GPU
- **Model Size**: 7B parameters (LoRA adapters: ~100MB)
- **Inference Speed**: ~2-5 tokens/second on RTX 3090

### Research Capabilities
- **Hypothesis Quality**: Expert-validated scientific relevance
- **Synthesis Route Accuracy**: Based on peer-reviewed literature
- **Literature Coverage**: 10,000+ aerogel research papers
- **Tool Integration**: 6 specialized research tools

## 🤝 Contributing

We welcome contributions to L.A.R.A! Here's how you can help:

### Areas for Contribution

1. **Dataset Expansion**: Add more aerogel research data
2. **Tool Development**: Create new research tools
3. **Model Improvements**: Enhance fine-tuning approaches
4. **Documentation**: Improve guides and examples
5. **Testing**: Add unit tests and integration tests

### Development Setup

```bash
# Fork and clone the repository
git clone https://github.com/username/lara-aerogel-research.git
cd lara-aerogel-research

# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests
python -m pytest tests/
```

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.


## 🙏 Acknowledgments

- **LLaMat Team**: For the base materials science language model
- **Hugging Face**: For the transformers library and model hosting
- **Materials Knowledge Graph**: For enhanced materials data
- **Carbon Aerogel Community**: For research insights and validation

## 📞 Support

- **Email**: sugan.kanagasenthinathan@dlr.de, prakul.pandit@dlr.de, hemangi.patel@dlr.de

## 🗓️ Changelog

### v1.0.0 (2024-09-13)
- Initial release of L.A.R.A framework
- Fine-tuned LLaMat model for carbon aerogels
- Hypothesis generation and inverse design capabilities
- Multi-agent tool system integration
- Interactive chat interface
- HPC-compatible training scripts

---

**Made with ❤️ for the materials science community**

Copyright 2025 Sugan Kanagasenthinathan, Prakul Pandit, Hemangi Patel
German Aerospace Center, Cologne

