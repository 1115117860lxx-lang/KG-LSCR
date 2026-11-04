# KG-LSCR: A Traffic Criminal Case Reasoning Method

This repository provides the core implementation and reproducibility instructions for the paper: **"基于知识图谱与法律三段式推理的交通刑事案情分析方法"**.

## 1. Overview
The **KG-LSCR** framework integrates Knowledge Graph constraints with Legal Syllogism Chain-of-Thought (LS-CoT) reasoning to improve the accuracy and interpretability of traffic-related criminal case analysis. The core pipeline includes:
- Tri-state factor extraction
- Topology-based backbone mapping for retrieval
- Dual-channel isolated retrieval (Statutes & Historical Cases)
- Legal syllogism reasoning generation

## 2. Environment Setup
Create a Python environment and install the required dependencies:
```bash
conda create -n legal_reasoning python=3.8
conda activate legal_reasoning
pip install torch faiss-cpu openai sentence-transformers
```

## 3. Dataset Privacy & Reproducibility
Due to strict legal constraints and privacy policies regarding real-world criminal case records (CAIL2018), **we do not provide the raw, sensitive corpus or absolute file paths**. 

To demonstrate the reproducibility of the proposed algorithm without leaking sensitive information, we provide a synthetic, desensitized toy example within `main.py`.

## 4. Quick Start
Run the minimal viable product (MVP) pipeline to simulate the extraction and reasoning process:
```bash
python main.py
```

## 5. Main Results
Experiments on the manually reviewed gold-standard evaluation set demonstrate that KG-LSCR achieves:
- **Macro-F1 (Charge Determination):** 93.6%
- **MAP (Statutory Provision Recommendation):** 89.2%
- **Factual Hallucination Rate:** Decreased to 3.2%
