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
pip install -r requirements.txt
