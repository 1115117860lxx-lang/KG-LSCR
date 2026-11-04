# KG-LSCR: A Traffic Criminal Case Reasoning Method

This repository provides the core implementation and a runnable demonstration of **KG-LSCR**, a knowledge-graph-constrained legal reasoning framework for traffic criminal case analysis.

The corresponding paper is:

> **基于知识图谱与法律三段式推理的交通刑事案情分析方法**

KG-LSCR combines tri-state fact extraction, local case-graph construction, topology-based query mapping, dual-channel retrieval, and legal syllogism reasoning to improve the factual consistency, legal-retrieval accuracy, and interpretability of traffic criminal case analysis.

---

## 1. Overview

The KG-LSCR pipeline consists of four main components:

1. **Tri-state Factor Extraction (TFE)**  
   Extracts legally relevant factual elements using three semantic states:

   - Explicitly present
   - Explicitly absent
   - Insufficient information

   The module also applies subject–behavior attribution constraints to reduce factual misattribution between defendants, victims, and other participants.

2. **Topology-Based Backbone Mapping for Retrieval (TBM-SR)**  
   Constructs a local case graph containing the core parties, legally relevant facts, behaviors, and related elements. Topology-based pruning is then applied to reduce irrelevant background information and generate a compact legal query.

3. **Dual-Channel Isolated Retrieval-Augmented Generation (DC-IRAG)**  
   Separates statutory knowledge from historical-case knowledge at the retrieval-channel level. Statutory provisions are used as normative legal references, while historical cases provide empirical case-level references.

4. **Legal Syllogism Chain-of-Thought Reasoning (LS-CoT)**  
   Organizes the reasoning process into four stages:

   - Norm identification
   - Fact confirmation
   - Legal subsumption
   - Conclusion generation

The overall workflow is:

```text
Case Text
   │
   ▼
Tri-state Factor Extraction
   │
   ▼
Local Case Graph Construction
   │
   ▼
Topology-based Backbone Mapping
   │
   ├── Statutory Knowledge Retrieval
   │
   └── Historical Case Retrieval
   │
   ▼
Legal Syllogism Reasoning
   │
   ▼
Charge, Statutory Provision, and Sentencing Analysis
