# ClinGraph-RAG

**A Hybrid Knowledge Graph and Patient Similarity Framework for Grounded Clinical Question Answering over Electronic Health Records**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://python.org)
[![MIMIC-III](https://img.shields.io/badge/Data-MIMIC--III--10K-teal)](https://www.kaggle.com/datasets/bilal1907/mimic-iii-10k/data)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![HuggingFace](https://img.shields.io/badge/🤗%20Try%20it-HuggingFace%20Spaces-orange)](https://huggingface.co/spaces/YOUR_SPACE_LINK_HERE)
[![Medium](https://img.shields.io/badge/Read-Medium%20Article-black?logo=medium)](https://medium.com/YOUR_ARTICLE_LINK_HERE)

---

## Overview

Electronic Health Records (EHRs) contain decades of accumulated clinical knowledge, but that knowledge is fragmented, siloed, and inaccessible for real-time reasoning. Large language models offer a natural-language interface to this data, but when queried without retrieval grounding, they hallucinate.

We demonstrate this empirically: **BioMistral-7B-DARE achieves a faithfulness score of 0.000** when generating patient-specific summaries without retrieval.

**ClinGraph-RAG** addresses this with a three-component hybrid RAG framework evaluated on the full diabetes mellitus cohort from MIMIC-III-10K (2,732 admissions, 1,885 patients):

| Component | Description |
|---|---|
| **Clinical Knowledge Graph** | 8,129 nodes · 525,281 edges · 16 edge types including NER-derived symptom and temporal relationships |
| **KNN Physiological Retriever** | 32-dimensional feature vectors over first-24-hour lab trajectories, medication flags, and severity proxies |
| **FAISS Semantic Index** | 454,430 text chunks encoded with Sentence-BERT |

---

## Key Results

| Retrieval Mode | Specificity | Completeness | Faithfulness | Composite |
|---|---|---|---|---|
| LLM Only (baseline) | 0.500 | 0.400 | 0.000 ❌ | 0.300 |
| Text RAG | 0.407 | 0.330 | 0.553 | 0.430 |
| KNN Only | 0.690 | 0.345 | 0.287 | 0.441 |
| **KNN+Text (Ours)** | **0.480** | **0.308** | **0.584** ✅ | **0.458** |

KNN physiological similarity is independently validated by:
- **LOS Pearson r = 0.685** (p < 0.0001) — physiologically similar patients share genuine clinical trajectories
- **Random Forest Mortality AUROC = 0.791 ± 0.020** — on 547 held-out test admissions

---

## System Architecture

```
MIMIC-III (10K patients)
        │
        ▼
Diabetes Cohort (2,732 admissions, ICD-9 250.xx)
        │
   ┌────┴──────────────────────────────────┐
   │                                        │
   ▼                                        ▼
Structured Data                       Clinical Notes
   │                                        │
   ├── KNN Retriever                  ├── scispaCy NER
   │   (32-dim feature vectors)       │   (symptom entities)
   │                                  │
   ├── Knowledge Graph ───────────────┘
   │   ├── Summary Layer (patient-level facts)
   │   ├── Concept Layer (normalized labels)
   │   └── Population Layer (statistical relationships)
   │
   └── FAISS Index (454,430 chunks, Sentence-BERT)
        │
        ▼
   RAG Pipeline (4 ablation modes)
        │
        ▼
BioMistral-7B-DARE → Grounded Clinical Answers
        │
        ▼
   Evaluation (Specificity · Completeness · Faithfulness)
```

---

## Try It

### 🤗 HuggingFace Space

A lightweight demo is available on HuggingFace Spaces using **Flan-T5-base** as the generation model (instead of BioMistral-7B-DARE used in the paper, due to compute constraints).

👉 **[Try the demo](https://huggingface.co/spaces/Shambhavi111/clingraph-rag)**

> **Note:** The HuggingFace demo uses Flan-T5-base for accessibility. The full paper results use BioMistral-7B-DARE in 4-bit NF4 quantization. Retrieval quality and faithfulness behavior will differ accordingly.

### 📖 Medium Article

For a full walkthrough of the methodology, design decisions, and findings:

👉 **[Read on Medium](https://medium.com/@shambhaviadhikari/we-asked-an-ai-to-summarise-a-real-patients-medical-records-every-number-was-made-up-dc1dcd6a7a61)**

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/clingraph-rag.git
cd clingraph-rag
pip install -r requirements.txt
```

### Requirements

```
torch
transformers
sentence-transformers
faiss-cpu
networkx
scikit-learn
pandas
numpy
scispacy
https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_md-0.5.1.tar.gz
matplotlib
bitsandbytes  # for 4-bit quantization
```

---

## Data Setup

This project uses the [MIMIC-III-10K dataset](https://www.kaggle.com/datasets/bilal1907/mimic-iii-10k/data), a freely available 10,000-patient subset of MIMIC-III.

1. Download the dataset from Kaggle
2. Place the CSV files in `data/mimic-iii-10k/`
3. Run the cohort extraction script:

```bash
python scripts/01_cohort_extraction.py
```

> **Important:** MIMIC-III requires credentialed access. Follow the [PhysioNet data use agreement](https://physionet.org/content/mimiciii/1.4/) before using real MIMIC data. The 10K Kaggle subset is a de-identified public release.

---

## Knowledge Graph Structure

The merged clinical knowledge graph contains:

| Node Type | Count | Edge Type | Count |
|---|---|---|---|
| Admission | 2,732 | has_lab | 199,694 |
| Patient | 1,885 | has_lab_state | 192,124 |
| Drug | 1,137 | prescribed | 70,216 |
| Lab state concept | 642 | has_symptom | 41,879 |
| Symptom concept | 616 | has_outcome | 8,196 |
| Lab test | 562 | precedes | 3,776 |
| ... | ... | ... | ... |
| **Total** | **8,129** | **Total** | **525,281** |

The graph combines three layers:
- **Summary layer** — per-patient clinical facts (admissions, diagnoses, labs, prescriptions)
- **Concept layer** — normalized labels (e.g., Humulin + Novolog → Insulin)
- **Population layer** — statistically derived edges across all 2,732 admissions (min. 15 admissions, OR > 1.5, p < 0.05)

---

## RAG Pipeline Modes

The pipeline supports four ablation modes:

| Mode | Context Sources |
|---|---|
| **LLM Only** | No retrieval — parametric memory only |
| **Text RAG** | KG clinical pathway + FAISS semantic chunks |
| **KNN Only** | 5 physiologically similar past patients |
| **KNN+Text** | All three sources combined |

KNN context (≤500 chars) is placed first in the prompt to prevent truncation. Generation uses BioMistral-7B-DARE with 4-bit NF4 quantization and greedy decoding for reproducibility.

---

## Evaluation Metrics

Three heuristic metrics per generated answer:

- **Specificity** — fraction of sentences containing a numeric value (factual detail)
- **Completeness** — fraction of 20 diabetes clinical terms present (glucose, HbA1c, insulin, bicarbonate, potassium, creatinine, nephropathy, neuropathy, etc.)
- **Faithfulness** — fraction of numeric values in the answer that appear in the retrieved context (directly measures hallucination)
 the three.

---
<div align="center">

@2026

</div>
