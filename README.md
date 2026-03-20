# brain-network-gat: Graph Attention Networks for Post-Stroke Aphasia Treatment Outcome Prediction

> **Research Internship Project** — University of South Carolina, Department of Psychology  
> Supervisors: Prof. Rutvik H. Desai & Prof. Yuan Wang  
> Duration: September 2024 – March 2025

---

## Overview

This project develops a **Graph Attention Network (GAT)** pipeline to classify post-stroke aphasia (PSA) patients as **responders vs. non-responders** to speech-language therapy. Brain connectivity data from structural (DTI) and functional (resting-state fMRI) neuroimaging is represented as a graph — brain regions as nodes, connectivity strengths as edges — and a GAT model is trained to predict treatment outcome.

Graph Neural Networks were chosen over classical ML specifically because the dataset is small (~42 patients). GNNs leverage the relational structure of brain connectivity directly, making them well-suited to settings where flattening data into feature vectors would discard critical spatial information.

---

## Background

Post-stroke aphasia is a language impairment caused by stroke-related brain damage. Identifying which patients will respond to treatment enables better personalization of therapy. This project uses longitudinal neuroimaging data — patients scanned at baseline and follow-up visits — to extract brain connectivity change as the predictive signal, capturing neuroplasticity over time.

Labels are derived from Philadelphia Naming Test (PNT) score improvement over 6 months, thresholded at the dataset median to produce binary responder / non-responder labels.

---

## Repository Structure

```
brain-network-gat/
├── 01_gat_longitudinal_pipeline.ipynb       # v1: longitudinal feature engineering + GAT
├── 02_gat_leakage_corrected_pipeline.ipynb  # v2: leakage-corrected, complete pipeline
└── README.md
```

### Notebook Descriptions

**`01_gat_longitudinal_pipeline.ipynb`** — v1 (reference only)

Introduces the core longitudinal feature engineering idea: instead of using a static snapshot of each patient's brain, compute the *change* in combined structural + functional connectivity between consecutive visits and average across all available visits. This inter-visit delta captures neuroplasticity as the predictive signal. The GAT model architecture is defined here. Edge construction and dataset saving are intentionally incomplete — see notebook 02 for the full working pipeline.

**`02_gat_leakage_corrected_pipeline.ipynb`** — v2 (complete, correct)

The complete, methodologically valid pipeline. Key improvements over v1:
- **Leakage fix:** replaces `train_test_split` with `GroupShuffleSplit` using patient IDs as groups, ensuring no patient appears in both train and test sets across visits
- **Patient ID persistence:** saves and reloads patient IDs separately so group-aware splitting works correctly after dataset caching
- **Edge construction:** uses `dense_to_sparse` to convert adjacency matrices to sparse graph format
- **Gradient clipping:** `max_norm=1.0` for training stability
- **Removed unused `edge_attr`** from GAT forward pass

---

## Methodology

### Graph Construction

| Component | Description |
|---|---|
| Nodes | Brain regions (~344, standard atlas) |
| Node features | Rows of the average inter-visit connectivity change matrix |
| Edges | Derived from combined structural + functional adjacency via `dense_to_sparse` |
| Predictive signal | Mean change in connectivity across visits (captures neuroplasticity) |

### Model Architecture

```
Input Graph
    │
GAT Layer 1 — 4 attention heads, concat=True  →  hidden_dim × 4
    │
BatchNorm → LeakyReLU → Dropout(0.3)
    │
GAT Layer 2 — 2 attention heads, concat=False →  hidden_dim / 2
    │
BatchNorm → LeakyReLU
    │
Global Mean Pooling  (node-level → graph-level vector)
    │
Linear Layer → Binary output (responder / non-responder)
```

### Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | Adam (lr=0.001, weight_decay=1e-5) |
| Loss | BCEWithLogitsLoss |
| Epochs | 50 |
| Split | 80/20, patient-level (GroupShuffleSplit) |
| Gradient clipping | max_norm=1.0 |
| Device | CUDA if available, else CPU |

---

## The Leakage Problem — and Why It Matters

Early versions split data at the graph level (one graph per patient-visit). Since patients had multiple visits, **the same patient could appear in both train and test sets**, causing the model to memorize patient-specific patterns rather than learn generalizable ones.

| Version | Accuracy | F1 | Valid? |
|---|---|---|---|
| v1 — with leakage | ~94.7% | ~0.94 | ❌ Inflated |
| **v2 — leakage corrected** | **~66.67%** | **~0.66** | ✅ Honest |

Fix: `GroupShuffleSplit(groups=patient_ids)` keeps all visits of a patient entirely within train or entirely within test. The lower number is the real one — and still a meaningful improvement over the 59.65% prior best on this dataset.

---

## Results

| Model | Test Accuracy | Notes |
|---|---|---|
| Prior best (GCN baseline) | 59.65% | Before this work |
| **GAT v2 (leakage corrected)** | **~66.67%** | ~7 pp improvement, valid methodology |

---

## Dataset

> ⚠️ Data is not included — proprietary clinical data under IRB protocols.

To adapt this code to your own data you will need:
- `.mat` files per patient per visit, each containing:
  - `structural_connectivity` — structural brain connectivity matrix (e.g. DTI)
  - `functional_connectivity` — functional brain connectivity matrix (e.g. rs-fMRI)
- A CSV with columns `patient_id` and `label` (1 = responder, 0 = non-responder)
- File naming convention: `MXXXXV.mat` (XXXX = 4-digit patient ID, V = visit number) — adapt the parsing in the notebooks if your convention differs

---

## Dependencies

```
torch
torch-geometric
scipy
numpy
pandas
scikit-learn
```

> PyTorch Geometric installation is CUDA-version dependent. See the [official guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

---

## Limitations and Future Work

- **Small dataset (~42 patients):** primary bottleneck; more patients would significantly improve generalization
- **Regression extension:** predicting continuous PNT score improvement rather than binary classification
- **Multi-modal fusion:** separating structural connectivity (edges) from BOLD signals (node features) via PyG `HeteroData`
- **Interpretability:** Layer-Wise Relevance Propagation (LRP) via Captum to identify which brain regions drive predictions

---

## Acknowledgements

This work was conducted during a research internship at the **University of South Carolina, Department of Psychology**, under the supervision of **Prof. Rutvik H. Desai** (Carolina Trustees Professor; Director, Institute for Mind and Brain) and **Prof. Yuan Wang** (Assistant Professor, Department of Epidemiology and Biostatistics).

Patient data is the property of the University of South Carolina and is not included in this repository.

---

## Copyright

© 2025 Haryaksh Manuh Bhardwaj. All rights reserved.

This code is shared for portfolio and reference purposes only. No part of this repository may be copied, modified, distributed, or used without explicit written permission from the author.
