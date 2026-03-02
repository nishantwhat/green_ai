# green_ai

# 🍃 Green AI: Model Optimization & Energy Efficiency Study

[![Python 3.14](https://img.shields.io/badge/Python-3.14-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CPU%2FGPU-EE4C2C.svg)](https://pytorch.org/)
[![CodeCarbon](https://img.shields.io/badge/Energy_Tracking-CodeCarbon-00cc66.svg)](https://codecarbon.io/)

## 📖 Overview
This repository contains a comprehensive ablation study on **Model Optimization Techniques for Edge AI**. The project evaluates how Quantization, Structural Pruning, and Knowledge Distillation impact the accuracy, memory footprint, and physical energy consumption (Joules) of a Natural Language Processing (NLP) model.

The goal is to answer a critical industry question: **Which optimization technique achieves the best trade-off between predictive accuracy and energy efficiency?**

### 🎯 Task details
* **Task:** Sentiment Analysis (Positive, Negative, Neutral)
* **Dataset:** Hugging Face `tweet_eval` (25,000 Train / 5,000 Test Split)
* **Baseline Architecture:** Custom 1D-Convolutional Neural Network (1D-CNN)
* **Telemetry:** Hardware energy tracked via `CodeCarbon` during the inference phase.

---

## 🗺️ Experimental Pipeline

```mermaid
graph TD

classDef data fill:#E3F2FD,stroke:#1565C0,stroke-width:2px,color:#000;
classDef process fill:#F3E5F5,stroke:#7B1FA2,stroke-width:2px,color:#000;
classDef model fill:#E8F5E9,stroke:#2E7D32,stroke-width:2px,color:#000;
classDef eval fill:#FFF3E0,stroke:#E65100,stroke-width:2px,color:#000;
classDef joint fill:#FFFDE7,stroke:#FBC02D,stroke-width:2px,color:#000;

A[(Hugging Face 'tweet_eval' Dataset\n25k Train / 5k Test)]:::data --> B(Tokenization & Preprocessing\nVocab: 10,000 | Max Len: 50):::process
B --> C[Train Baseline 1D-CNN\nFP32 Precision]:::model

C --> D{Apply Optimization Techniques}:::process

D -->|Precision Reduction| E[Quantization\nDynamic INT8]:::model
D -->|Structural Sparsity| F[Pruning\nUnstructured 20%, 40%, 60%]:::model
D -->|Knowledge Transfer| G[Knowledge Distillation\nHeavy Teacher -> Student T=2.0 & 5.0]:::model

G -->|Extract Best Student| H(Joint Pipeline Execution:\nDistillation -> 20% Prune -> INT8 Quantize):::joint
H --> I[Ultimate Edge Model]:::model

C -.-> J[[Hardware Evaluation via CodeCarbon\nCPU vs. GPU Inference Tracking]]:::eval
E -.-> J
F -.-> J
G -.-> J
I -.-> J

J --> K(Record & Compile Metrics:\nAccuracy, F1-Score, Energy Joules):::process
K --> L((Final Comparative Analysis\n& Radar Trade-off Charts)):::data
