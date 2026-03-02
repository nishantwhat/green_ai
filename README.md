# 🍃 Green AI: Model Optimization & Energy Efficiency Study

## 📖 Overview
This repository contains an end-to-end ablation study on **Model Optimization Techniques for Edge AI**. It evaluates how Quantization, Structural Pruning, and Knowledge Distillation impact the predictive accuracy and physical energy consumption (Joules) of a Natural Language Processing (NLP) model.

### 🎯 Task Details
* **Task:** Sentiment Analysis (Positive, Negative, Neutral)
* **Dataset:** Hugging Face `tweet_eval` (25,000 Train / 5,000 Test)
* **Architecture:** Custom 1D-Convolutional Neural Network (1D-CNN)
* **Energy Telemetry:** Tracked via `CodeCarbon` during the inference phase.

---

## ⚠️ Important Note: CUDA-First vs. CPU-Objective
This codebase is designed with a **CUDA-first training approach, but a strict CPU-bound inference objective**. 

**Why?** While training on a GPU (CUDA) is highly efficient, PyTorch's native dynamic INT8 Quantization engine (`torch.ao.quantization`) currently only supports the CPU backend. To ensure a scientifically valid, apples-to-apples comparison of energy consumption across all optimization techniques, the telemetry tracker strictly forces the model to execute inference on the CPU.

### 🔄 How to Change the Execution Device
If you wish to bypass the CPU benchmarking and run the entire pipeline (including inference) on your GPU, you can modify the device configuration at the top of the scripts. 

Change this line:
```python
# Current CPU-forced setup for accurate benchmarking
DEVICE = torch.device("cpu")

To this:
Python

# Unrestricted hardware utilization
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

(Note: If you do this, the INT8 Quantization script will fail unless you implement a specialized GPU quantization backend like NVIDIA TensorRT).
```
🚀 How to Run the Pipeline

1. Install Dependencies
Bash

pip install torch pandas matplotlib seaborn scikit-learn codecarbon datasets

2. Execute the Pipeline
Run the scripts sequentially from the src/ directory. Ensure background applications are closed to prevent telemetry noise.
Bash

cd src
python eda_and_setup_01.py
python train_baseline_02.py
python prune_03.py
python quantize_04.py
python distill_05.py
python joint_pipeline_07.py
python final_analysis_06.py

📊 Key Results Summary

Optimization efficacy is strictly hardware-dependent. Below is a snapshot of the performance metrics.
Model Variant	Accuracy	F1 Score	CPU Energy	GPU Energy	Highlights
Baseline (FP32)	54.04%	51.31%	45.24 J	41.02 J	Reference
Quantized (INT8)	53.98%	51.28%	62.13 J	34.47 J	Preserves Accuracy perfectly
Pruned (60%)	52.86%	51.13%	25.03 J	54.76 J	🏆 Best for CPU Deployment
Distilled (T=2.0)	52.08%	48.87%	47.40 J	11.83 J	🏆 Best for GPU Deployment

    Pruning: Most effective on CPUs, physically removing mathematical overhead and halving energy consumption.

    Knowledge Distillation: Most effective on modern GPUs, achieving massive 71% energy reductions by efficiently processing smaller, smarter student networks.
