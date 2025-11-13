# Comparative Analysis of RNN Architectures for Sentiment Classification

This project implements and evaluates various **Recurrent Neural Network (RNN)** architectures — **Standard RNN**, **LSTM**, and **Bidirectional LSTM** — on the **IMDb Movie Review Dataset** for sentiment classification.  
The objective is to compare model performance across different **architectural**, **optimization**, and **stability** configurations (sequence length, optimizer type, and gradient clipping).

The entire experimental pipeline can be executed using the single entry script **`run_all.py`**.

---

## Project Structure

```
RNN-Sentiment-Classification/
├── preprocess.py         # Handles dataset loading, tokenizing, padding, and saving.
├── run_all.py            # Master script for executing setup, preprocessing, experiments, and evaluation.
├── utils.py              # Utility functions (seed setting, system info, model summary, etc.).
├── run_experiments.py    # Defines RNN, LSTM, and Bi-LSTM classes and runs all configurations.
├── requirements.txt      # Python dependencies.
├── README.md             # This file.
├── results/
│   ├── metrics.csv       # Table with accuracy, F1-score, and configuration data for all experiments.
│   ├── analysis.txt      # Summary highlighting the best-performing configuration.
│   ├── plots/
│   │   ├── accuracy_vs_length_full.png  # Plot showing accuracy vs. sequence length.
│   │   └── accuracy_by_optimizer_clipping.png          
└── report.pdf
```

---

## 1. Setup Instructions

### 1.1. Python Environment
This project was developed and tested on **Python 3.10+**.  
Ensure you have an up-to-date version of Python installed before proceeding.

### 1.2. Installing Dependencies
Install all required libraries using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

> **Note:**  
> The `tensorflow` library is required **only** to import the IMDb dataset via `tensorflow.keras.datasets`.  
> All model training and evaluation are performed using **PyTorch**.

---

## 2. Running the Experiments

The main script **`run_all.py`** automates the entire pipeline:
- Environment setup  
- Data preprocessing  
- Running experiments across architectures, optimizers, and stability settings  
- Saving and analyzing results  

### 2.1. Available Modes

| Mode | Description | Epochs | Configurations | Est. Runtime (CPU) |
|------|--------------|---------|----------------|--------------------|
| `quick` | Runs a single model configuration (LSTM, length=50) for a quick test. | 3 | ≈ 1 | ~5–15 minutes |
| `full` | Runs all **54** model configurations (RNN, LSTM, Bi-LSTM × 3 optimizers × 3 sequence lengths × clipping on/off). | 10 (default) | 54 | ~3–5 hours |

---

### 2.2. Example Commands

Run the **full experiment suite** (recommended for final report generation):

```bash
python run_all.py full
```

Run a **quick test** to validate the pipeline:

```bash
python run_all.py quick
```

Run the **full experiment** with a custom number of epochs (e.g., 5):

```bash
python run_all.py full --epochs 5
```

---

##  3. Expected Runtime and Outputs

### 3.1. Expected Runtime
The **full mode** executes 54 model training runs and can take approximately **3–5 hours** on an **8-core CPU** system.  
Each model typically takes between **5–20 seconds per epoch**, depending on sequence length.

### 3.2. Output Files

| File | Description |
|------|--------------|
| `results/metrics.csv` | Table summarizing all experiments (architecture, optimizer, gradient clipping, sequence length, accuracy, and F1-score). |
| `results/analysis.txt` | Text summary highlighting the best-performing configuration. |
| `results/plots/accuracy_vs_length_full.png` | Line plot showing validation accuracy and F1 vs. sequence length. |
| `results/plots/loss_vs_epoch.png` | Comparison of training loss for best and worst model configurations. |

---

## 4. System Information

Experiments were executed on the following hardware:

```
======================================================================
SYSTEM INFORMATION
======================================================================
platform: Darwin
platform_version: Darwin Kernel Version 24.5.0: Tue Apr 22 19:54:33 PDT 2025; root:xnu-11417.121.6~2/RELEASE_ARM64_T8122
processor: arm
python_version: 3.12.2
ram_gb: 16.0
cpu_count: 8
cpu_count_logical: 8
cuda_available: False
======================================================================
```

All experiments were run **CPU-only**, ensuring reproducibility under limited compute resources.

---

## 5. Results Summary

- **Dataset:** IMDb 50,000 movie reviews (balanced binary sentiment labels)
- **Vocabulary Size:** 10,000 most frequent tokens
- **Sequence Lengths Tested:** 25, 50, 100
- **Architectures:** RNN, LSTM, Bi-LSTM
- **Optimizers:** Adam, SGD, RMSprop
- **Stability Strategy:** Gradient Clipping (on/off)

**Best Configuration:**
- Model: **Unidirectional LSTM**
- Optimizer: **RMSprop**
- Sequence Length: **100**
- Gradient Clipping: **Enabled**
- Accuracy: **0.80412**
- F1-Score: **0.798**
- Avg. Epoch Time: **≈16.7s**

---

## 6. Citation and Academic Use

If you reference this work for coursework or academic purposes, cite it as:

> *Parmar, Priyanshee (2025). Comparative Analysis of RNN Architectures for Sentiment Classification. University of Maryland — DATA 641.*

---

## 7. License

This repository is provided for **academic and educational use** under the MIT License.

---

**Author:** Priyanshee Parmar  
**Date:** November 2025  
**Course:** DATA 641 – Advanced Machine Learning  
**Institution:** University of Maryland
