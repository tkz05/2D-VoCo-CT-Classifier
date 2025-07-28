# 2D-VoCo-CT-Classifier

## 📘 Project Overview

This repository contains two major components:

- **VoCo-main/**  
  - Includes the code and models for Volume Contrastive Learning (VoCo), mainly for self-supervised pretraining and feature representation learning.
- **lstm_cnn/**  
  - Contains LSTM + CNN classifier code for multi-organ and single-organ classification on abdominal CT images.

---

## 📂 Project Structure
```
2D-VoCo-CT-Classifier/
│
├── VoCo-main/ # VoCo pretraining framework (Self-supervised learning)
│ ├── data/ # Data preprocessing scripts for pretraining
│ ├── models/ # VoCo models (EfficientNet, SwinViT, etc.)
│ ├── voco_head/ # VoCo projection head implementations (v1, v2)
│ ├── scripts/ # Training & evaluation scripts
│ └── ...
│
├── lstm_cnn/ # LSTM-CNN classifier
│ ├── dataset/ # Dataset loading & preprocessing
│ ├── models/ # LSTM + CNN architecture implementations
│ ├── scripts/ # Training & evaluation scripts
│ └── ...
│
└── README.md
```


---

## 🚀 Usage

### 1️⃣ Create environment and install dependencies
```bash
conda create -n voco python=3.10
conda activate voco
pip install -r requirements.txt
