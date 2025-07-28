# Leveraging 2D VoCo-Based Pretraining to Enhance Multi-Task Multi-Class Classification of Abdominal CT Scan Medical Images

In the field of medical image analysis, the performance of deep learning models heavily depends on large-scale, high-quality annotated datasets.
However, medical annotations often face high costs and require specialized expertise. To reduce reliance on manual labeling, this study proposes a self-supervised contrastive learning framework tailored for 2D medical imaging , adapted from the 3D Volume Contrastive Learning Framework (VoCo), and integrates sequence modeling techniques to enhance performance in abdominal trauma classification tasks.
This study explores the application of the improved 2D VoCo method on abdominal CT image classification. By conducting slice-level contrastive pretraining on publicly available abdominal datasets, the model learns semantic structures across slices and transfers the pretrained backbone to the RSNA 2023 dataset for downstream multi-organ and single-organ injury classification tasks. The downstream model adopts a CNN-LSTM architecture to capture spatial-temporal correlations across slices, and a series of ablation studies are conducted to validate the effectiveness of the proposed contrastive strategy.

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




## Dataset Download

This project relies on two main abdominal CT datasets for pretraining and classification:

### RSNA 2023 Abdominal Trauma CT Dataset
- Official download page: [https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/data](https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/data)  
- Note: You need a Kaggle account to access and download the dataset.

### FLARE 2023 Dataset (used for VoCo pretraining)
- Provided via Hugging Face by the VoCo authors collected by Dr. Jiaxin Zhuang: [https://huggingface.co/datasets/Luffy503/VoCo-10k/tree/main](https://huggingface.co/datasets/Luffy503/VoCo-10k/tree/main)  
- 這邊提供原始voco的github:  


### Notes
- Due to licensing restrictions, this repository does not include any raw CT data.  
- Please download the datasets from the official sources listed above.  
- After downloading, follow the preprocessing scripts in `VoCo-main/data/` to prepare the data for training.

