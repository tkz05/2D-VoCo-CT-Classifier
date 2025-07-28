# 2D-VoCo-CT-Classifier

# 2D-VoCo-CT-Classifier

## 📘 專案簡介 (Project Overview)

本專案包含兩個主要資料夾：

- **VoCo-main/**  
  - 內含 Volume Contrastive Learning (VoCo) 相關程式碼與模型，主要用於自監督預訓練與特徵學習。
- **lstm_cnn/**  
  - 包含基於 LSTM + CNN 的分類模型程式碼，用於腹部 CT 影像的多器官/單器官分類。

---
This repository contains two major components:

- **VoCo-main/**
  - Includes the code and models for Volume Contrastive Learning (VoCo), mainly for self-supervised pretraining and feature representation learning.
- **lstm_cnn/**  
  - Contains LSTM + CNN classifier code for multi-organ and single-organ classification on abdominal CT images.

## 📂 專案結構 (Project Structure)

2D-VoCo-CT-Classifier/
│
├── VoCo-main/          # VoCo pretraining framework (Self-supervised learning)
│   ├── data/           # Data preprocessing scripts for pretraining
│   ├── models/         # VoCo models (EfficientNet, SwinViT, etc.)
│   ├── voco_head/      # VoCo projection head implementations (v1, v2)
│   ├── scripts/        # Training & evaluation scripts
│   └── ...
│
├── lstm_cnn/           # LSTM-CNN classifier
│   ├── dataset/        # Dataset loading & preprocessing
│   ├── models/         # LSTM + CNN architecture implementations
│   ├── scripts/        # Training & evaluation scripts
│   └── ...
│
└── README.md
