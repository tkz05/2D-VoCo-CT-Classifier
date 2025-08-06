# Leveraging 2D VoCo-Based Pretraining to Enhance Multi-Task Multi-Class Classification of Abdominal CT Scan Medical Images

In the field of medical image analysis, the performance of deep learning models heavily depends on large-scale, high-quality annotated datasets.
However, medical annotations often face high costs and require specialized expertise. To reduce reliance on manual labeling, this study proposes a self-supervised contrastive learning framework tailored for 2D medical imaging , adapted from the 3D Volume Contrastive Learning Framework (VoCo), and integrates sequence modeling techniques to enhance performance in abdominal trauma classification tasks.
This study explores the application of the improved 2D VoCo method on abdominal CT image classification. By conducting slice-level contrastive pretraining on publicly available abdominal datasets, the model learns semantic structures across slices and transfers the pretrained backbone to the RSNA 2023 dataset for downstream multi-organ and single-organ injury classification tasks. The downstream model adopts a CNN-LSTM architecture to capture spatial-temporal correlations across slices, and a series of ablation studies are conducted to validate the effectiveness of the proposed contrastive strategy.

---

## Project Overview

This repository contains two major components:

- **VoCo-main/**  
  - Includes the code and models for Volume Contrastive Learning (VoCo), mainly for self-supervised pretraining and feature representation learning.
- **lstm_cnn/**  
  - Contains LSTM + CNN classifier code for multi-organ and single-organ classification on abdominal CT images.

---
## Setup

To ensure consistent and reproducible environments for both pretraining and classification tasks, this project uses a **Docker-based setup**.

A complete Docker environment has been published on [Hugging Face Datasets](https://huggingface.co/datasets/tkz22005/docker_env/tree/main), which includes:

- Preinstalled dependencies for VoCo-based contrastive learning and CNN-LSTM classification  
- Compatible versions of PyTorch, MONAI, and other required libraries  
- GPU support via NVIDIA CUDA and cuDNN

To get started, simply pull and run the Docker container using the instructions provided in the repository above.

---
## Dataset Download

This project relies on two main abdominal CT datasets for pretraining and classification:

### RSNA 2023 Abdominal Trauma CT Dataset
- Official download page: [https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/data](https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/data)  
- Note: You need a Kaggle account to access and download the dataset.

### FLARE 2023 Dataset (used for VoCo pretraining)
- Official challenge page: [https://codalab.lisn.upsaclay.fr/competitions/12239#learn_the_details-dataset](https://codalab.lisn.upsaclay.fr/competitions/12239#learn_the_details-dataset)  
- Hugging Face version (organized by VoCo authors): [https://huggingface.co/datasets/Luffy503/VoCo-10k/tree/main](https://huggingface.co/datasets/Luffy503/VoCo-10k/tree/main)  
  - This version was collected and organized by Dr. [Jiaxin Zhuang](https://scholar.google.com/citations?user=PfM5gucAAAAJ&hl=en) .

---
## Quick Start

### 2D VoCo

This section demonstrates how to run 2D VoCo pretraining using the provided training script.

> Dataset path and crop size settings can be modified in `data_utils.py`.  
> Training configurations such as epochs, batch size, and learning rate are specified via command-line arguments in `voco_train.py`.

#### Example
```bash
# Launch pretraining with default configuration
python voco_train.py
```
| Argument               | Type  | Default            | Description                                          |
| ---------------------- | ----- | ------------------ | ---------------------------------------------------- |
| `--logdir`             | str   | `logs`             | Directory to save logs and checkpoints               |
| `--epochs`             | int   | `100`              | Number of training epochs                            |
| `--num_steps`          | int   | `250000`           | Total training steps (overrides epochs if both used) |
| `--batch_size`         | int   | `4`                | Batch size                                           |
| `--lr`                 | float | `1e-4`             | Learning rate                                        |
| `--opt`                | str   | `adamw`            | Optimizer type (`adam`, `adamw`, etc.)               |
| `--lr_schedule`        | str   | `warmup_cosine`    | Learning rate scheduler                              |
| `--resume`             | str   | `None`             | Resume from checkpoint (e.g. `runs/model_epoch.pt`)  |
| `--loss_type`          | str   | `SSL`              | Type of loss function                                |
| `--roi_x/y/z`          | int   | *(from `roi`)*     | Patch size in each axis                              |
| `--space_x/y/z`        | float | `1.5`              | Spacing of voxel grid                                |
| `--a_min` / `a_max`    | float | `-175.0` / `250.0` | Intensity clipping range                             |
| `--use_checkpoint`     | bool  | `True`             | Use gradient checkpointing to save memory            |
| `--noamp`              | bool  | `True`             | Disable mixed-precision (AMP) training               |
| `--grad_clip`          | flag  | `False`            | Enable gradient clipping                             |
| `--smartcache_dataset` | bool  | `False`            | Use MONAI SmartCache dataset                         |
| `--cache_dataset`      | flag  | `False`            | Use MONAI CacheDataset                               |
For a complete list, see the argparse block in voco_train.py.


### Downstream

This section shows how to run the downstream classification task using the pretrained VoCo backbone and CNN-LSTM model.

> All training and evaluation parameters (e.g., model architecture, input size, learning rate) are specified in the YAML configuration file:  
> `Downstream/3d_1w_contour_cropped_96x256x256/rsna-2023-abdominal-trauma-detection-main/models/voco_lstm_efficientnetv2t/config.yaml`

You can modify this file to adjust downstream training settings.


#### Example
```bash 
    # Change to the training script directory
    cd 2D-VoCo-CT-Classifier/Downstream/3d_1w_contour_cropped_96x256x256/rsna-2023-abdominal-trauma-detection-main/src/3d_classification_voco

    # Launch training using the specified model folder
    python torch_classification_trainer.py /workspace/rsna/rsna-2023-abdominal-trauma-detection-main/models/voco_lstm_efficientnetv2t training
```
#### Usage
```bash 
    python torch_classification_trainer.py [model_folder_path] training
```
- `model_folder_path`: Directory where logs and checkpoints will be saved.  
- `training`: Mode flag .


