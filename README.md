# 🚀 CCT-based CIFAR-10 Image Classification

## 📖 Project Overview

This project presents an implementation of a **Compact Convolutional Transformer (CCT)** for classifying images from the **CIFAR-10** dataset — a well-known benchmark dataset for computer vision tasks.  
The Compact Convolutional Transformer is an efficient deep learning model that integrates the strengths of Convolutional Neural Networks (CNNs) for local feature extraction and Transformer Encoders for capturing global dependencies, while keeping the architecture lightweight and scalable for smaller datasets like CIFAR-10.

The primary objectives of this project are:
- To design an efficient model that outperforms traditional CNNs by utilizing self-attention mechanisms.
- To incorporate advanced data augmentation and regularization techniques to improve generalization.
- To build a reproducible pipeline for training, validating, and testing models in a modular and scalable way.
- To perform thorough evaluation using standard classification metrics such as Accuracy, Precision, Recall, F1-Score, and visualize results with a confusion matrix.

---

## 📦 Repository Structure

```plaintext
CCT-CIFAR10-Classification/
│
├── README.md                # Project description and documentation
├── requirements.txt         # Python package dependencies
│
├── code/                    # Main Python scripts
│   ├── Configure.py          # Hyperparameter and configuration settings
│   ├── DataLoader.py         # Dataset loading and augmentation
│   ├── ImageUtils.py         # Data preprocessing and visualization utilities
│   ├── loss.py               # Label smoothing cross-entropy loss
│   ├── main.py               # Entry point for training, testing, and prediction
│   ├── Model.py              # Model training and evaluation logic
│   ├── Network.py            # CCT architecture implementation
│
├── data/                    # CIFAR-10 dataset files
│
├── saved_models/            # Model checkpoints after training
│
├── outputs/                 # Visualizations (confusion matrices, sample predictions)
