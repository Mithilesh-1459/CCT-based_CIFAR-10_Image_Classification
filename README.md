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
```

## 🔥 Key Features and Highlights
Compact Convolutional Transformer (CCT):
- Combines convolutional layers for token generation and lightweight transformer encoders for context modeling, enabling powerful yet efficient feature learning.

Label Smoothing Cross-Entropy Loss:
- Introduced to mitigate model overconfidence and improve generalization during training.

Data Augmentation with AutoAugment Policies:
- Applies random combinations of color transformations, geometric shifts, and flips to enhance dataset variability.

Dynamic Learning Rate Scheduling:
- Utilizes cosine annealing strategy to adjust the learning rate during training, promoting better convergence.

Training-Validation Split:
- Implements an 80-20 split on the training set for model validation during training.

GPU-Enabled Training:
- Fully utilizes CUDA devices if available for faster computations.

Evaluation Metrics and Visualization:
- Generates a confusion matrix along with Precision, Recall, and F1-Score calculations after testing.

Reproducible Architecture:
- Modularized scripts to allow easy hyperparameter tuning, architecture changes, or dataset swapping.

## 🧠 CCT Model Architecture Overview
Tokenizer:
Initial convolutional layers extract dense local feature representations (tokens) from the input images.

Transformer Encoder Layers:
Multiple transformer blocks process the tokenized embeddings with self-attention and feed-forward networks.

Sequence Pooling (Optional):
Rather than using a class token, adaptive sequence pooling is applied to aggregate the final representations.

Final Classifier:
A fully connected layer outputs logits corresponding to the 10 CIFAR-10 classes.

📊 CIFAR-10 Dataset
60,000 color images (32x32 pixels)

10 object categories:

Airplane

Automobile

Bird

Cat

Deer

Dog

Frog

Horse

Ship

Truck

50,000 images for training and 10,000 images for testing.

🛠️ Technologies Used
Python 3.8+

PyTorch 1.10+

Torchvision

Matplotlib

Seaborn

NumPy
