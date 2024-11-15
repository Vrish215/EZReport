# Medical Document Classification

This project is a student-led initiative for a hackathon, aiming to improve medical diagnostics by automating the classification of brain MRI scans. The goal is to identify the presence or absence of tumors in MRI images, aiding radiologists and healthcare professionals in faster, more accurate diagnoses.

## Table of Contents
- [Overview](#overview)
- [Project Goals](#project-goals)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation and Setup](#installation-and-setup)
- [Usage](#usage)
- [Results and Evaluation](#results-and-evaluation)
- [Future Enhancements](#future-enhancements)

## Overview
This project uses deep learning with a pre-trained VGG16 model to classify brain MRI images into two categories:
- **Yes**: Tumor is present.
- **No**: No tumor is present.

The project leverages transfer learning to fine-tune the VGG16 model, making it suitable for binary classification on medical images.

## Project Goals
1. Automate brain tumor detection in MRI images.
2. Improve diagnostic accuracy and speed, providing a helpful tool for radiologists.
3. Create a baseline model with potential for further development and optimization.

## Dataset
The dataset used for this project includes brain MRI images organized into two folders:
- **yes**: Images showing a tumor.
- **no**: Images without a tumor.

These images are resized to 224x224 pixels for compatibility with the VGG16 model.

## Model Architecture
- **Base Model**: VGG16 pre-trained on the ImageNet dataset.
- **Custom Layers**: Added dense layers for binary classification:
  - Global Average Pooling layer
  - Dense layer with 1024 neurons and ReLU activation
  - Dense layer with 512 neurons and ReLU activation
  - Final Dense layer with 2 neurons and softmax activation for classification.
## Results and Evaluation
- **Training and Validation Accuracy**: The model's accuracy for both training and validation datasets is recorded across epochs. These metrics provide insight into the model’s learning progress and potential overfitting.
- **Loss and Accuracy Metrics**: Both loss and accuracy are evaluated for each epoch, helping to gauge the model's performance and generalization capability.




![GG](https://github.com/user-attachments/assets/9a2ab294-a9fd-4837-8880-faeb4fd5676c)


![Screenshot 2024-11-10 174219](https://github.com/user-attachments/assets/3cdba0ff-d5d9-4b76-81c6-a2046d85bb88)
