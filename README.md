# Cats vs Dogs Classifier

This project aims to classify images as either cats or dogs using a VGG16 model with fine-tuning.

## Overview

- **Objective**: Classify images as cats or dogs.
- **Dataset**: 20,000 images (10,000 cats, 10,000 dogs).
- **Model**: VGG16 with fine-tuning (unfroze the last convolutional block).
- **Validation Accuracy**: 90.38%.
- **Techniques Used**:
  - Transfer Learning using VGG16.
  - Fine-tuning to improve accuracy.
  - Extensive data augmentation (e.g., rotation, brightness adjustment).
  - Real-time classification using a camera.

## Files

- `Cats_vs_Dogs_Classifier.py`: Script for training the model using Google Colab.
- `classify_with_camera.py`: Script for real-time classification using a camera.
- `cats_vs_dogs_model_vgg16_finetuned.keras`: The trained model (validation accuracy: 90.38%).
  - Download the model from [Google Drive](https://drive.google.com/file/d/1fGvzr3ZQXqDpoTHi_C-RHyLdFzq3gRnL/view?usp=drive_link).
- `accuracy_plot_vgg16_finetuned.png`: Plot of training and validation accuracy.

## Setup

### Clone the Repository

```bash
git clone https://github.com/engHasan1/cats-vs-dogs-classifier.git
cd cats-vs-dogs-classifier
