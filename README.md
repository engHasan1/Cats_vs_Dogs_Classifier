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
  - **Download the Model**: Due to its large size (116 MB), you can download the model from [Google Drive](https://drive.google.com/file/d/1fGvzr3ZQXqDpoTHi_C-RHyLdFzq3gRnL/view?usp=drive_link).
- `accuracy_plot_vgg16_finetuned.png`: Plot of training and validation accuracy.

## Requirements
- Python 3.x
- Required libraries:
  - TensorFlow
  - OpenCV (`cv2`)
  - NumPy
  - Matplotlib
- Install the libraries using:

## Instructions
- **Requirements**: You need Python 3.x and the following libraries: TensorFlow, OpenCV (`cv2`), NumPy, and Matplotlib. Install them using:
- - **How to Use**:
1. Clone the repository from GitHub:
2. Ensure the requirements are installed (see above).
3. Download the model from the link above and place it in the project folder.
4. To classify using a camera:
- Run the `classify_with_camera.py` script:
- Point the camera at a cat or dog image.
- The model will display the classification with a confidence score (e.g., "Dog (0.92)").
- If the confidence is below 60%, it will display "Uncertain".
5. To retrain the model:
- Open `Cats_vs_Dogs_Classifier.py` in a Python environment (e.g., Google Colab or Spyder).
- Ensure the dataset is available (20,000 images in the `dogs_vs_cats/train` folder).
- Run the script step by step.
- **Notes**: To improve accuracy further, you can experiment with a ResNet50 model or unfreeze additional layers of VGG16. If you encounter issues downloading the model, ensure you use the provided link.
- **License**: This project is licensed under the [MIT License](LICENSE).
