# Emotion Detection using ResNet50

## Overview
This project implements an emotion detection model for images using the ResNet50 architecture with TensorFlow and Keras. The model classifies facial expressions into seven emotions: disgust, anger, contempt, happy, fear, surprise, and sadness. The CK+48 dataset is used, and the code is implemented in a Jupyter Notebook designed for Google Colab with GPU acceleration.

## Features
- **Dataset**: CK+48 dataset with images organized by emotion.
- **Preprocessing**: Images converted to grayscale and resized to 48x48 pixels.
- **Model**: ResNet50 with custom layers (Dense, Dropout, BatchNormalization) for emotion classification.
- **Training**: Uses Adam optimizer, categorical cross-entropy loss, and ModelCheckpoint for saving best weights.
- **Prediction**: Supports emotion prediction on new images.
- **Dependencies**: TensorFlow, Keras, OpenCV, NumPy, Pandas, Matplotlib, scikit-learn.

## Requirements
- Python 3.x
- Install dependencies:
  ```bash
  pip install tensorflow opencv-python numpy pandas matplotlib scikit-learn
  ```

## Setup
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/Emotion-Detection-for-Images-using-ResNet50.git
   cd Emotion-Detection-for-Images-using-ResNet50
   ```

2. **Dataset**:
   - Place the CK+48 dataset in `/content/drive/MyDrive/CVIP Project/CK+48` on Google Drive.
   - Subfolders should be named: `disgust`, `anger`, `contempt`, `happy`, `fear`, `surprise`, `sadness`.
   - Obtain the dataset from [CK+ dataset source](http://www.consortium.ri.cmu.edu/ckagree/) (requires permission).

3. **Google Colab**:
   - Upload `Emotion_detection_ResNet50.ipynb` to Google Colab.
   - Mount Google Drive:
     ```python
     from google.colab import drive
     drive.mount('/content/drive')
     ```
   - Set runtime to GPU: `Runtime > Change runtime type > Hardware accelerator > GPU`.

## Usage
1. Open `Emotion_detection_ResNet50.ipynb` in Google Colab or a local Jupyter environment.
2. Run cells to:
   - Load and preprocess the dataset.
   - Build and train the ResNet50 model.
   - Save model weights.
   - Predict emotions on new images (e.g., `/content/drive/MyDrive/CVIP Project/test_image2.jpg`).
3. The model outputs predictions like `sadness` for test images.

## Project Structure
- `Emotion_detection_ResNet50.ipynb`: Main notebook with all code.
- `CK+48/`: Dataset directory (not included; user-provided).
- `README.md`: This file.

## Dataset
CK+48 contains images of facial expressions in seven categories. Images are preprocessed to grayscale and 48x48 pixels.

## Model Architecture
- Based on ResNet50 (pre-trained on ImageNet).
- Custom layers: Dense, Dropout, BatchNormalization.
- Output: Softmax for seven emotion classes.

## Training
- Data split: Training and validation sets via `train_test_split`.
- Optimizer: Adam.
- Loss: Categorical cross-entropy.
- Saves best weights using ModelCheckpoint.

## Results
The model predicts emotions on test images (e.g., `sadness` for `test_image2.jpg`). Training metrics include accuracy and loss.

## Contributing
- Submit issues for bugs or feature requests.
- Create pull requests for improvements.

## Acknowledgments
- CK+48 dataset providers.
- TensorFlow, Keras, and Google Colab communities.

## Contact
Open an issue or contact [tanishkaravirala@gmail.com](mailto:tanishkaravirala@gmail.com).
