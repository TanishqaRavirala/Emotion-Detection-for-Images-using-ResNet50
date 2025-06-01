Emotion Detection using ResNet50
Overview
This project implements an emotion detection model for images using the ResNet50 architecture with TensorFlow and Keras. The model is trained to classify facial expressions into seven emotion categories: disgust, anger, contempt, happy, fear, surprise, and sadness. The dataset used is the CK+48 dataset, which contains labeled images of facial expressions.
The project is implemented in a Jupyter Notebook (Emotion_detection_ResNet50.ipynb) and is designed to run on Google Colab with GPU acceleration for efficient training.
Features

Dataset: Utilizes the CK+48 dataset, organized into subfolders for each emotion.
Preprocessing: Images are converted to grayscale and resized to 48x48 pixels.
Model: Leverages the ResNet50 architecture with additional custom layers (e.g., Dense, Dropout, BatchNormalization) for emotion classification.
Training: Includes data splitting, model training with Adam optimizer, and model checkpointing for saving the best weights.
Evaluation: Provides functionality to predict emotions on new images.
Dependencies: Uses TensorFlow, Keras, OpenCV, NumPy, Pandas, and Matplotlib.

Requirements
To run this project, you need the following dependencies:

Python 3.x
TensorFlow (tensorflow)
Keras (keras)
OpenCV (opencv-python)
NumPy (numpy)
Pandas (pandas)
Matplotlib (matplotlib)
scikit-learn (scikit-learn)

You can install the dependencies using pip:
pip install tensorflow opencv-python numpy pandas matplotlib scikit-learn

Setup

Clone the Repository:
git clone https://github.com/your-username/Emotion-Detection-for-Images-using-ResNet50.git
cd Emotion-Detection-for-Images-using-ResNet50


Dataset:

The project uses the CK+48 dataset, which should be placed in a folder named CK+48 in your Google Drive (e.g., /content/drive/MyDrive/CVIP Project/CK+48).
Ensure the dataset is organized into subfolders named after each emotion (e.g., disgust, anger, contempt, happy, fear, surprise, sadness).
If you don't have the dataset, you can obtain it from the CK+ dataset source (ensure you have the necessary permissions).


Google Colab Setup:

Upload the Emotion_detection_ResNet50.ipynb notebook to Google Colab.
Mount your Google Drive in Colab to access the dataset:from google.colab import drive
drive.mount('/content/drive')


Ensure your dataset path in the notebook matches the location in your Google Drive.


GPU Acceleration:

In Google Colab, enable GPU acceleration by navigating to Runtime > Change runtime type > Hardware accelerator > GPU.



Usage

Open the Notebook:

Open Emotion_detection_ResNet50.ipynb in Google Colab or a local Jupyter environment.


Run the Cells:

Execute the cells sequentially to:
Import dependencies.
Load and preprocess the CK+48 dataset.
Build and train the ResNet50-based model.
Save the trained model weights.
Test the model on sample images.




Predict Emotions:

To predict emotions on a new image, update the image path in the prediction cell (e.g., /content/drive/MyDrive/CVIP Project/test_image2.jpg).
The model will output the predicted emotion (e.g., sadness).



Project Structure

Emotion_detection_ResNet50.ipynb: Main Jupyter Notebook containing the code for data loading, preprocessing, model building, training, and prediction.
CK+48/: Directory containing the dataset (not included in the repository; must be provided by the user).
README.md: This file, providing an overview and instructions for the project.

Dataset
The CK+48 dataset contains images of facial expressions categorized into seven emotions:

disgust
anger
contempt
happy
fear
surprise
sadness

Each subfolder in the dataset directory corresponds to one emotion, and images are stored in PNG format. The notebook preprocesses these images by converting them to grayscale and resizing them to 48x48 pixels.
Model Architecture
The model is based on ResNet50, pre-trained on ImageNet, with the following modifications:

Custom layers added for emotion classification (e.g., Dense, Dropout, BatchNormalization).
Input images are grayscale, resized to 48x48 pixels.
Output layer uses softmax activation to predict one of the seven emotion classes.

Training

The dataset is split into training and validation sets using train_test_split from scikit-learn.
The model is trained using the Adam optimizer with categorical cross-entropy loss.
ModelCheckpoint callback is used to save the best model weights based on validation accuracy.

Results
The model can predict emotions on new images, as demonstrated in the notebook's output (e.g., predicting sadness for a test image). Performance metrics such as accuracy and loss can be evaluated during training.
Contributing
Contributions are welcome! Please feel free to:

Submit issues for bugs or feature requests.
Create pull requests with improvements or additional features.

License
This project is licensed under the MIT License. See the LICENSE file for details.
Acknowledgments

The CK+48 dataset providers for making the dataset available.
TensorFlow and Keras communities for their excellent documentation and tools.
Google Colab for providing free GPU resources for training.

Contact
For questions or suggestions, please open an issue on this repository or contact the maintainer at your-email@example.com.
