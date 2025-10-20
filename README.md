# A-Convolutional-Neural-Network-CNN-Classifier-for-Emotion-Recognition

## Project Overview

This project focuses on building a Convolutional Neural Network (CNN) for facial expression recognition. Given the complexity of the task, the main objective was not necessarily to achieve state-of-the-art performance, but to understand how the network processes images and, more importantly, where it tends to misclassify.  

Several experimental approaches were explored: in the first approach, a classifier with 7 output classes (corresponding to the original classes in the dataset) was implemented; in the second approach, the number of classes was reduced to 5 by removing classes with fewer samples or those that introduced significant confusion to the network; finally, the classification problem was binarized by grouping the expressions into two categories: “Positive” and “Negative”.  

The dataset used for this study is [FER-2013](https://www.kaggle.com/msambare/fer2013), consisting of 48x48 grayscale images distributed across different emotion classes. Data preprocessing included normalization, data augmentation, and class balancing through either class weighting or random sampling, depending on the experiment. The performance of the network was evaluated using standard metrics such as accuracy, precision, confusion matrices, ROC curves, AUC, and saliency maps to visualize the areas of the images most relevant for classification.


# A Convolutional Neural Network Classifier for Emotion Recognition

## 📘 Project Overview

This project aims to develop a Convolutional Neural Network (CNN) for facial expression recognition using the FER2013 dataset. The primary objective is to analyze how the network processes images and identify areas where it tends to misclassify, rather than solely achieving state-of-the-art performance.

The project explores various experimental approaches:

- **7-class classification**: Utilizing the original dataset with seven emotion classes.
- **5-class classification**: Removing the 'Disgust' and 'Neutral' classes due to their imbalance and ambiguity.
- **Binary classification**: Merging 'Happy' and 'Surprise' into 'Positive' and 'Angry', 'Sad', 'Fear' into 'Negative' to evaluate performance on a simplified classification task.

## 📂 Repository Structure

A_Convolutional_Neural_Network_Classifier_for_Emotion_Recognition/
│
├── report.pdf # Full project report in PDF format
├── code/ # Folder containing Jupyter notebooks
│ ├── 1_CNN_Facial_Expression_Fine_Classes.ipynb
│ └── 2_CNN_Facial_Expression_Binary_Classes.ipynb
└── README.md # Project overview and instructions


## 📥 Dataset

The dataset used in this project is the FER2013 dataset, which can be downloaded from Kaggle:

👉 [https://www.kaggle.com/msambare/fer2013](https://www.kaggle.com/msambare/fer2013)

Please ensure you have the dataset extracted and organized as per the structure required by the notebooks.

## 🧪 Notebooks

### 1. `1_CNN_Facial_Expression_Fine_Classes.ipynb`

This notebook implements a CNN model trained on the original 7-class FER2013 dataset. It includes:

- Data preprocessing and augmentation
- Model architecture definition
- Training with class weights to handle class imbalance
- Evaluation using metrics such as accuracy, precision, recall, F1-score, and ROC curves
- Generation of saliency maps to visualize model focus areas

### 2. `2_CNN_Facial_Expression_Binary_Classes.ipynb`

This notebook modifies the dataset for binary classification:

- Merges 'Happy' and 'Surprise' into 'Positive'
- Merges 'Angry', 'Sad', 'Fear', and 'Disgust' into 'Negative'
- Balances the dataset by equalizing the number of samples in each class
- Trains a CNN model on the modified dataset
- Evaluates performance using the same metrics as the first notebook for comparison

## ⚙️ Requirements

To run the notebooks, you will need:

- Python 3.6+
- TensorFlow 2.x
- Keras
- NumPy
- Matplotlib
- Scikit-learn
- OpenCV (for image processing)

You can install the required packages using pip:

```bash
pip install tensorflow keras numpy matplotlib scikit-learn opencv-python


## 📄 Report
A detailed report discussing the methodology, experiments, and results is available as report.pdf in the repository. This document provides in-depth insights into the project and its findings.

