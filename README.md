# A-Convolutional-Neural-Network-CNN-Classifier-for-Emotion-Recognition

## Project Overview

This project focuses on building a Convolutional Neural Network (CNN) for facial expression recognition. Given the complexity of the task, the main objective was not necessarily to achieve state-of-the-art performance, but to understand how the network processes images and, more importantly, where it tends to misclassify.  

Several experimental approaches were explored: in the first approach, a classifier with 7 output classes (corresponding to the original classes in the dataset) was implemented; in the second approach, the number of classes was reduced to 5 by removing classes with fewer samples or those that introduced significant confusion to the network; finally, the classification problem was binarized by grouping the expressions into two categories: “Positive” and “Negative”.  

The dataset used for this study is [FER-2013](https://www.kaggle.com/msambare/fer2013), consisting of 48x48 grayscale images distributed across different emotion classes. Data preprocessing included normalization, data augmentation, and class balancing through either class weighting or random sampling, depending on the experiment. The performance of the network was evaluated using standard metrics such as accuracy, precision, confusion matrices, ROC curves, AUC, and saliency maps to visualize the areas of the images most relevant for classification.
