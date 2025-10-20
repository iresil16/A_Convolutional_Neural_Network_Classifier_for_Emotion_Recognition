# Computational Neuroscience: Facial Expression Recognition

**Author:** Irene Silvestro  
**Date:** October 2025  

This repository contains the code and analyses for the project *“A Convolutional Neural Network Classifier for Emotion Recognition”*.  

The project investigates facial expression recognition using convolutional neural networks (CNNs), focusing on understanding network behavior, interpretability, and the influence of dataset characteristics on classification performance.

## Project Overview

The primary objective is to explore how CNNs process facial images to discriminate between emotional states. The workflow involves dataset preprocessing, model design, training under various class configurations, and interpretability analyses using saliency maps.

### Dataset Exploration

- The FER-2013 dataset was used ([Kaggle link](https://www.kaggle.com/msambare/fer2013)).  
- Contains 35,887 grayscale images (48×48 pixels) across 7 emotion classes: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise.  
- Original splits: training and test sets; further partitioned into 70% training, 20% validation, 10% test.  
- Class imbalance addressed via balancing, class weights, and selective removal of minority or confusing classes.  

### Preprocessing

- Data augmentation: random horizontal/vertical shifts (up to 10% of image size), horizontal flips.  
- Normalization to scale pixel values between 0 and 1.  
- Minibatch size: 64 images per batch.  

### Model Architecture

- 6 convolutional layers with ReLU activations, batch normalization, max pooling, and dropout.  
- Fully connected layers followed by a softmax output layer.  
- Total parameters: ~1.39 million.  
- Architecture designed empirically to balance accuracy and training time.  

### Training

- Optimizer: ADAM; Learning rate: 0.0001.  
- Loss: categorical crossentropy.  
- Epochs: 40.  
- Class weights applied when training on imbalanced datasets.  

### Interpretability

- Saliency maps highlight input regions contributing most to class predictions.  
- Confusion matrices and ROC curves used for qualitative assessment of class separability.  

## Repository Structure

code/
├─ CNN_Classifier_FineClasses.ipynb # 5-class training with post-training binarization
├─ CNN_Classifier_Binary.ipynb # 2-class pre-binarized training
data/
├─ fer2013/ # FER-2013 dataset files
report.pdf # Full project report with methodology and analyses
figures/
├─ saliency_maps/ # Sample saliency map visualizations
