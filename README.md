# Hate Speech Detection

## Introduction
Online hate speech has seen a surge in recent years, leading to real-world consequences such as mental health issues and the perpetuation of discrimination and violence. This project aims to mitigate the proliferation of hate speech on social media platforms by automatically detecting such language. We focus on identifying hate speech using a dynamically generated dataset and employ various machine learning models to classify input documents as either hate speech or not.

# Introduction

## Table of Contents
- [Introduction](#introduction)
- [Related Work and Methodologies](#related-work-and-methodologies)
  - [Avoiding Overfitting](#avoiding-overfitting)
  - [Contextualizing Hate Speech Classifiers](#contextualizing-hate-speech-classifiers)
- [Dataset and Evaluation](#dataset-and-evaluation)
  - [Dataset](#dataset)
  - [Dataset Split](#dataset-split)
  - [Evaluation](#evaluation)
- [Methods](#methods)
  - [Baseline](#baseline)
  - [Naive Bayes](#naive-bayes)
  - [Recurrent Neural Network (RNN)](#recurrent-neural-network)
  - [Word2Vec](#word2vec)
  - [BERT](#bert)
- [References](#references)

## Related Work and Methodologies

### Avoiding Overfitting
Overfitting in text classification models can lead to high performance on training data but poor generalization on unseen data. To address this:
- **Regularization**: Dropout layers will be added to LSTM and RNN models to reduce overfitting.
- **Cross-Validation**: K-fold cross-validation will be employed to ensure models generalize well to unseen data.

### Contextualizing Hate Speech Classifiers
To enhance accuracy and reduce false positives, context-aware models are crucial. We will:
- **Utilize BERT for Context Awareness**: Pre-training BERT on a large corpus will help in understanding the context of words, improving hate speech detection accuracy.

## Dataset and Evaluation

### Dataset
We use Kaggle's "Dynamically Generated Hate Speech Dataset" with 40,463 samples, ensuring a balanced distribution of 54% hate speech and 46% non-hate.

### Dataset Split
- **Training Set (80%)**: Used for model training.
- **Development Set (10%)**: Used for model evaluation and tuning.
- **Testing Set (10%)**: Provides an unbiased performance measure on unseen data.

### Evaluation
Metrics used for evaluation include:
- **Accuracy**: General model reliability.
- **Precision**: Minimizing false positives.
- **Recall**: Detecting all instances of hate speech.
- **F1-Score**: Harmonic mean of precision and recall.

## Methods

### Baseline
The Majority Classifier, which always predicts the most frequent label from the training dataset, serves as our baseline.

### Naive Bayes
A probabilistic framework based on Bayes' theorem, suitable for textual data classification.

### Recurrent Neural Network (RNN)
RNNs process sequences in text data, recognizing patterns and using past inputs to inform future predictions.

### Word2Vec
Generates dense vector embeddings to capture semantic nuances and offer richer context to machine learning models.

### BERT
A transformer-based model with bidirectional context understanding. We fine-tune BERT's pre-trained embeddings on our dataset for optimal results.

## References
- Usharengaraju2021: Dynamically generated hate speech dataset.
- Dixon2018: Insights on overfitting in text classification models.
- Kennedy2020: Importance of contextualizing hate speech classifiers.
