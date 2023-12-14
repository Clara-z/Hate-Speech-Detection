# Hate Speech Detection

## Introduction
The proliferation of hate speech on social media poses serious societal challenges, from mental health impacts to fostering discrimination and violence. Addressing this, our project aims to automate the detection of hate speech, utilizing a diverse dataset and advanced machine learning techniques. Our goal is to enhance online safety and inclusivity by accurately classifying texts as hate speech or non-hate speech.

## Table of Contents
- [Introduction](#introduction)
- [Requirements](#requirements)
- [Related Work and Methodologies](#related-work-and-methodologies)
  - [Avoiding Overfitting](#avoiding-overfitting)
  - [Context-Aware Hate Speech Detection](#context-aware-hate-speech-detection)
  - [Model Hyperparameter Optimization](#model-hyperparameter-optimization)
- [Dataset and Evaluation](#dataset-and-evaluation)
  - [Dataset](#dataset)
  - [Dataset Split](#dataset-split)
  - [Evaluation](#evaluation)
- [Methods](#methods)
  - [Baseline](#baseline)
  - [Naive Bayes](#naive-bayes)
  - [BERT](#bert)
- [Results](#results-and-comparative-analysis)
- [Conclusion](#conclusion)
- [References](#references)

## Requirements
Clone the repository and install the required packages:
```
git clone https://github.com/Clara-z/CSCI467-final-project.git
cd CSCI467-final-project
pip install -r requirements.txt
```
Train and evaluate the model:
```
python baseline.py
python naive_bayes.py
python BERT.py
```

## Related Work and Methodologies

### Avoiding Overfitting
Ensuring our models' robust performance on unseen data, we implemented the following strategies:
- **Cross-Validation**: We utilized k-fold cross-validation, particularly in the Naive Bayes model, to enhance the model's ability to generalize across different data samples.
- **Early Stopping**: In the BERT model, we incorporated an early stopping mechanism during training. This approach halts the training process if the model's performance on the validation set does not improve for a predetermined number of epochs, thus preventing overfitting.

### Context-Aware Hate Speech Detection
Informed by the research of Kennedy et al. (2020) and Dixon et al. (2018), our approach emphasizes the crucial role of context in hate speech detection. Utilizing the advanced capabilities of BERT, known for its deep contextual analysis, we aim to capture the subtle and complex nuances of hate speech. This methodology aligns with the current best practices in the field, advocating for balanced datasets and context-aware models to ensure effective and unbiased detection of hate speech across various online platforms.

### Model Hyperparameter Optimization
In optimizing our BERT model, we utilized Weights & Biases (WandB) for hyperparameter tuning. WandB's capabilities in experiment tracking and hyperparameter space exploration greatly enhanced our model's performance. By systematically evaluating combinations of learning rates, optimizer types, and batch sizes, WandB helped identify the most effective settings for our model, leading to improved accuracy and efficiency in hate speech detection.

## Dataset and Evaluation

### Dataset
The project utilizes the "Dynamically Generated Hate Speech Dataset" from Kaggle, featuring 40,463 entries. This dataset is meticulously balanced, with a near-equal distribution of hate speech and non-hate categories.

### Dataset Split
- **Training Set (70%)**: Used for model training.
- **Development Set (10%)**: Used for model evaluation and tuning.
- **Testing Set (20%)**: Provides an unbiased performance measure on unseen data.

### Evaluation
Metrics used for evaluation include:
- **Accuracy**: General model reliability.
- **Precision**: Minimizing false positives.
- **Recall**: Detecting all instances of hate speech.
- **F1-Score**: Harmonic mean of precision and recall.

## Methods

### Baseline
Serving as a foundational comparison point, this model predicts the training dataset's most frequent label, regardless of input text.

### Naive Bayes
This probabilistic model, based on Bayes’ theorem, is enhanced with TF-IDF vectorization and optimized through cross-validation and hyperparameter tuning, focusing on linguistic patterns in text classification.

### BERT
BERT, with its deep learning architecture and contextual word understanding, represents a significant advancement in hate speech detection. We fine-tune BERT with a dense layer and optimize it using a Weights and Biases (WandB) system, ensuring robust performance in complex linguistic analysis.

## Results and Comparative Analysis
Our study reveals distinct performance variations among the Majority Classifier, Naive Bayes, and BERT models. BERT's superior performance highlights its advanced capabilities in contextual understanding, though it also encounters challenges with ambiguities and complex sentence structures.

## Conclusion
Our evaluation of various models, including Naive Bayes and BERT, underscores the significance of advanced models for effective hate speech detection. We propose future work focusing on data enrichment, fine-tuning sensitivity to subtleties, and hybrid modeling approaches for more refined hate speech detection systems.

## References
- Usharengaraju2021: Dynamically generated hate speech dataset.
- Dixon2018: Insights on overfitting in text classification models.
- Kennedy2020: Importance of contextualizing hate speech classifiers.
