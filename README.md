## Emotion Signal Machine Learning for Behavioral Tuning of LLMs

## Introduction

Large language models (LLMs) are increasingly deployed in conversational and interactive systems where understanding the emotional context of user input is critical. Traditional emotion detection methods typically classify text into discrete categories such as anger, joy, or sadness. While useful, these approaches fail to capture the complexity of human emotional expression, where multiple emotional signals often coexist within the same message.

This project explores predicting continuous emotional signal vectors from text using machine learning models. Instead of assigning a single emotion label, the system represents each piece of text using a structured multi-dimensional emotional state. These signals include interpretable emotional dimensions such as valence, arousal, threat, injustice, warmth, curiosity, and social proximity.

By learning this mapping from text to emotional signal space, the project aims to provide a richer emotional representation that can later be used to guide how large language models respond to users.

## Problem

Most emotion-aware NLP systems rely on discrete emotion classification, which limits their ability to capture nuanced emotional states. Human communication often expresses multiple emotional signals simultaneously, such as frustration combined with curiosity or sadness combined with social distance.

The problem addressed in this project is therefore:

How can we learn a continuous emotional signal representation of text that captures multiple emotional dimensions simultaneously?

Such a representation would allow AI systems to better interpret emotional context and make more informed decisions about how to respond.

## Dataset

This project uses the GoEmotions dataset, a large dataset of Reddit comments annotated with emotion-related labels. GoEmotions provides a diverse collection of short-form text examples that reflect a wide range of emotional expression.The dataset serves as the primary training and evaluation source for learning the relationship between text and emotional signals. From this data, target emotional signal vectors are constructed and used to supervise the machine learning models.

GoEmotions is well suited for this task because it provides real-world conversational language with rich emotional variation.

## Models

The project evaluates both baseline and neural network approaches for predicting emotional signal vectors from text.

## Baseline Model

*TF-IDF + Ridge Regression*

The baseline model converts text into TF-IDF feature vectors and uses Ridge Regression to predict continuous emotional signal values. This provides a simple and interpretable reference point for evaluating more complex models.

## Neural Models

*BiGRU Regressor*

A Bidirectional GRU (BiGRU) neural network processes sequences of word embeddings to capture contextual information within the text. The encoded representation is then passed to a regression layer that predicts the emotional signal vector.

*DistilRoBERTa Regressor*

A transformer-based model using the DistilRoBERTa architecture generates contextualized embeddings for each text input. Mean pooling of token representations is used to produce a sentence-level embedding, which is passed through a regression head to predict the emotional signal vector.

## Goal

The goal of this project is to build machine learning models capable of predicting structured emotional signal representations from text. These signal vectors provide a richer representation of emotional context than traditional discrete emotion labels. Once predicted, these signals can be used to influence behavioral strategies for large language models. For example, different signal levels may trigger different response strategies such as de-escalation, validation, explanation, or empathetic support. By allowing multiple emotional signals to influence responses simultaneously, this framework enables more nuanced and flexible behavior in AI systems. Ultimately, the project explores how emotional signal modeling can contribute to behavioral tuning and emotional alignment of large language models.

## Data Annotation

The emotional signal targets used in this project are derived from the GoEmotions dataset, which contains Reddit comments annotated with discrete emotion labels. While GoEmotions provides categorical emotion annotations, this project converts those labels into continuous emotional signal vectors that represent different emotional dimensions. Each example in the dataset is first associated with its original GoEmotions emotion labels. These labels are then mapped to a set of emotional signal dimensions such as valence, arousal, threat, injustice, warmth, curiosity, and social proximity. Each emotion label contributes weighted values to one or more emotional signal dimensions based on predefined emotion prototypes. When a comment contains multiple emotion labels, the corresponding signal contributions are combined to form a continuous emotional signal vector representing the emotional state of the text. These vectors serve as the supervised training targets for the machine learning models.

This process effectively transforms the categorical emotion annotations in GoEmotions into a structured emotional signal space, allowing models to learn a richer representation of emotional meaning in text.

## Dataset Credit

This project uses the GoEmotions dataset, introduced in:

*Demszky, D., Movshovitz-Attias, D., Ko, J., Cowen, A., Nemade, G., & Ravi, S. (2020).
GoEmotions: A Dataset of Fine-Grained Emotions.
Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL).*

The dataset consists of Reddit comments labeled with fine-grained emotion categories and provides a large-scale resource for studying emotional expression in text By allowing multiple emotional signals to influence responses simultaneously, this framework enables more nuanced and flexible behavior in AI systems. Ultimately, the project explores how emotional signal modeling can contribute to behavioral tuning and emotional alignment of large language models.

## Evaluation Metrics

To evaluate how accurately the models predict emotional signal vectors, this project uses both regression metrics and classification-style metrics derived from binned signal values.

*Regression Metrics*

Because the models predict continuous emotional signal values, the primary evaluation metrics measure the difference between predicted signals and ground truth signals.

*Mean Absolute Error (MAE)*

MAE measures the average absolute difference between predicted emotional signal values and the true values. Lower MAE indicates that the predicted signals are numerically closer to the true emotional signals.

*Root Mean Squared Error (RMSE)*

RMSE measures the square root of the average squared prediction error. This metric penalizes larger prediction errors more strongly than MAE.

*R² (Coefficient of Determination)*

R² measures how well the model explains the variance in the emotional signal values. Higher values indicate better predictive performance.

## Binned Signal Evaluation
To make the results easier to interpret, continuous signal values are also converted into discrete intensity bins. Each signal value is mapped to one of the following categories:

none

very_low

low

medium

high

This allows the evaluation to also measure how accurately the model predicts coarse emotional signal intensity levels.

## Classification Metrics

After binning the signal values, the following metrics are calculated:

*Accuracy*

Measures how often the predicted signal intensity category matches the true category.

*Precision*

Measures how often predicted signal intensities are correct for each category.

*Recall*

Measures how well the model identifies all true instances of each signal intensity.

*F1 Score*

The harmonic mean of precision and recall, providing a balanced evaluation of classification performance.

## Per-Signal Evaluation

In addition to overall performance metrics, evaluation is also performed individually for each emotional signal dimension. This allows analysis of which signals are easier or harder for models to predict.

Per-signal metrics include:

*MAE*

*RMSE*

*R²*

*Accuracy*

*Precision*

*Recall*

*F1 Score*

This detailed evaluation helps compare how different models perform across the full emotional signal space.

## Dependencies

Python 3.10+

Main libraries used:
- PyTorch
- Transformers
- Hugging Face Datasets
- scikit-learn
- NumPy
- pandas
- tqdm

## Running the Models

Train baseline model:

python train_TFIDF_RR.py

Train BiGRU model:

python train_BIGRU.py

Train DistilRoBERTa model:

python train_distilroberta.py

Run inference:

python test_roberta.py

python test_TFIDF_RR.py

python test_BIGRU.py

## Future Work

A natural extension of this work is to integrate the predicted emotional signal representations directly into large language model (LLM) response generation.

In a full conversational system, the emotional signal vectors produced by the models in this project could be used as an intermediate layer between user input and LLM output. These signals could guide behavioral strategies for the language model, influencing how responses are generated based on emotional context.

Future work may explore:

Real-time emotional signal detection during conversation so that emotional context can influence responses dynamically.

Integrating emotional signals into LLM prompting or decoding strategies to guide response tone and behavior.

Tracking emotional signals across multi-turn conversations, allowing the system to model evolving emotional states over time.

Optimizing the signal prediction pipeline for speed, enabling practical use in interactive systems.

Such capabilities would be particularly relevant for emotion-sensitive conversational systems, including applications such as mental health support tools, therapeutic chatbots, and emotionally aware digital assistants. By combining structured emotional signal modeling with LLM response generation, future systems could produce responses that are more contextually aware, adaptive, and emotionally aligned with users.

## Trained Models

The trained model weights for BIGRU and distilroberta are not included in the repository due to file size limits.

Models can be reproduced by running the training scripts:

python train_BIGRU.py
python train_distilroberta.py
