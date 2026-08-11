# Political Discourse Dataset

## Overview
This dataset complements the [USE24-XD dataset](https://github.com/Sensify-Lab/USE24-XD) and contains social media posts annotated with **political candidate affiliation**, **temporal context**, and **topic tokens** derived from LDA preprocessing.

It is designed for research on:
- Political discourse analysis  
- Temporal dynamics of narratives  
- Topic modeling and NLP tasks  

---

## Citation
If you use this dataset or build upon it in your research, please cite the following paper:

Wang, Q., Ostadrahimi, S., Faghri, S., Baksh, M., Mauriello, M. L., & Barner, K. E. (2026). Political Stance Detection on X During the 2024U. S. Election: A Comparative Study of Classical, Neural, and Transformer Models. [Preprint](https://www.researchsquare.com/article/rs-9735442/v1)

```bibtex
@article{wang2026political,
  title={Political Stance Detection on X During the 2024U. S. Election: A Comparative Study of Classical, Neural, and Transformer Models},
  author={Wang, Qile and Ostadrahimi, Sahar and Faghri, Safoura and Baksh, Mohammad and Mauriello, Matthew Louis and Barner, Kenneth E},
  year={2026}
}

```

## Data Schema

Each row represents a single post with the following fields:

| Column | Type | Description |
|--------|------|-------------|
| `id` | int/string | Unique identifier for the post |
| `candidate_fan` | string | Candidate affiliation label: `Trump`, `Neutral`, or `Biden_Kamala` |
| `user_state` | string | Inferred U.S. state from user-provided location |
| `sentiment_score` | float | Continuous sentiment score in the range [-1, 1] |
| `sentiment_label` | string | Sentiment category: `positive`, `negative`, or `neutral` |
| `hate_count` | int | Number of detected hate-related keywords |
| `hate_flag` | int/bool | Binary indicator for hate-related content |
| `misinfo_count` | int | Number of detected misinformation-related keywords |
| `misinfo_flag` | int/bool | Binary indicator for misinformation-related content |
| `tweet_date` | date/string | Date when the post was created |
| `tweet_hour` | int | Hour of day when the post was created, from 0 to 23 |
| `tweet_dayofweek` | int/string | Day of week when the post was created |
| `temporal_period` | string | Time period relative to the election or event |
| `engagement_raw` | int/float | Total engagement count |
| `engagement_rate` | float | Engagement normalized by impressions |
| `engagement_rate_log` | float | Log-transformed engagement rate |
| `sensitive_flag` | int/bool | Binary indicator for platform-flagged sensitive content |
| `lda_tokens` | list[string] | Preprocessed tokens used for topic modeling |

## Example

### 👉 Checkout preview: [annotation_data_sample100.csv](annotation_dataset_sample100.csv)

## Sample Rows

| id | candidate_affiliation | user_state | sentiment_score | sentiment_label | hate_count | hate_flag | misinfo_count | misinfo_flag | tweet_date | tweet_hour | tweet_dayofweek | engagement_raw | engagement_rate | engagement_rate_log | sensitive_flag | lda_tokens |
|---|---|---|---:|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|
| 1882403261095043491 | Trump | Non_US | -0.8658 | negative | 3 | 1 | 0 | 0 | 2025-01-23 | 12 | 3 | 0 | 0 | 0 | 0 | `['moron', 'get', 'ass', 'kick', 'election', ...]` |
| 1932964629195731097 | Trump | CA | -0.9482 | negative | 3 | 1 | 1 | 1 | 2025-06-12 | 0 | 3 | 2 | 0.06896551724 | 0.0666913745 | 0 | `['evidence', 'emerge', 'prove', 'wrongdoing', ...]` |
| 1886480963905183809 | Trump | Non_US | -0.9779 | negative | 3 | 1 | 0 | 0 | 2025-02-03 | 18 | 0 | 0 | 0 | 0 | 1 | `['mean', 'trump', 'deport', 'well', ...]` |


# Stance Detection Experiments

This repository contains the notebook **`stance_detection_experiment.ipynb`**, which implements the full pipeline for political stance detection on social media data. The notebook covers preprocessing, feature engineering, model training, and evaluation across multiple machine learning approaches.

---

## Overview

The notebook provides a unified experimental framework to compare:

- Classical machine learning models  
- Neural models using sentence embeddings  
- Transformer-based models  

All models are evaluated under a consistent preprocessing and evaluation pipeline.

---

## Pipeline Components

### 1. Data Preprocessing

The notebook applies a multi-stage preprocessing pipeline:

- Text normalization and cleaning  
- Removal of URLs, mentions, and noise  
- Tokenization and lemmatization  
- Hashtag and emoji handling  
- Filtering short or empty posts  

---

### 2. Feature Engineering

The following features are constructed:

#### Text Features
- TF-IDF representations (unigrams + bigrams)  
- Sentence embeddings using SBERT (all-mpnet-base-v2)  

#### Sentiment Features
- Sentiment score in [-1, 1]  
- Sentiment label (positive, negative, neutral)  

#### Hate and Misinformation Signals
- Keyword-based counts and binary flags  

#### Temporal Features
- Hour of day  
- Day of week  
- Event-based temporal period  

#### Engagement Features
- Raw engagement counts  
- Normalized engagement rate  
- Log-transformed engagement rate  

#### Metadata
- User location (state-level)  
- Platform sensitivity flag  

#### Topic Modeling
- LDA token features for thematic analysis  

---

### 3. Models Implemented

The notebook includes the following models:

#### Classical Machine Learning
- Logistic Regression (TF-IDF features)  
- Linear Support Vector Machine (SVM)  
- Histogram Gradient Boosting (HGB with SVD)  

#### Neural Model
- SBERT (all-mpnet-base-v2) embeddings + MLP classifier  

#### Transformer Model
- BERTweet (vinai/bertweet-base) fine-tuned for stance classification  

---

### 4. Training Setup

- Stratified data split:
  - 64% training  
  - 16% validation  
  - 20% test  

- Early stopping based on validation macro-F1  
- Hyperparameter tuning for all models  

---

### 5. Evaluation

Models are evaluated using:

- Macro-F1 score (primary metric)  
- Accuracy  
- Confusion matrices for error analysis  

---

## How to Run

Open the notebook:

```bash
jupyter notebook stance_detection_experiment.ipynb
