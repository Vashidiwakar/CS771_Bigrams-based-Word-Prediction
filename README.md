# Bigram-Based Word Prediction using Decision Trees

This repository contains the implementation and analysis of a machine learning model that predicts words from a dictionary using bigram features. The project was developed as part of **CS771: Introduction to Machine Learning** (IIT Kanpur).

## Overview

The objective is to build a classifier that predicts words based on the **bigrams** (pairs of consecutive characters) present in them. A **Decision Tree Classifier** is trained on processed bigram features to enable efficient and accurate prediction.

---

## Methodology

### 1. Generating Bigrams
For a given word `w` of length `n`, we generate:
bigrams(w) = {(w_i, w_{i+1}) | 1 ≤ i < n}

### 2. Processing Bigrams
- Remove duplicates
- Sort lexicographically
- Truncate to the first 5 unique bigrams

### 3. Feature Vector Construction
- Each word is mapped to a binary vector indicating the presence of selected bigrams.

### 4. Training the Model
- A **Decision Tree** is trained using `entropy` as the criterion.
- Max depth: 10

### 5. Prediction
- The trained model uses bigram presence to predict the most likely word.

---

## Evaluation Metrics

| Metric           | Value            |
|------------------|------------------|
| **Training Time** | ~0.69 seconds    |
| **Model Size**    | ~9 MB (Pickled)  |
| **Prediction Time** | ~0.07 seconds |
| **Precision**     | High (evaluated across multiple trials) |

---

## Files

- `CS771_Bigrams.pdf` – Report describing the approach and results
---

## Author

- **Name:** Vashi Diwakar  
- **Course:** CS771: Introduction to Machine Learning  
- **Instructor:** Prof. Purushottam Kar  
- **Institute:** IIT Kanpur  
- **Date:** July 2024

---

## References

- Quinlan, J. R. (1986). *Induction of Decision Trees*, Machine Learning.  
- Van Rossum, G. (2009). *Python 3 Reference Manual*.

