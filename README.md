# Intelligent Farsi Search Suggestion System

An NLP-based search suggestion system for Farsi that improves the search experience by handling spelling mistakes, keyboard layout errors, and query variations.

## Overview

Search suggestion is a key component of modern search systems. Traditional autocomplete methods only suggest queries that start with the entered prefix, which often fails when users make typing mistakes or use the wrong keyboard layout.

This project presents a practical search suggestion pipeline for Farsi that combines text preprocessing, language detection, keyboard layout correction, machine learning, and edit-distance ranking to generate meaningful search suggestions.

The project was developed individually as part of an NLP challenge inspired by a real-world transportation search scenario.


## Problem Statement

Users often submit imperfect search queries due to:

* Typographical errors
* Using the English keyboard instead of the Farsi keyboard
* Incomplete queries
* Different writing styles

These issues reduce search quality and negatively impact user experience.

The objective of this project is to build a robust suggestion system that predicts the intended destination and returns the five most relevant search suggestions.


## Dataset

The project uses search log data containing users' search behavior.

Each record includes information such as:

* User typed query (`TypedStrings`)
* Final selected destination (`AcceptString`)
* Service type
* Sequential search inputs

The dataset was filtered to include Bus and Taxi services.


## Solution Pipeline

The search pipeline consists of the following stages:

1. Data preprocessing
2. Language detection
3. Keyboard layout correction
4. Feature engineering
5. Feature encoding
6. Destination prediction using a Decision Tree classifier
7. Candidate ranking using Levenshtein Distance
8. Generation of the Top-5 search suggestions

```
User Query
      │
      ▼
Language Detection
      │
      ▼
Keyboard Layout Correction
      │
      ▼
Feature Engineering
      │
      ▼
Decision Tree Classifier
      │
      ▼
Levenshtein Ranking
      │
      ▼
Top 5 Suggestions
```


## Features

* Automatic language detection
* English-to-Farsi keyboard layout conversion
* Query normalization
* Machine learning–based destination prediction
* Edit-distance ranking for similar city names
* Top-5 search suggestion generation


## Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* python-Levenshtein
* Jupyter Notebook


## Project Structure

```
Farsi-Search-Suggestions/
│
├── data/
├── notebooks/
├── src/
├── README.md
└── requirements.txt
```


## Results

The prediction model is evaluated using:

* Accuracy
* Weighted F1-score

After predicting the most likely destination, Levenshtein Distance is used to retrieve similar city names and generate the final Top-5 search suggestions.


## Example

**Input**

```
fhfg
```

**Output**

```
بابل
بابلسر
زابل
...
```

---

## Author

**Mohadese Kohansal**
