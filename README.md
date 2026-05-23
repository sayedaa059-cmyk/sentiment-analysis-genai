# 🐦 Twitter Disaster Sentiment Analysis

A Natural Language Processing (NLP) project that classifies tweets as real disaster events or non-disaster content using machine learning. This project covers the full ML pipeline — from raw text cleaning to model evaluation — built entirely in Python.

---

## 📌 Problem Statement

During real-world disasters, Twitter becomes a critical communication channel. However, not every tweet mentioning words like "fire" or "flood" is an actual emergency. This project builds a classifier to automatically distinguish genuine disaster tweets from unrelated ones, which can help emergency response teams filter signal from noise in real time.

---

## 📂 Dataset

- **Source** — Twitter Disaster Dataset
- **Size** — 7,613 labelled tweets
- **Target** — Binary classification
  - `1` → Real Disaster
  - `0` → Not a Disaster

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3 |
| NLP | NLTK |
| ML & Vectorization | Scikit-learn |
| Data Handling | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Environment | Google Colab / Jupyter Notebook |

---

## 🔄 Project Pipeline

```
Raw Tweet Text
      │
      ▼
 Text Cleaning         →   Lowercase, remove URLs, mentions, punctuation,
                            stopwords, lemmatization
      │
      ▼
 Vectorization         →   TF-IDF  (for ML models)
                            CountVectorizer (for Naive Bayes)
      │
      ▼
 Model Training        →   Naive Bayes
                            Logistic Regression
                            Linear SVM
                            Random Forest
      │
      ▼
 Evaluation            →   Accuracy, Precision, Recall, F1, Confusion Matrix
```

---

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 82.5% | 0.81 | 0.75 | 0.78 |
| Linear SVM | 82.1% | 0.80 | 0.75 | 0.77 |
| Naive Bayes | 81.5% | 0.79 | 0.74 | 0.76 |
| Random Forest | 79.3% | 0.77 | 0.70 | 0.73 |

> ✅ **Best Performing Model — Logistic Regression** with an F1 Score of **0.78**

---

## 📁 Repository Structure

```
twitter-disaster-sentiment/
│
├── sentiment_analysis            # Full pipeline — cleaning, vectorization, models, evaluation
├── sentiment_analysis.csv        # Raw dataset
└── README.md                     # Project documentation
```

## 📈 Key Highlights

- End-to-end NLP pipeline built from scratch
- Compared 4 different classification algorithms side by side
- Used both TF-IDF and Bag-of-Words vectorization strategies
- Visualized confusion matrices and F1 score comparisons across all models
- Applied 5-fold cross-validation to check model generalization
