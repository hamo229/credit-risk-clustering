# Home Equity Loan Risk Analysis — ML & AI Project

> An AI-driven credit risk analysis system using clustering, PCA, fuzzy logic, and genetic algorithms on the Home Equity dataset.

---

## Overview

This project analyzes home equity loan applicants to identify credit risk patterns and estimate loan approval probability using machine learning and AI techniques.

The workflow includes:
- Data preprocessing & missing value handling
- Exploratory Data Analysis (EDA)
- PCA dimensionality reduction
- K-Medoids & Hierarchical clustering
- Fuzzy logic inference system
- Genetic Algorithm feature selection

---

## Dataset

- **Rows:** 5,960
- **Features:** 13
- **Target:** `BAD` (1 = default, 0 = good credit)

### Main Features
`LOAN`, `VALUE`, `DEBTINC`, `DEROG`, `DELINQ`, `CLAGE`, `JOB`, `REASON`

---

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Scikit-fuzzy
- Matplotlib
- Seaborn

---

## Key Techniques

### Data Preprocessing
- Median imputation
- One-Hot Encoding
- Standardization
- Outlier treatment using IQR

### PCA
- Reduced dimensionality after encoding
- Retained 82.4% explained variance using 8 principal components

### Clustering
- **K-Medoids** with Manhattan distance
- **Agglomerative Clustering** (Ward, Average, Single linkage)

### Fuzzy Logic System
Estimated loan approval probability based on:
- Cluster risk level
- Debt-to-income ratio

### Genetic Algorithm
Optimized feature selection for Logistic Regression classification.

---

## Results

| Task | Result |
|------|--------|
| PCA Variance Retained | 82.4% |
| Best Clustering Method | Ward Linkage |
| Logistic Regression Accuracy | 82.38% |
| Selected Features via GA | 15 / 20 |


## How to Run

```bash
git clone https://github.com/yourusername/home-equity-risk-analysis.git

cd home-equity-risk-analysis

pip install -r requirements.txt

jupyter notebook
```

---

## Key Learnings

- Missing value strategy significantly impacts clustering quality.
- Silhouette score alone is not enough to evaluate clustering.
- Fuzzy logic can transform clustering outputs into interpretable decision systems.
- Genetic Algorithms effectively reduce dimensionality while maintaining performance.

---

## Notes

This project was developed as part of an AI & Machine Learning coursework focused on:
- Credit Risk Analysis
- Clustering
- PCA
- Fuzzy Logic
- Evolutionary Algorithms
