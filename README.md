# 🧠 Mann-o-meter: Mental Health Chatbot + Stress Analyzer Web App

**Mann-o-meter** is an intelligent mental health support system built with **Flask**, integrating:
- A responsive **chatbot** for mental health queries using advanced **Information Retrieval (IR)** models.
- A **stress analyzer** using **Machine Learning (ML)** to predict and visualize user stress levels from lifestyle inputs.
- Data visualizations for personal and population-level stress patterns.

---

## 🚀 Features

### 🗣️ 1. Mental Health Chatbot (IR-Powered)
- **Emergency Detection**: Detects high-risk keywords like *"suicide"*, *"want to die"*, etc., and immediately provides **India-specific emergency support**.
- **FAQ-based Retrieval**:
  - Uses **hybrid retrieval** with:
    - **TF-IDF** (term frequency–inverse document frequency)
    - **BM25** (Okapi Best Match retrieval)
    - **BERT-based embeddings** via `SentenceTransformers`
  - Combines multiple scores to rank answers.
  - Evaluated with **NDCG** (Normalized Discounted Cumulative Gain) to assess retrieval relevance.

### 🤖 2. Stress Analyzer (ML-Powered)
- **Input Parameters**: Age, Sleep Duration, Activity Level, Heart Rate, Blood Pressure
- **Outputs**:
  - **Stress Level (0–10)**: via **Random Forest Regressor**
  - **Stress Category** (`Low`, `Medium`, `High`): via **Random Forest Classifier**
- **Model Evaluation**:
  - `R² Score` for regression
  - `Accuracy`, `Classification Report` for classification

### 📊 3. Data Visualizations
- **Stress Distribution**: Bar chart showing population stress levels.
- **Age vs Stress**: Average stress comparison across age groups.
- **Personal Health Dashboard**:
  - Radar chart (sleep, activity, heart health)
  - Metric comparison (sleep, BP, heart rate)
  - Stress level vs peer group

---

## 🧠 Key Concepts Used

### 🔍 Information Retrieval (IR)
- **Text Preprocessing**: Tokenization, Stopword Removal, Lemmatization, Synonym Normalization
- **Document Representation**:
  - `TF-IDF Vectorization` for sparse document scoring
  - `BM25Okapi` for term-based scoring
  - `Sentence-BERT` for semantic similarity
- **Hybrid Retrieval**: Combines sparse (TF-IDF, BM25) and dense (BERT) representations
- **Ranking & Evaluation**:
  - Score fusion with custom weights
  - Evaluation using `ndcg_score` for ranking effectiveness

### 🤖 Machine Learning (ML)
- **Supervised Learning Models**:
  - `RandomForestRegressor` for predicting numerical stress levels
  - `RandomForestClassifier` for categorical stress category prediction
- **Feature Engineering**:
  - Gender encoding, BP parsing, health metric derivation
- **Model Evaluation**:
  - Mean Absolute Error (MAE)
  - R², Accuracy, Confusion Matrix, Classification Report
- **Fallback Handling**:
  - Uses `DummyRegressor`/`DummyClassifier` for robustness if data is missing

---

## 📂 Dataset References

- **FAQ Dataset**: `Mental_Health_FAQ.csv` — pre-defined mental health questions and answers.
- **User Lifestyle Dataset**: `Sleep_health_and_lifestyle_dataset.csv` — includes fields like `Age`, `Heart Rate`, `Blood Pressure`, etc.

---

## 🛠️ Tech Stack

- **Backend**: Python, Flask
- **Libraries**: `sklearn`, `nltk`, `matplotlib`, `sentence-transformers`, `rank_bm25`, `pandas`, `numpy`
- **Deployment-ready**: Suitable for web hosting platforms like Heroku, Render, etc.

---

## 📸 Sample Screenshots (Add if available)

- Chatbot interface showing emergency detection
- Visual dashboards (Radar chart, Stress vs Age, Bar plots)

---

## ✅ How to Run

```bash
git clone https://github.com/Cyberpunk-San/IR.git
cd IR
cd mental-health
pip install -r requirements.txt
python app.py  # Flask entry point
