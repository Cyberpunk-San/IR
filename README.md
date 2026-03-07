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
<img src="https://github.com/user-attachments/assets/ae9dc69a-5a83-46c4-8700-28714ae85535" width="200"/>
<img src="https://github.com/user-attachments/assets/d45b0aab-6317-4689-a637-262994cb9d67" width="600"/>
<img src="https://github.com/user-attachments/assets/7bc86a58-78a7-4b69-b3d4-015b70f5f5a7" width="400"/>
<img src="https://github.com/user-attachments/assets/9719a01a-e794-4eaa-bfee-2cb836170783" width="400"/>
<img src="https://github.com/user-attachments/assets/3620a070-2339-49db-833e-7ab42e7bc77b" width="250"/>
<img src="https://github.com/user-attachments/assets/d9a05e79-d5b9-4f0b-bfc2-2cbea99e42c2" width="300"/>
<img src="https://github.com/user-attachments/assets/ec07c56c-f452-44ba-8984-657ba6d82f6a" width="250"/>


---

## ✅ How to Run

```bash
git clone https://github.com/Cyberpunk-San/IR.git
cd IR
cd mental-health
pip install -r requirements.txt
python app.py  # Flask entry point
