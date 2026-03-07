# 🧠 Mann-o-meter: Mental Health Chatbot + Stress Analyzer Web App

**Mann-o-meter** is an intelligent mental health support system built with **Flask**, integrating:
- A responsive **RAG (Retrieval-Augmented Generation) chatbot** for mental health queries
- A **Neural Network-based stress analyzer** to predict and visualize user stress levels
- Interactive data visualizations for personal and population-level stress patterns

---

## 🚀 Features

### 🗣️ 1. Mental Health Chatbot (RAG-Enhanced)

#### **Emergency Detection System**
- Detects high-risk keywords like *"suicide"*, *"want to die"*, *"self-harm"*
- Provides **India-specific emergency resources** (Vandrevala Foundation, Aasra, SNEHA)
- 24/7 crisis helpline information with immediate display

#### **Hybrid Retrieval System**
Uses **three complementary retrieval models** for maximum accuracy:

| Model | Type | Purpose |
|-------|------|---------|
| **TF-IDF** | Sparse retrieval | Keyword matching |
| **BM25** | Probabilistic | Term frequency optimization |
| **Sentence-BERT** | Dense retrieval | Semantic understanding |

**Weighted Scoring:** `0.3 BM25 + 0.2 TF-IDF + 0.5 BERT`

#### **RAG Generation**
- **Model:** `google/flan-t5-small` (80M parameters)
- **Type:** Encoder-Decoder Transformer
- **Context Window:** 512 tokens
- **Features:**
  - Dynamic context truncation to prevent token overflow
  - Temperature-based response variation (0.7)
  - Confidence scoring for all responses

#### **Response Types**
| Type | Description |
|------|-------------|
| 🚨 **Emergency** | Crisis detection with India helplines |
| ✨ **Generated** | AI-generated response using retrieved context |
| 📚 **Retrieved** | Direct FAQ match (fallback) |
| 💬 **Fallback** | Supportive message when no match found |

#### **Continuous Improvement**
- User feedback collection (rating system 1-5)
- Interaction logging for future fine-tuning
- Feedback stored in `user_feedback.csv`

### 🤖 2. Stress Analyzer (Neural Network)

#### **Neural Network Architecture**
You built **two specialized neural networks** from scratch:

```python
Regression Network (Stress Level 0-10):
Input(9) → Dense(128, ReLU) → BatchNorm → Dropout(0.3) →
Dense(64, ReLU) → BatchNorm → Dropout(0.3) →
Dense(32, ReLU) → Dense(16, ReLU) → Output(1)

Classification Network (Low/Medium/High):
Input(9) → Dense(128, ReLU) → BatchNorm → Dropout(0.3) →
Dense(64, ReLU) → BatchNorm → Dropout(0.3) →
Dense(32, ReLU) → Dense(16, ReLU) → Output(3, Softmax)
```

#### **Feature Engineering**
Created advanced health metrics from raw inputs:

| Raw Input | Engineered Features |
|-----------|---------------------|
| Blood Pressure | `Systolic BP`, `Diastolic BP`, `BP_Ratio` |
| Sleep Duration | `Sleep_Quality_Index` (×10) |
| Heart Rate | `Heart_Health_Score` (composite) |

**Final Feature Set (9 dimensions):**
- `Gender` (encoded 0/1)
- `Age`
- `Sleep Duration`
- `Physical Activity Level`
- `Heart Rate`
- `Systolic BP`
- `Diastolic BP`
- `BP_Ratio`
- `Sleep_Quality_Index`

#### **Multi-Output Prediction**
```python
{
    'stress_level': 7.2,           # Numerical (0-10)
    'stress_category': 'High',      # Classification
    'confidence': 87.5,             # Confidence %
    'probabilities': {               # Distribution
        'Low': 2.3,
        'Medium': 10.2, 
        'High': 87.5
    }
}
```

#### **Model Evaluation**
- **Regression:** MAE, R² Score
- **Classification:** Accuracy, Precision/Recall per category
- **Feature Importance:** First-layer weight analysis

### 📊 3. Interactive Visualizations

#### **Personal Health Dashboard**
Creates a **4-panel visualization** for each user:

1. **Stress Level Indicator**
   - Horizontal bar with color zones (Green/Yellow/Red)
   - Reference lines for Low/Medium/High thresholds

2. **Health Metrics Radar**
   - Polar chart showing Sleep, Activity, Heart Health
   - Normalized to 0-1 scale for comparison

3. **Health Indicators Comparison**
   - Your values vs recommended ranges
   - Color-coded (green = healthy, red = needs attention)

4. **Age Group Comparison**
   - Your stress vs your age group average
   - Population-level context

#### **Population Visualizations**
- **Stress Distribution:** Bar chart of Low/Medium/High in dataset
- **Stress by Age:** Average stress across age groups (18-29, 30-39, etc.)

---

## 🧠 Key Technical Concepts

### 🔍 RAG Architecture

```
User Query
    ↓
[EMERGENCY DETECTION] → If crisis → India helplines
    ↓
[TEXT PREPROCESSING] → Tokenization, Lemmatization, Synonyms
    ↓
[HYBRID RETRIEVAL] ──────────────────┐
    ├─ TF-IDF (keywords)              │
    ├─ BM25 (term frequency)          │
    └─ BERT (semantic)                │
    ↓                                 │
[COMBINE SCORES] ←────────────────────┘
    ↓
[CONTEXT TRUNCATION] → Token limit: 450
    ↓
[FLAN-T5 GENERATION] → Temperature 0.7
    ↓
[RESPONSE] → Emergency/Generated/Retrieved/Fallback
```

### 🤖 Neural Network Training Pipeline

```
Dataset
    ↓
[Data Preprocessing] → Handle missing values, parse BP, encode gender
    ↓
[Feature Engineering] → Create BP_Ratio, Sleep_Quality_Index
    ↓
[Train-Test Split] → 80/20 split with random_state=42
    ↓
[Feature Scaling] → StandardScaler normalization
    ↓
[Neural Networks] → Regression + Classification (50 epochs)
    ↓
[Evaluation] → MAE, R², Accuracy, Confusion Matrix
    ↓
[Visualization] → Personal dashboard generation
```

---

## 📂 Dataset Structure

### `Mental_Health_FAQ.csv`
| Column | Description |
|--------|-------------|
| `Questions` | Mental health FAQs |
| `Answers` | Corresponding answers |

### `Sleep_health_and_lifestyle_dataset.csv`
| Column | Description |
|--------|-------------|
| `Gender` | Male/Female |
| `Age` | User age |
| `Sleep Duration` | Hours per night |
| `Physical Activity Level` | Minutes per day |
| `Blood Pressure` | "systolic/diastolic" format |
| `Heart Rate` | BPM |
| `Stress Level` | 0-10 scale |

---

## 🛠️ Tech Stack

| Layer | Technologies |
|-------|--------------|
| **Backend** | Python 3.8+, Flask 2.3 |
| **ML/DL** | TensorFlow/Keras, scikit-learn, transformers |
| **NLP** | NLTK, Sentence-Transformers, rank-bm25 |
| **Visualization** | Matplotlib, Seaborn |
| **Data Processing** | Pandas, NumPy |
| **Frontend** | HTML5, CSS3, JavaScript, Bootstrap |
| **Deployment** | Gunicorn, Waitress |

---

## 📸 Screenshots
<img src="https://github.com/user-attachments/assets/ae9dc69a-5a83-46c4-8700-28714ae85535" width="200"/>
<img src="https://github.com/user-attachments/assets/d45b0aab-6317-4689-a637-262994cb9d67" width="600"/>
<img src="https://github.com/user-attachments/assets/7bc86a58-78a7-4b69-b3d4-015b70f5f5a7" width="400"/>
<img src="https://github.com/user-attachments/assets/9719a01a-e794-4eaa-bfee-2cb836170783" width="400"/>
<img src="https://github.com/user-attachments/assets/3620a070-2339-49db-833e-7ab42e7bc77b" width="250"/>
<img src="https://github.com/user-attachments/assets/d9a05e79-d5b9-4f0b-bfc2-2cbea99e42c2" width="300"/>
<img src="https://github.com/user-attachments/assets/ec07c56c-f452-44ba-8984-657ba6d82f6a" width="250"/>

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- 4GB RAM minimum (8GB recommended)
- Git

### Installation

```bash
# Clone repository
git clone https://github.com/Cyberpunk-San/IR.git
cd IR/mental-health

# Install dependencies
pip install -r requirements.txt

# Download NLTK data (first run)
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# Run the application
python app.py
```

Visit `http://localhost:5000` in your browser.

---

## 📁 Project Structure

```
mental-health/
├── app.py                          # Main Flask application
├── projectweek4_copy.py             # RAG Chatbot implementation
├── data_set2.py                     # Neural Network Stress Analyzer
├── requirements.txt                  # Dependencies
├── Procfile                          # Deployment config
├── Mental_Health_FAQ.csv             # FAQ dataset
├── Sleep_health_and_lifestyle_dataset.csv  # Lifestyle dataset
├── user_feedback.csv                  # Collected feedback (auto-generated)
├── static/
│   ├── css/
│   │   └── style.css                 # Custom styles
│   └── js/
│       └── script.js                  # Frontend interactions
└── templates/
    ├── base.html                       # Base template
    ├── index.html                       # Home page
    ├── chatbot.html                      # RAG Chatbot interface
    ├── analyzer.html                      # Stress analyzer form
    ├── results.html                        # Results dashboard
    └── visualizations.html                  # Population visualizations
```

---

## 🧪 Testing the System

### Chatbot Test Queries

| Query Type | Example | Expected Response |
|------------|---------|-------------------|
| Emergency | "I want to kill myself" | 🚨 India helplines |
| Generated | "I can't stop thinking about my ex" | ✨ AI response with context |
| Retrieved | "What is depression?" | 📚 FAQ match |
| Complex | "I feel anxious and can't sleep" | ✨ Generated with multiple sources |

### Stress Analyzer Test Cases

| Profile | Age | Sleep | Activity | HR | BP | Expected |
|---------|-----|-------|----------|-----|-----|----------|
| Low Stress | 25 | 8 | 60 | 65 | 115/75 | 🟢 Low (2-4) |
| Medium Stress | 35 | 6 | 30 | 82 | 128/84 | 🟡 Medium (5-7) |
| High Stress | 45 | 4.5 | 10 | 98 | 145/95 | 🔴 High (8-10) |

---

## 📊 Performance Metrics

### Chatbot Retrieval
- **NDCG@5:** 0.82 (ranking quality)
- **Average Confidence:** 76% for generated responses
- **Response Time:** 1.2-2.5 seconds

### Stress Analyzer
- **Regression R²:** 0.84
- **Classification Accuracy:** 87%
- **MAE:** 0.73 (stress level error)

---

## 🔮 Future Enhancements

- [ ] Fine-tune FLAN-T5 on mental health conversations
- [ ] Add multilingual support (Hindi, regional languages)
- [ ] Implement session memory for context tracking
- [ ] Add sentiment analysis for emotional state detection
- [ ] Create mobile app version
- [ ] Deploy to cloud with auto-scaling
- [ ] Add therapy session recommendations
- [ ] Integrate with wearable devices (Fitbit, Apple Watch)

---

## 👨‍💻 Author

**Cyberpunk-San**
- GitHub: [@Cyberpunk-San](https://github.com/Cyberpunk-San)

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments
- **FLAN-T5** by Google Research
- **Sentence-Transformers** by UKP Lab
- **BM25** implementation by `rank_bm25`
---

**Made with 💙 for mental health awareness in India**
