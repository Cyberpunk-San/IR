# 🧠 Mann-o-meter: Intelligent Mental Health Assistant

**Mann-o-meter** is a comprehensive mental health support ecosystem that bridges the gap between data-driven research and interactive AI support. It combines rigorous exploratory data analysis (EDA) and optimized neural networks with a production-ready **Retrieval-Augmented Generation (RAG)** chatbot.

---

## 🚀 Key Features

### 🗣️ 1. RAG-Enhanced Mental Health Chatbot
An empathetic assistant powered by a hybrid retrieval system and the **FLAN-T5** model.
*   **Emergency Detection**: Real-time monitoring for high-risk keywords with immediate India-specific crisis resources.
*   **Hybrid Retrieval Engine**: Combines sparse and dense vectors for precision:
    *   **TF-IDF** (10% weight): Keyword matching.
    *   **BM25** (40% weight): Probabilistic term frequency.
    *   **Sentence-BERT** (50% weight): Semantic understanding via `all-MiniLM-L6-v2`.
*   **Contextual Generation**: Generates detailed, empathetic responses using `google/flan-t5-small` based on retrieved FAQ context.

### 🤖 2. Neural Network Stress Analyzer
Predicts user stress levels based on physiological and lifestyle parameters.
*   **Engineered Features**: Utilizes advanced metrics like `BP_Ratio`, `Sleep_Quality_Index`, and `Heart_Health_Score`.
*   **Intelligent Prediction**: Uses a deep Neural Network to predict precise stress scores (0-10).
*   **Heuristic Categorization**: Implements logic-based classification (Low, Medium, High) derived from the regression output.
*   **Optimization**: Hyperparameters tuned using **Optuna** for maximum accuracy.

### 📊 3. Interactive Insights & Dashboards
*   **Personal Dashboard**: A 4-panel visualization suite (Radar charts, comparison bars) provided instantly after analysis.
*   **Population Analytics**: Visualizes global trends, such as stress distribution and age-based stress averages.

---

## 🏗️ Project Lifecycle & Research

The project followed a structured development pipeline, documented in the root-level notebooks:

1.  **Exploratory Data Analysis (`eda.ipynb`)**:
    *   Deep dive into lifestyle factors (Sleep, Activity, BMI).
    *   Correlation analysis between physical metrics and mental stress.
2.  **Data Preprocessing (`preprocessed.ipynb`)**:
    *   Handling missing values in complex datasets.
    *   Feature engineering and normalization for neural network readiness.
3.  **Model Prototyping (`TestingonMLmodels (1).ipynb`)**:
    *   Comparative analysis of Random Forest, XGBoost, and SVM.
4.  **Neural Network Optimization (`NeuralNetwork.ipynb`)**:
    *   Automated hyperparameter tuning with Optuna.
    *   Final architecture design and saving models (`.keras`, `.h5`).

---

## 🛠️ Technical Stack

| Category | Tools & Libraries |
| :--- | :--- |
| **Backend** | Python 3.8+, Flask 2.3 |
| **Deep Learning** | TensorFlow/Keras, Optuna |
| **NLP & RAG** | Transformers (HuggingFace), Sentence-Transformers, NLTK, rank-bm25 |
| **Data Analysis** | Pandas, NumPy, Scikit-learn |
| **Visualization** | Matplotlib, Seaborn |
| **Frontend** | HTML5, CSS3 (Glassmorphism), JavaScript, Bootstrap 5 |

---

## 📁 Project Structure

```text
IR/
├── mental-health/              # Production Web Application
│   ├── app.py                  # Flask Main Entry Point
│   ├── projectweek4_copy.py    # RAG Chatbot Logic
│   ├── data_set2.py            # Stress Analyzer Logic
│   ├── static/                 # CSS, JS, and Images
│   ├── templates/              # HTML Templates (Jinja2)
│   ├── stress_model.keras      # Pre-trained Neural Network
│   └── requirements.txt        # Web App Dependencies
├── eda.ipynb                   # Phase 1: Exploratory Data Analysis
├── preprocessed.ipynb          # Phase 2: Feature Engineering
├── NeuralNetwork.ipynb         # Phase 3: Model Training & Tuning
├── stressdata.csv              # Raw Dataset
├── stressdata_preprocessed.csv # Processed ML-Ready Dataset
└── README.md                   # Project Documentation
```

---

## 🚀 Getting Started

### Prerequisites
*   Python 3.8 or higher
*   Minimum 4GB RAM (8GB recommended for transformer models)

### Installation & Execution

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/Cyberpunk-San/IR.git
    cd IR/mental-health
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Download Required NLP Data**:
    ```bash
    python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
    ```

4.  **Launch the Application**:
    ```bash
    python app.py
    ```
    Access the platform at `http://127.0.0.1:5000`.

---

## 📊 Model Performance

| Metric | RAG Chatbot | Stress Analyzer (NN) |
| :--- | :--- | :--- |
| **Confidence / Accuracy** | 76% Avg. Confidence | 0.84 R² Score (Regression) |
| **Retrieval Quality** | 0.82 NDCG@5 | 0.73 MAE |
| **Latent Response Time** | ~1.5s | <100ms |

---

## 🔮 Future Roadmap

- [ ] Fine-tuning FLAN-T5 on domain-specific medical journals.
- [ ] Integration with wearable IoT devices (Apple Health/Google Fit).
- [ ] Multilingual support for regional Indian languages.
- [ ] Advanced Sentiment analysis for dynamic bot empathy levels.

---

**Developed with 💙 by Cyberpunk-San**
*Empowering mental wellness through data and AI.*
