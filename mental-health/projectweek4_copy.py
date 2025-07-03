import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import warnings
from sklearn.metrics import ndcg_score
warnings.filterwarnings('ignore')

class MentalHealthChatbot:
    def __init__(self):
        # Initialize emergency keywords
        self.emergency_keywords = [
            'suicide', 'kill myself', 'end my life', 'want to die',
            'murder', 'harm others', 'hurt someone', 'shoot',
            'self-harm', 'cutting', 'overdose', 'jump off'
        ]
        
        # Initialize FAQ system with advanced retrieval
        self.faq_df = self._prepare_faq_data()
        self._setup_retrieval_models()
        self.user_feedback = []
        
        # Localized emergency resources (now includes India)
        self.local_resources = {
            'India': {
                'helpline': '9152987821',  # Vandrevala Foundation
                'text': '85258',          # Crisis Text Line India
                'emergency': '112',        # National Emergency Number
                'additional': [
                    '044-24640050 (SNEHA Foundation)',
                    '022-25521111 (Aasra)'
                ]
            },
            'US': {'helpline': '988', 'text': '741741', 'emergency': '911'},
            'UK': {'helpline': '116123', 'emergency': '999'}
        }
    
    def _check_emergency(self, text):
        """Check for emergency keywords with India-specific response"""
        text_lower = text.lower()
        for keyword in self.emergency_keywords:
            if keyword in text_lower:
                return self._get_india_emergency_response()
        return None
    
    def _get_india_emergency_response(self):
        """Generate India-specific crisis response"""
        return (
            "\n🚨 CRISIS ALERT: You're not alone. Immediate help in India:\n"
            f"• Vandrevala Foundation: {self.local_resources['India']['helpline']} (24/7)\n"
            f"• Crisis Text: 'HOME' to {self.local_resources['India']['text']}\n"
            f"• Emergency: Dial {self.local_resources['India']['emergency']}\n"
            f"• Additional Help:\n   - {self.local_resources['India']['additional'][0]}\n"
            f"   - {self.local_resources['India']['additional'][1]}\n\n"
        )

    # ==================== FAQ SYSTEM ====================
    def _prepare_faq_data(self):
        """Load and preprocess FAQ data with enhanced cleaning"""
        df = pd.read_csv('Mental_Health_FAQ.csv')[['Questions', 'Answers']]
        df.drop_duplicates(inplace=True)
        df['processed'] = df['Questions'].apply(self._advanced_preprocess)
        return df
    
    def _advanced_preprocess(self, text):
        """Enhanced text processing pipeline"""
        tokens = word_tokenize(text.lower())
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))
        
        # Handle mental health specific terms
        mental_health_terms = {
            'depressed': 'depression',
            'anxious': 'anxiety',
            'stress': 'stressed'
        }
        
        processed_tokens = []
        for word in tokens:
            if word.isalpha() and word not in stop_words:
                word = mental_health_terms.get(word, word)  # Normalize synonyms
                processed_tokens.append(lemmatizer.lemmatize(word))
        return " ".join(processed_tokens)
    
    def _setup_retrieval_models(self):
        """Initialize multiple retrieval models"""
        # TF-IDF
        self.tfidf_vectorizer = TfidfVectorizer()
        self.tfidf_vectors = self.tfidf_vectorizer.fit_transform(self.faq_df['processed'])
        
        # BM25
        tokenized_faqs = [q.split() for q in self.faq_df['processed']]
        self.bm25 = BM25Okapi(tokenized_faqs)
        
        # Sentence-BERT
        self.bert_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.bert_embeddings = self.bert_model.encode(self.faq_df['Questions'])
        try:
            if os.path.exists('user_feedback.csv'):
                self.user_feedback = pd.read_csv('user_feedback.csv').to_dict('records')
        except Exception as e:
            print(f"Error loading feedback: {str(e)}")
    
    def get_answer(self, question, top_k=3):
        """First check for emergency keywords"""
        emergency_response = self._check_emergency(question)
        if emergency_response:
            self._log_interaction(question, "EMERGENCY_TRIGGERED_INDIA")
            return emergency_response
        
        """Hybrid retrieval with ranking and NDCG evaluation"""
        processed_query = self._advanced_preprocess(question)
        
        # Get scores from all models
        tfidf_scores = cosine_similarity(
            self.tfidf_vectorizer.transform([processed_query]),
            self.tfidf_vectors
        ).flatten()
        
        bm25_scores = self.bm25.get_scores(processed_query.split())
        
        bert_scores = cosine_similarity(
            self.bert_model.encode([question]),
            self.bert_embeddings
        ).flatten()
        
        # Combine scores (adjust weights as needed)
        combined_scores = 0.4*bm25_scores + 0.3*tfidf_scores + 0.3*bert_scores
        
        # Get top answers
        top_idx = np.argsort(combined_scores)[-top_k:][::-1]
        top_answers = [(self.faq_df.iloc[i]['Answers'], combined_scores[i]) for i in top_idx]
        
        # Calculate NDCG
        ideal_relevance = np.sort(combined_scores)[-top_k:][::-1]
        ndcg = ndcg_score([ideal_relevance], [combined_scores[top_idx]])
        print(f"\nRetrieval Quality (NDCG@{top_k}): {ndcg:.3f}")
        
        # Log interaction
        self._log_interaction(question, top_answers[0][0])
        
        return top_answers[0][0] if combined_scores[top_idx[0]] > 0.5 else "I'm not sure I understand. Could you provide more details?"
    
    def _log_interaction(self, query, selected_answer):
        """Store user interactions for improvement"""
        self.user_feedback.append({
            'query': query,
            'answer': selected_answer,
            'timestamp': pd.Timestamp.now()
        })
    
    # ==================== VISUALIZATION ====================
    def plot_stress_distribution(self, fig=None, data_path='Sleep_health_and_lifestyle_dataset.csv'):
        """Visualize stress distribution from dataset"""
        try:
            df = pd.read_csv(data_path)
            if 'Stress Level' not in df.columns:
                print("Error: 'Stress Level' column not found in dataset")
                return None
            
            if fig is None:
                fig = plt.figure(figsize=(10, 6))
            
            # Categorize stress levels
            df['Stress Category'] = pd.cut(df['Stress Level'],
                                        bins=[0, 3, 6, 10],
                                        labels=['Low', 'Medium', 'High'])
            
            stress_counts = df['Stress Category'].value_counts().sort_index()
            
            ax = fig.add_subplot(111)
            colors = ['#4CAF50', '#FFC107', '#F44336']
            stress_counts.plot(kind='bar', color=colors, ax=ax)
            
            ax.set_title('Stress Level Distribution in Dataset')
            ax.set_xlabel('Stress Level')
            ax.set_ylabel('Count')
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            
            return fig
        except Exception as e:
            print(f"Error generating visualization: {str(e)}")
            return None

    def plot_stress_vs_age(self, fig=None, data_path='Sleep_health_and_lifestyle_dataset.csv'):
        """Visualize stress distribution across age groups"""
        try:
            df = pd.read_csv(data_path)
            if 'Stress Level' not in df.columns or 'Age' not in df.columns:
                print("Error: Required columns not found in dataset")
                return None
            
            if fig is None:
                fig = plt.figure(figsize=(10, 6))
            
            age_bins = [18, 30, 40, 50, 60, 100]
            age_labels = ['18-29', '30-39', '40-49', '50-59', '60+']
            df['Age Group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels)
            
            avg_stress = df.groupby('Age Group', observed=True)['Stress Level'].mean()
            
            ax = fig.add_subplot(111)
            colors = ['#4CAF50', '#8BC34A', '#FFC107', '#FF9800', '#F44336']
            avg_stress.plot(kind='bar', color=colors, edgecolor='black', ax=ax)
            
            ax.set_title('Average Stress Level by Age Group')
            ax.set_xlabel('Age Group')
            ax.set_ylabel('Average Stress (1-10 Scale)')
            ax.set_ylim(0, 10)
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            
            for i, v in enumerate(avg_stress):
                ax.text(i, v + 0.2, f"{v:.1f}", ha='center', fontsize=10)
            
            return fig
        except Exception as e:
            print(f"Error generating visualization: {str(e)}")
            return None
        
    # ==================== INTERACTIVE INTERFACE ====================
    def chat(self):
        """Enhanced chat interface with feedback option"""
        print("\n" + "="*50)
        print(" MENTAL HEALTH CHATBOT ".center(50, '#'))
        print("="*50)
        
        while True:
            print("\nOptions:")
            print("1. Ask a mental health question")
            print("2. View stress visualizations")
            print("3. Provide feedback on answers")
            print("4. Exit")
            
            choice = input("\nEnter your choice (1-4): ").strip()
            
            if choice == '1':
                question = input("\nWhat's your mental health question? ")
                answer = self.get_answer(question)
                print("\nAssistant:", answer)
                
                if "🚨" in answer:
                    follow_up = input().strip().lower()
                    if follow_up == 'yes':
                        print("\nAssistant: Connecting you to Vandrevala Foundation...")
            
            elif choice == '2':
                while True:
                    print("\nVisualization Options:")
                    print("1. Stress distribution in population")
                    print("2. Stress vs Age analysis")
                    print("3. Return to main menu")
                    
                    viz_choice = input("\nChoose visualization (1-3): ").strip()
                    
                    if viz_choice == '1':
                        print("\nGenerating population stress distribution...")
                        self.plot_stress_distribution()
                    elif viz_choice == '2':
                        print("\nGenerating stress vs age analysis...")
                        self.plot_stress_vs_age()
                    elif viz_choice == '3':
                        break
                    else:
                        print("Please enter a number between 1-3")
            
            elif choice == '3':
                self._collect_feedback()
            
            elif choice == '4':
                print("\nThank you for using the Mental Health Assistant!")
                if self.user_feedback:
                    pd.DataFrame(self.user_feedback).to_csv('user_feedback.csv', index=False)
                break
            
            else:
                print("Please enter a number between 1-4")
    
    def _collect_feedback(self):
        """Collect explicit user feedback on answers"""
        if not self.user_feedback:
            print("\nNo recent interactions to provide feedback on.")
            return
        
        print("\nYour recent interactions:")
        for i, interaction in enumerate(self.user_feedback[-3:], 1):
            print(f"{i}. Q: {interaction['query'][:50]}...")
        
        try:
            selection = int(input("\nSelect interaction to rate (1-3): ")) - 1
            if 0 <= selection < len(self.user_feedback[-3:]):
                rating = int(input("Rate this answer (1-5, 5=best): "))
                self.user_feedback[-3:][selection]['rating'] = rating
                print("Thank you for your feedback!")
            else:
                print("Invalid selection.")
        except ValueError:
            print("Please enter a valid number.")

# Run the chatbot
if __name__ == "__main__":
    print("Initializing Mental Health Assistant...")
    try:
        chatbot = MentalHealthChatbot()
        chatbot.chat()
    except Exception as e:
        print(f"Error during initialization: {str(e)}")
        print("Please ensure all data files are in the correct location.")
