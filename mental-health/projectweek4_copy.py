import os
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
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import re
warnings.filterwarnings('ignore')

class MentalHealthChatbot:
    def __init__(self, use_generation=True):
        self.emergency_keywords = [
            'suicide', 'kill myself', 'end my life', 'want to die',
            'murder', 'harm others', 'hurt someone', 'shoot',
            'self-harm', 'cutting', 'overdose', 'jump off'
        ]
        
        self.use_generation = use_generation
        self.generation_model = None
        self.tokenizer = None
        
        self.faq_df = self._prepare_faq_data()
        self._setup_retrieval_models()
        self.user_feedback = []
        
        if self.use_generation:
            self._initialize_generation_model()
        
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
    
    def get_model_info(self):
        model_info = {
            'rag_enabled': self.use_generation,
            'retrieval_models': {
                'sparse': ['TF-IDF', 'BM25'],
                'dense': ['Sentence-BERT (all-MiniLM-L6-v2)'],
                'weights': {'BM25': 0.4, 'TF-IDF': 0.1, 'BERT': 0.5}
            }
        }
        
        if self.use_generation and self.generation_model is not None:
            model_info['generation_model'] = {
                'name': 'google/flan-t5-small',
                'type': 'Encoder-Decoder Transformer',
                'parameters': '80M',
                'max_length': 512,
                'training': 'Fine-tuned on instruction datasets',
                'source': 'Hugging Face (Apache 2.0 license)'
            }
        else:
            model_info['generation_model'] = None
        
        return model_info
    
    def _initialize_generation_model(self):
        try:
            print("Loading generation model (FLAN-T5-small)...")
            model_name = "google/flan-t5-small"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.generation_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            
            self.generator = pipeline(
                "text2text-generation",
                model=self.generation_model,
                tokenizer=self.tokenizer,
                max_length=300,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                truncation=True
            )
            print("Generation model loaded successfully!")
        except Exception as e:
            print(f"Could not load generation model: {str(e)}")
            print("   Falling back to retrieval-only mode")
            self.use_generation = False
    
    def _check_emergency(self, text):
        text_lower = text.lower()
        for keyword in self.emergency_keywords:
            if keyword in text_lower:
                return self._get_india_emergency_response()
        return None
    
    def _get_india_emergency_response(self):
        return {
            'type': 'emergency',
            'content': (
                "CRISIS ALERT: You're not alone. Immediate help in India:\n"
                f"• Vandrevala Foundation: {self.local_resources['India']['helpline']} (24/7)\n"
                f"• Crisis Text: 'HOME' to {self.local_resources['India']['text']}\n"
                f"• Emergency: Dial {self.local_resources['India']['emergency']}\n"
                f"• Additional Help:\n   - {self.local_resources['India']['additional'][0]}\n"
                f"   - {self.local_resources['India']['additional'][1]}\n\n"
            ),
            'emergency': True
        }

    def _prepare_faq_data(self):
        df = pd.read_csv('Mental_Health_FAQ.csv')[['Questions', 'Answers']]
        df.drop_duplicates(inplace=True)
        df['processed'] = df['Questions'].apply(self._advanced_preprocess)
        return df
    
    def _advanced_preprocess(self, text):
        tokens = word_tokenize(text.lower())
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))
        
        mental_health_terms = {
            'depressed': 'depression',
            'anxious': 'anxiety',
            'stress': 'stressed',
            'sad': 'depression',
            'worried': 'anxiety',
            'fear': 'anxiety'
        }
        
        processed_tokens = []
        for word in tokens:
            if word.isalpha() and word not in stop_words:
                word = mental_health_terms.get(word, word)
                processed_tokens.append(lemmatizer.lemmatize(word))
        return " ".join(processed_tokens)
    
    def _setup_retrieval_models(self):
        self.tfidf_vectorizer = TfidfVectorizer()
        self.tfidf_vectors = self.tfidf_vectorizer.fit_transform(self.faq_df['processed'])
        
        tokenized_faqs = [q.split() for q in self.faq_df['processed']]
        self.bm25 = BM25Okapi(tokenized_faqs)
        
        self.faq_df['retrieval_text'] = self.faq_df['Questions'] + " " + self.faq_df['Answers'].str.slice(0, 200)
        self.bert_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.bert_embeddings = self.bert_model.encode(
            self.faq_df['retrieval_text'].tolist(),
            show_progress_bar=False
        )
        
        try:
            if os.path.exists('user_feedback.csv'):
                self.user_feedback = pd.read_csv('user_feedback.csv').to_dict('records')
        except Exception as e:
            print(f"Error loading feedback: {str(e)}")
    
    def retrieve_relevant_context(self, question, top_k=5):
        processed_query = self._advanced_preprocess(question)
        
        tfidf_scores = cosine_similarity(
            self.tfidf_vectorizer.transform([processed_query]),
            self.tfidf_vectors
        ).flatten()
        
        bm25_scores = self.bm25.get_scores(processed_query.split())
        
        bert_scores = cosine_similarity(
            self.bert_model.encode([question]),
            self.bert_embeddings
        ).flatten()
        
        tfidf_norm = (tfidf_scores - tfidf_scores.min()) / (tfidf_scores.max() - tfidf_scores.min() + 1e-8)
        bm25_norm = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-8)
        
        combined_scores = 0.4*bm25_norm + 0.1*tfidf_norm + 0.5*bert_scores
        
        top_indices = np.argsort(combined_scores)[-top_k:][::-1]
        
        min_score = 0.3
        context_items = []
        for idx in top_indices:
            if combined_scores[idx] > min_score:
                context_items.append({
                    'question': self.faq_df.iloc[idx]['Questions'],
                    'answer': self.faq_df.iloc[idx]['Answers'],
                    'score': combined_scores[idx]
                })
        
        return context_items, combined_scores
    
    def generate_response(self, question, context_items):
        if not self.use_generation or not context_items:
            return None
        
        MAX_TOKENS = 450
        
        def estimate_tokens(text):
            return len(text) // 4
        
        base_prompt = f"Question: {question}\nAnswer:"
        current_tokens = estimate_tokens(base_prompt)
        
        context_parts = []
        for i, item in enumerate(context_items[:3]):
            answer = item['answer']
            if len(answer) > 500:
                answer = answer[:500] + "..."
                
            context_text = f"FAQ {i+1}: {item['question']} - {answer}\n"
            context_tokens = estimate_tokens(context_text)
            
            if current_tokens + context_tokens < MAX_TOKENS:
                context_parts.append(context_text)
                current_tokens += context_tokens
            else:
                remaining = MAX_TOKENS - current_tokens
                max_chars = remaining * 4
                if max_chars > 50:
                    truncated = context_text[:max_chars] + "..."
                    context_parts.append(truncated)
                break
        
        context_str = "".join(context_parts)
        prompt = (
            "Answer the following mental health question using the provided context. "
            "Provide a detailed, empathetic, and complete response based on the FAQs.\n\n"
            f"Context:\n{context_str}\n"
            f"Question: {question}\n\n"
            "Detailed Answer:"
        )
        
        try:
            generated = self.generator(
                prompt,
                max_length=250,
                do_sample=True,
                temperature=0.7,
                repetition_penalty=1.2,
                truncation=True
            )[0]['generated_text']
            
            return generated.strip()
        except Exception as e:
            print(f"Generation error: {str(e)}")
            return None
    
    def get_answer(self, question, top_k=5):
        emergency_response = self._check_emergency(question)
        if emergency_response:
            self._log_interaction(question, "EMERGENCY_TRIGGERED_INDIA")
            return emergency_response
        
        context_items, all_scores = self.retrieve_relevant_context(question, top_k=top_k)
        
        if context_items and context_items[0]['score'] > 0.85:
            response = {
                'type': 'retrieved',
                'content': context_items[0]['answer'],
                'question_matched': context_items[0]['question'],
                'confidence': float(context_items[0]['score'])
            }
            self._log_interaction(question, response['content'])
            return response

        if self.use_generation and context_items:
            generated_response = self.generate_response(question, context_items)
            
            is_echo = generated_response and question.lower() in generated_response.lower()
            
            if generated_response and len(generated_response) > 5 and not is_echo:
                self._log_interaction(question, generated_response)
                
                return {
                    'type': 'generated',
                    'content': generated_response,
                    'context_used': [
                        {
                            'question': item['question'],
                            'confidence': float(item['score'])
                        } for item in context_items[:3]
                    ],
                    'confidence': float(context_items[0]['score'])
                }
        
        if context_items:
            best_score = context_items[0]['score']
            if best_score > 0.45:
                response = {
                    'type': 'retrieved',
                    'content': context_items[0]['answer'],
                    'question_matched': context_items[0]['question'],
                    'confidence': float(best_score)
                }
            elif best_score > 0.3:
                response = {
                    'type': 'retrieved',
                    'content': f"I'm here to listen. While I don't have a direct answer in my database, here is some related information that might help: {context_items[0]['answer']}",
                    'question_matched': context_items[0]['question'],
                    'confidence': float(best_score),
                    'low_confidence': True
                }
            else:
                response = self._get_default_fallback()
        else:
            response = self._get_default_fallback()
            
        self._log_interaction(question, response['content'])
        return response

    def _get_default_fallback(self):
        return {
            'type': 'retrieved',
            'content': (
                "I'm sorry you're feeling this way. I don't have specific information on that "
                "in my FAQ database, but I'm here to listen. It might help to rephrase your "
                "question, or if you're feeling very overwhelmed, please consider reaching out "
                "to one of the helplines mentioned in the emergency section."
            ),
            'confidence': 0.0
        }
    
    def _log_interaction(self, query, selected_answer):
        self.user_feedback.append({
            'query': query,
            'answer': selected_answer if isinstance(selected_answer, str) else selected_answer.get('content', ''),
            'timestamp': pd.Timestamp.now()
        })
    
    def plot_stress_distribution(self, fig=None, data_path='Sleep_health_and_lifestyle_dataset.csv'):
        try:
            df = pd.read_csv(data_path)
            if 'Stress Level' not in df.columns:
                print("Error: 'Stress Level' column not found in dataset")
                return None
            
            if fig is None:
                fig = plt.figure(figsize=(10, 6))
            
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
        
    def chat(self):
        while True:
            print("\n" + "="*60)
            print(" MANN-O-METER MENTAL HEALTH CHATBOT ".center(60, '#'))
            print("="*60)
            
            model_info = self.get_model_info()
            print("\nMODEL ARCHITECTURE:")
            print(f"   • Retrieval: Hybrid ({', '.join(model_info['retrieval_models']['sparse'])} + {model_info['retrieval_models']['dense'][0]})")
            print(f"   • Weights: BM25={model_info['retrieval_models']['weights']['BM25']}, "
                  f"TF-IDF={model_info['retrieval_models']['weights']['TF-IDF']}, "
                  f"BERT={model_info['retrieval_models']['weights']['BERT']}")
            
            if model_info['generation_model']:
                gen = model_info['generation_model']
                print(f"\nGENERATION MODEL: {gen['name']}")
                print(f"   • Type: {gen['type']}")
                print(f"   • Parameters: {gen['parameters']}")
                print(f"   • Context Window: {gen['max_length']} tokens")
                print(f"   • License: {gen['source']}")
            else:
                print("\nMODE: Retrieval-only (no generation)")
            
            print("="*60)
            print("\nOptions:")
            print("1. Ask a mental health question")
            print("2. View stress visualizations")
            print("3. Provide feedback on answers")
            print("4. Toggle RAG mode (currently: {})".format("ON" if self.use_generation else "OFF"))
            print("5. Exit")
            
            choice = input("\nEnter your choice (1-5): ").strip()
            
            if choice == '1':
                question = input("\nWhat's your mental health question? ")
                print("\nProcessing your question...")
                response = self.get_answer(question)
                
                if isinstance(response, dict):
                    if response.get('type') == 'emergency':
                        print("\nCRISIS SUPPORT:")
                        print(response['content'])
                    elif response.get('type') == 'generated':
                        print("\nAI-GENERATED RESPONSE:")
                        print(response['content'])
                        print(f"\nConfidence: {response['confidence']:.2f}")
                        print("Based on:")
                        for ctx in response.get('context_used', []):
                            print(f"  • {ctx['question']} (conf: {ctx['confidence']:.2f})")
                    elif response.get('type') == 'retrieved':
                        print("\nFAQ RESPONSE:")
                        print(response['content'])
                        print(f"\nMatched question: {response['question_matched']}")
                        print(f"   Confidence: {response['confidence']:.2f}")
                    else:
                        print("\nAssistant:", response['content'])
                else:
                    print("\nAssistant:", response)
            
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
                        plt.show()
                    elif viz_choice == '2':
                        print("\nGenerating stress vs age analysis...")
                        self.plot_stress_vs_age()
                        plt.show()
                    elif viz_choice == '3':
                        break
                    else:
                        print("Please enter a number between 1-3")
            
            elif choice == '3':
                self._collect_feedback()
            
            elif choice == '4':
                self.use_generation = not self.use_generation
                if self.use_generation and self.generation_model is None:
                    self._initialize_generation_model()
                print(f"\nRAG mode toggled to: {'ON' if self.use_generation else 'OFF'}")
            
            elif choice == '5':
                print("\nThank you for using Mann-o-meter Mental Health Assistant!")
                if self.user_feedback:
                    pd.DataFrame(self.user_feedback).to_csv('user_feedback.csv', index=False)
                break
            
            else:
                print("Please enter a number between 1-5")
    
    def _collect_feedback(self):
        if not self.user_feedback:
            print("\nNo recent interactions to provide feedback on.")
            return
        
        print("\nYour recent interactions:")
        for i, interaction in enumerate(self.user_feedback[-3:], 1):
            answer_text = interaction['answer'][:50] + "..." if len(interaction['answer']) > 50 else interaction['answer']
            print(f"{i}. Q: {interaction['query'][:50]}...\n   A: {answer_text}")
        
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

if __name__ == "__main__":
    print("Initializing Mann-o-meter RAG-Enhanced Mental Health Assistant...")
    try:
        chatbot = MentalHealthChatbot(use_generation=True)
        chatbot.chat()
    except Exception as e:
        print(f"Error during initialization: {str(e)}")
        print("Please ensure all data files are in the correct location.")