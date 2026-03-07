import datetime
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from projectweek4_copy import MentalHealthChatbot
from data_set2 import StressAnalyzer
import os
import pandas as pd
import numpy as np
import base64
from io import BytesIO
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import re
from flask import flash
import matplotlib
matplotlib.use('Agg')
from sklearn.dummy import DummyRegressor, DummyClassifier

app = Flask(__name__)
app.secret_key = os.urandom(24)

chatbot = MentalHealthChatbot(use_generation=True)
analyzer = StressAnalyzer()

chat_history = []

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chatbot', methods=['GET', 'POST'])
def chatbot_interface():
    if request.method == 'POST':
        user_input = request.form.get('user_input')
        
        response = chatbot.get_answer(user_input)
        
        if isinstance(response, dict):
            if response.get('type') == 'emergency':
                formatted_response = {
                    'text': response['content'],
                    'type': 'emergency',
                    'emergency': True
                }
            elif response.get('type') == 'generated':
                formatted_response = {
                    'text': response['content'],
                    'type': 'generated',
                    'confidence': response.get('confidence', 0),
                    'context_used': response.get('context_used', [])
                }
            elif response.get('type') == 'retrieved':
                formatted_response = {
                    'text': response['content'],
                    'type': 'retrieved',
                    'confidence': response.get('confidence', 0),
                    'question_matched': response.get('question_matched', '')
                }
            else:
                formatted_response = {
                    'text': response.get('content', str(response)),
                    'type': 'fallback'
                }
        else:
            formatted_response = {
                'text': str(response),
                'type': 'basic'
            }
        
        chat_history.append({
            'user': user_input,
            'bot': formatted_response['text'],
            'timestamp': datetime.datetime.now().strftime('%H:%M:%S'),
            'type': formatted_response.get('type', 'basic')
        })
        
        return jsonify(formatted_response)
    
    return render_template('chatbot.html', history=chat_history[-20:])

@app.route('/analyzer', methods=['GET', 'POST'])
def stress_analyzer():
    if request.method == 'POST':
        try:
            
            bp = request.form.get('blood_pressure')
            if not re.match(r"^\d{2,3}/\d{2,3}$", bp):
                raise ValueError("Blood pressure must be in format 'systolic/diastolic' (e.g., 120/80)")
            
            
            session['user_data'] = {
                'gender': request.form.get('gender'),
                'age': int(request.form.get('age')),
                'sleep_duration': float(request.form.get('sleep_duration')),
                'activity': int(request.form.get('activity')),
                'heart_rate': int(request.form.get('heart_rate')),
                'blood_pressure': bp
            }
            return redirect(url_for('analysis_results'))
            
        except ValueError as e:
            flash(f"Invalid input: {str(e)}", "danger")
            return redirect(url_for('stress_analyzer'))
            
    return render_template('analyzer.html')

@app.route('/analysis_results')
def analysis_results():
    try:
        if 'user_data' not in session:
            flash('No analysis data found. Please complete the stress analyzer form first.', 'danger')
            return redirect(url_for('stress_analyzer'))
        
        user_data = session['user_data']
        print("\nUser data received:", user_data)

        try:
            if not re.match(r'^\d{2,3}/\d{2,3}$', user_data['blood_pressure']):
                raise ValueError("Invalid blood pressure format")
            systolic, diastolic = map(int, user_data['blood_pressure'].split('/'))
            print("Blood pressure parsed:", systolic, "/", diastolic)
        except Exception as e:
            flash(f'Error processing blood pressure: {str(e)}', 'danger')
            return redirect(url_for('stress_analyzer'))

        input_data = {
            'Gender': 0 if user_data['gender'].lower() == 'male' else 1,
            'Age': user_data['age'],
            'Sleep Duration': user_data['sleep_duration'],
            'Physical Activity Level': user_data['activity'],
            'Heart Rate': user_data['heart_rate'],
            'Systolic BP': systolic,
            'Diastolic BP': diastolic
        }
        print("Input data for prediction:", input_data)

        try:
            prediction_results = analyzer.predict_stress(
                age=user_data['age'],
                sleep_duration=user_data['sleep_duration'],
                activity_level=user_data['activity'],
                heart_rate=user_data['heart_rate'],
                blood_pressure=user_data['blood_pressure'],
                gender=user_data['gender']
            )
            
            stress_level = prediction_results['stress_level']
            stress_category = prediction_results['stress_category']
            confidence = prediction_results.get('confidence', 0)
            probabilities = prediction_results.get('probabilities', {})
            
            img_data = fig_to_base64(prediction_results['visualization'])
            plt.close(prediction_results['visualization'])
            
            print(f"Prediction result: {stress_level} ({stress_category}) with {confidence}% confidence")
        except Exception as e:
            flash(f'Error making predictions: {str(e)}', 'danger')
            return redirect(url_for('stress_analyzer'))

        health_factors = {
            'Sleep Duration': (user_data['sleep_duration'], 7, 9, 'hours'),
            'Physical Activity': (user_data['activity'], 30, 60, 'minutes'),
            'Heart Rate': (user_data['heart_rate'], 60, 100, 'bpm'),
            'Systolic BP': (systolic, 90, 120, 'mmHg'),
            'Diastolic BP': (diastolic, 60, 80, 'mmHg')
        }

        recommendations = []
        
        if stress_level > 7:
            recommendations.extend([
                "High Stress Alert",
                "- Practice deep breathing exercises (4-7-8 technique)",
                "- Consider talking to a mental health professional",
                "- Take short breaks throughout the day",
                "- Limit caffeine and screen time before bed"
            ])
        elif stress_level > 4:
            recommendations.extend([
                "Moderate Stress Level",
                "- Maintain a consistent sleep schedule",
                "- Try meditation or mindfulness apps",
                "- Regular exercise (30 mins daily)",
                "- Connect with friends/family for support"
            ])
        else:
            recommendations.extend([
                "Low Stress Level - Great job!",
                "- Keep up your healthy habits",
                "- Continue regular exercise",
                "- Maintain work-life balance",
                "- Practice preventive self-care"
            ])
        
        if user_data['sleep_duration'] < 7:
            recommendations.append("Sleep: Aim for 7-9 hours of sleep. Try a relaxing bedtime routine.")
        elif user_data['sleep_duration'] > 9:
            recommendations.append("Sleep: Consider if oversleeping affects your energy. Try consistent wake times.")
        
        if user_data['activity'] < 30:
            recommendations.append("Activity: Start with 15-20 min walks, gradually increase to 30 min daily.")
        
        if user_data['heart_rate'] > 100:
            recommendations.append("Heart Rate: Your resting heart rate is high. Consider relaxation techniques.")
        elif user_data['heart_rate'] < 60 and user_data['age'] < 60:
            recommendations.append("Heart Rate: Your heart rate is on the lower side - this can be normal for athletes.")
        
        if systolic > 130 or diastolic > 85:
            recommendations.append("Blood Pressure: Monitor your BP regularly. Reduce salt intake and stay hydrated.")

        return render_template('results.html', 
                            stress_level=round(stress_level, 1),
                            stress_category=stress_category,
                            confidence=confidence,
                            probabilities=probabilities,
                            health_factors=health_factors,
                            recommendations=recommendations,
                            user_data=user_data,
                            visualization=img_data)

    except Exception as e:
        print(f"Unexpected error in analysis_results: {str(e)}")
        flash('An unexpected error occurred during analysis', 'danger')
        return redirect(url_for('stress_analyzer'))

def fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

@app.route('/visualizations')
def show_visualizations():
    try:
        fig1 = plt.figure(figsize=(10, 6))
        chatbot.plot_stress_distribution(fig1)
        img1 = fig_to_base64(fig1)
        plt.close(fig1)
        

        fig2 = plt.figure(figsize=(10, 6))
        chatbot.plot_stress_vs_age(fig2)
        img2 = fig_to_base64(fig2)
        plt.close(fig2)
        
        return render_template('visualizations.html', stress_dist_img=img1, stress_age_img=img2)
    except Exception as e:
        flash(f'Error generating visualizations: {str(e)}', 'danger')
        return redirect(url_for('home'))

@app.route('/feedback', methods=['POST'])
def handle_feedback():
    try:
        data = request.get_json()
        query = data.get('query')
        answer = data.get('answer')
        rating = int(data.get('rating'))
        
        chatbot.user_feedback.append({
            'query': query,
            'answer': answer,
            'rating': rating,
            'timestamp': datetime.datetime.now()
        })
        
        if len(chatbot.user_feedback) % 5 == 0:
            pd.DataFrame(chatbot.user_feedback).to_csv('user_feedback.csv', index=False)
        
        return jsonify({'success': True, 'message': 'Thank you for your feedback!'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/chat/history')
def get_chat_history():
    return jsonify(chat_history[-20:])

@app.route('/api/model-info')
def get_model_info_api():
    try:
        return jsonify(chatbot.get_model_info())
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'rag_enabled': chatbot.use_generation,
        'stress_analyzer_ready': True,
        'timestamp': datetime.datetime.now().isoformat()
    })

@app.route('/toggle-rag', methods=['POST'])
def toggle_rag():
    try:
        data = request.get_json()
        enabled = data.get('enabled', True)
        chatbot.use_generation = enabled
        return jsonify({
            'success': True, 
            'rag_enabled': chatbot.use_generation,
            'message': f'RAG mode {"enabled" if enabled else "disabled"}'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)