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

chatbot = MentalHealthChatbot()
analyzer = StressAnalyzer()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chatbot', methods=['GET', 'POST'])
def chatbot_interface():
    if request.method == 'POST':
        user_input = request.form.get('user_input')
        response = chatbot.get_answer(user_input)
        return jsonify({'response': response})
    return render_template('chatbot.html')

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
    """Route to display stress analysis results with visualizations"""
    try:
        # Check if user data exists in session
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
            input_df = pd.DataFrame([input_data], columns=[
                'Gender', 'Age', 'Sleep Duration', 'Physical Activity Level',
                'Heart Rate', 'Systolic BP', 'Diastolic BP'
            ])
            
            stress_level = analyzer.level_model.predict(input_df)[0]
            stress_category = analyzer.category_model.predict(input_df)[0]
            print("Prediction results - Level:", stress_level, "Category:", stress_category)  
        except Exception as e:
            flash(f'Error making predictions: {str(e)}', 'danger')
            return redirect(url_for('stress_analyzer'))

        try:
            fig = analyzer._generate_visualizations(
            age=user_data['age'],
            sleep=user_data['sleep_duration'],  
            activity=user_data['activity'], 
            hr=user_data['heart_rate'],
            systolic=systolic,
            diastolic=diastolic,
            stress_level=stress_level,
            stress_category=stress_category
        )
            
            img_data = fig_to_base64(fig)
            plt.close(fig)
            
            if not img_data:
                raise ValueError("Empty image data generated")
            print("Visualization generated successfully")  
        except Exception as e:
            print(f"Error generating visualization: {str(e)}")  
            img_data = None
            flash('Could not generate visualization', 'warning')

        health_factors = {
            'Sleep Duration': (user_data['sleep_duration'], 7, 9, 'hours'),
            'Physical Activity': (user_data['activity'], 30, 60, 'minutes'),
            'Heart Rate': (user_data['heart_rate'], 60, 100, 'bpm'),
            'Systolic BP': (systolic, 90, 120, 'mmHg'),
            'Diastolic BP': (diastolic, 60, 80, 'mmHg')
        }

        recommendations = []
        if stress_level > 7:
            recommendations = [
                "Your stress level is high. Consider:",
                "- Relaxation techniques (deep breathing, meditation)",
                "- Regular physical activity",
                "- Consulting a healthcare professional"
            ]
        elif stress_level > 4:
            recommendations = [
                "Your stress level is moderate. Consider:",
                "- Maintaining good sleep habits",
                "- Regular exercise",
                "- Time management strategies"
            ]
        else:
            recommendations = [
                "Your stress level is well-managed.",
                "Continue your healthy habits!"
            ]
        
        if user_data['sleep_duration'] < 7:
            recommendations.append("Sleep Note: You're getting less than recommended sleep (7-9 hours).")
        elif user_data['sleep_duration'] > 9:
            recommendations.append("Sleep Note: You're sleeping more than recommended (7-9 hours).")
        
        if user_data['activity'] < 30:
            recommendations.append("Activity Note: Your physical activity is below recommended levels.")

        return render_template('results.html', 
                            stress_level=round(stress_level, 1),
                            stress_category=stress_category,
                            health_factors=health_factors,
                            recommendations=recommendations,
                            user_data=user_data,
                            visualization=img_data)

    except Exception as e:
        print(f"Unexpected error in analysis_results: {str(e)}")  
        flash('An unexpected error occurred during analysis', 'danger')
        return redirect(url_for('stress_analyzer'))
def fig_to_base64(fig):
    """Convert matplotlib figure to base64 encoded image"""
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

@app.route('/visualizations')
def show_visualizations():
    
    fig1 = plt.figure(figsize=(10, 6))
    chatbot.plot_stress_distribution(fig1)
    img1 = fig_to_base64(fig1)
    plt.close(fig1)
    

    fig2 = plt.figure(figsize=(10, 6))
    chatbot.plot_stress_vs_age(fig2)
    img2 = fig_to_base64(fig2)
    plt.close(fig2)
    
    return render_template('visualizations.html', stress_dist_img=img1, stress_age_img=img2)

@app.route('/feedback', methods=['POST'])
def handle_feedback():
    try:
        data = request.get_json()
        query = data.get('query')
        answer = data.get('answer')
        rating = int(data.get('rating'))
        
        # Log the feedback
        chatbot.user_feedback.append({
            'query': query,
            'answer': answer,
            'rating': rating,
            'timestamp': datetime.datetime.now()
        })
        
        # Save to CSV periodically
        if len(chatbot.user_feedback) % 5 == 0:  
            pd.DataFrame(chatbot.user_feedback).to_csv('user_feedback.csv', index=False)
        
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
