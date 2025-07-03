import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score, classification_report
from sklearn.dummy import DummyRegressor, DummyClassifier
class StressAnalyzer:
    def __init__(self):
        # Load and preprocess data
        self.df = self._load_data()
        self._preprocess_data()
        self._train_models()
    
    def _load_data(self):
        """Load dataset with error handling"""
        try:
            df = pd.read_csv('Sleep_health_and_lifestyle_dataset.csv')
            required_columns = ['Gender', 'Age', 'Sleep Duration', 'Physical Activity Level',
                                'Blood Pressure', 'Heart Rate', 'Stress Level']
            if not all(col in df.columns for col in required_columns):
                raise ValueError("Missing required columns in dataset")
            return df[required_columns]
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            # Create empty dataframe with required columns if file not found
            return pd.DataFrame(columns=required_columns)
    
    def _preprocess_data(self):
        """Prepare the dataset for modeling"""
        if self.df.empty:
            print("Warning: Empty dataframe, using default values")
            return
            
        # Convert gender to numerical
        self.df['Gender'] = self.df['Gender'].map({'Male': 0, 'Female': 1})
        
        # Split blood pressure with error handling
        try:
            bp_split = self.df['Blood Pressure'].str.split('/', expand=True)
            self.df[['Systolic BP', 'Diastolic BP']] = bp_split.astype(int)
        except Exception as e:
            print(f"Error processing blood pressure: {str(e)}")
            # Set default values if BP parsing fails
            self.df[['Systolic BP', 'Diastolic BP']] = 120, 80
            
        self.df.drop(columns=['Blood Pressure'], inplace=True, errors='ignore')
        
        # Define stress categorization
        def categorize_stress(stress):
            if stress <= 3: return 'Low'
            elif stress <= 6: return 'Medium'
            else: return 'High'
        
        self.df['Stress Category'] = self.df['Stress Level'].apply(categorize_stress)
    
    def _train_models(self):
        """Train machine learning models with error handling"""
        if self.df.empty:
            print("Warning: No data available for training, using dummy models")
            self.level_model = DummyRegressor(strategy='mean')
            self.category_model = DummyClassifier(strategy='most_frequent')
            return
            
        # Prepare data
        X = self.df.drop(columns=['Stress Level', 'Stress Category'], errors='ignore')
        y_level = self.df['Stress Level']
        y_category = self.df['Stress Category']
        
        # Train-test split
        X_train, X_test, y_level_train, y_level_test, y_category_train, y_category_test = train_test_split(
            X, y_level, y_category, test_size=0.2, random_state=42
        )
        
        # Train regression model
        self.level_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.level_model.fit(X_train, y_level_train)
        
        # Train classification model
        self.category_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.category_model.fit(X_train, y_category_train)
        
        # Evaluate models
        print("Regression Model R^2 Score:", self.level_model.score(X_test, y_level_test))
        print("Classification Model Accuracy:", accuracy_score(
            y_category_test, self.category_model.predict(X_test))
        )
    
    def _generate_visualizations(self, age, sleep, activity, hr, systolic, diastolic, 
                            stress_level, stress_category):
        """
        Generate comprehensive visualizations for stress analysis results.
        
        Parameters:
            age (int): User's age
            sleep (float): Hours of sleep per night
            activity (int): Minutes of physical activity per day
            hr (int): Resting heart rate (bpm)
            systolic (int): Systolic blood pressure
            diastolic (int): Diastolic blood pressure
            stress_level (float): Predicted stress level (0-10)
            stress_category (str): Stress category ('Low', 'Medium', 'High')
        
        Returns:
            matplotlib.figure.Figure: Generated figure object
        """
        # Debug print input parameters
        print(f"\n[DEBUG] Generating visualizations with parameters:")
        print(f"Age: {age}, Sleep: {sleep}h, Activity: {activity}min")
        print(f"Heart Rate: {hr}bpm, BP: {systolic}/{diastolic}mmHg")
        print(f"Stress: {stress_level}/10 ({stress_category})")

        try:
            # Create figure with constrained layout
            fig = plt.figure(figsize=(18, 12), constrained_layout=True)
            fig.suptitle('Stress Analysis Dashboard', fontsize=16, y=1.02)
            
            # 1. Stress Level Indicator (Horizontal Bar)
            ax1 = fig.add_subplot(2, 2, 1)
            stress_color = 'red' if stress_level > 7 else 'orange' if stress_level > 4 else 'green'
            ax1.barh(['Your Stress'], [stress_level], color=stress_color, height=0.6)
            ax1.set_xlim(0, 10)
            
            # Add reference lines
            for x, color, label in [(3, 'green', 'Low'), 
                                (6, 'orange', 'Medium'), 
                                (7, 'red', 'High')]:
                ax1.axvline(x=x, color=color, linestyle='--', alpha=0.7, label=label)
                
            ax1.set_title('Stress Level Assessment', pad=20)
            ax1.set_xlabel('Stress Level (0-10 Scale)')
            ax1.legend(loc='lower right')
            ax1.grid(True, axis='x', alpha=0.3)

            # 2. Health Metrics Radar Chart
            ax2 = fig.add_subplot(2, 2, 2, polar=True)
            
            # Calculate composite heart health score (0-100 scale)
            hr_score = max(0, 100 - (abs(hr - 70) * 2))  # 70 bpm as ideal
            bp_score = max(0, 100 - (abs(systolic - 115) + abs(diastolic - 75)))
            heart_health = (hr_score + bp_score) / 2
            
            metrics = ['Sleep', 'Activity', 'Heart Health']
            values = [sleep, activity, heart_health]
            ranges = [(3, 12), (0, 180), (0, 100)]  # min/max for normalization
            
            # Normalize values to 0-1 scale for radar chart
            norm_values = [(v - min_v) / (max_v - min_v) 
                        for v, (min_v, max_v) in zip(values, ranges)]
            
            # Complete the loop for radar chart
            angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
            angles += angles[:1]
            norm_values += norm_values[:1]
            
            # Plot radar chart
            ax2.plot(angles, norm_values, 'o-', linewidth=2, color='royalblue')
            ax2.fill(angles, norm_values, alpha=0.25, color='royalblue')
            ax2.set_xticks(angles[:-1])
            ax2.set_xticklabels(metrics)
            ax2.set_yticks([0.25, 0.5, 0.75, 1.0])
            ax2.set_yticklabels(["Low", "Fair", "Good", "Excellent"])
            ax2.set_ylim(0, 1)
            ax2.set_title('Health Metrics Overview', pad=20)

            # 3. Health Indicators vs Recommended Ranges
            ax3 = fig.add_subplot(2, 2, 3)
            factors = {
                'Sleep (h)': (sleep, 7, 9),
                'Activity (min)': (activity, 30, 60),
                'Heart Rate': (hr, 60, 100),
                'Systolic BP': (systolic, 90, 120),
                'Diastolic BP': (diastolic, 60, 80)
            }
            
            x = np.arange(len(factors))
            width = 0.25
            
            # Plot healthy ranges
            ax3.bar(x - width, [f[1] for f in factors.values()], 
                    width, label='Healthy Min', color='lightgreen', edgecolor='darkgreen')
            ax3.bar(x + width, [f[2] for f in factors.values()], 
                    width, label='Healthy Max', color='lightgreen', edgecolor='darkgreen')
            
            # Plot user values with color coding
            colors = ['green' if low <= v <= high else 'red' 
                    for v, low, high in factors.values()]
            ax3.bar(x, [f[0] for f in factors.values()], 
                    width, label='Your Value', color=colors, edgecolor='black')
            
            ax3.set_title('Your Health Metrics vs Recommendations', pad=20)
            ax3.set_xticks(x)
            ax3.set_xticklabels(factors.keys(), rotation=45, ha='right')
            ax3.legend()
            ax3.grid(True, axis='y', alpha=0.3)

            # 4. Age vs Stress Comparison
            ax4 = fig.add_subplot(2, 2, 4)
            age_bins = [20, 30, 40, 50, 60, 100]
            age_labels = ['20-29', '30-39', '40-49', '50-59', '60+']
            
            try:
                if not self.df.empty:
                    self.df['Age Group'] = pd.cut(self.df['Age'], bins=age_bins, labels=age_labels)
                    avg_stress = self.df.groupby('Age Group')['Stress Level'].mean()
                    user_age_group = age_labels[np.digitize(age, age_bins) - 1]
                    group_avg = avg_stress.get(user_age_group, self.df['Stress Level'].mean())
                else:
                    group_avg = 5.0  # Fallback average
            except:
                group_avg = 5.0  # Fallback if any error occurs
                
            # Plot comparison
            bars = ax4.bar(['Your Age Group Avg', 'Your Stress'], 
                        [group_avg, stress_level], 
                        color=['lightblue', stress_color])
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}', ha='center', va='bottom')
            
            ax4.set_title('Stress Level Comparison', pad=20)
            ax4.set_ylabel('Stress Level (0-10)')
            ax4.set_ylim(0, 10)
            ax4.grid(True, axis='y', alpha=0.3)

            plt.tight_layout()
            print("[DEBUG] Visualizations generated successfully")
            return fig

        except Exception as e:
            print(f"[ERROR] Visualization generation failed: {str(e)}")
            fig = plt.figure(figsize=(12, 6))
            plt.text(0.5, 0.5, "Could not generate visualizations\nError: " + str(e),
                    ha='center', va='center', color='red')
            return fig

if __name__ == "__main__":
    analyzer = StressAnalyzer()
