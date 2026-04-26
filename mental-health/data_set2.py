import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import keras
from keras import layers, models
from keras.models import load_model
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, accuracy_score, classification_report
from sklearn.dummy import DummyRegressor, DummyClassifier
import warnings
warnings.filterwarnings('ignore')

class StressAnalyzer:
    def __init__(self):
        self.base_dir = Path(__file__).resolve().parent
        self.project_dir = self.base_dir.parent
        self.saved_model = None
        self.saved_model_path = None
        self.saved_feature_columns = []
        self.saved_feature_defaults = None
        self.raw_feature_ranges = {}

        self.df = self._load_data()
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self._preprocess_data()
        if not self._load_saved_model():
            self._build_and_train_nn()

    def _load_saved_model(self):
        preprocessed_path = self.project_dir / 'stressdata_preprocessed.csv'
        if not preprocessed_path.exists():
            print(f"Saved-model mode skipped: {preprocessed_path} not found")
            return False

        try:
            model_df = pd.read_csv(preprocessed_path)
            self.saved_feature_columns = [col for col in model_df.columns if col != 'Stress Level']
            self.saved_feature_defaults = model_df[self.saved_feature_columns].median(numeric_only=True)
            self._load_raw_feature_ranges()

            for model_path in (self.base_dir / 'stress_model.keras', self.base_dir / 'stress_model.h5'):
                if not model_path.exists():
                    continue

                model = load_model(model_path, compile=False)
                expected_inputs = model.input_shape[-1]
                if expected_inputs != len(self.saved_feature_columns):
                    print(
                        f"Skipping {model_path.name}: expects {expected_inputs} inputs, "
                        f"but {len(self.saved_feature_columns)} features are available"
                    )
                    continue

                self.saved_model = model
                self.saved_model_path = model_path
                print(
                    f"Loaded saved stress model: {model_path.name} "
                    f"({expected_inputs} inputs -> {model.output_shape[-1]} outputs)"
                )
                return True
        except Exception as e:
            print(f"Saved-model load failed: {type(e).__name__}: {str(e)}")

        return False

    def _load_raw_feature_ranges(self):
        raw_path = self.project_dir / 'stressdata.csv'
        if not raw_path.exists():
            return

        raw_df = pd.read_csv(raw_path)
        if 'Blood Pressure' in raw_df.columns:
            bp_split = raw_df['Blood Pressure'].astype(str).str.split('/', expand=True)
            raw_df['Systolic_BP'] = pd.to_numeric(bp_split[0], errors='coerce')
            raw_df['Diastolic_BP'] = pd.to_numeric(bp_split[1], errors='coerce')

        for col in ['Age', 'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level',
                    'Heart Rate', 'Daily Steps', 'Systolic_BP', 'Diastolic_BP']:
            if col in raw_df.columns:
                values = pd.to_numeric(raw_df[col], errors='coerce').dropna()
                if not values.empty:
                    self.raw_feature_ranges[col] = (float(values.min()), float(values.max()))

    def _scale_like_training_data(self, column, value):
        min_value, max_value = self.raw_feature_ranges.get(column, (None, None))
        if min_value is None or max_value is None or max_value == min_value:
            return value
        return (float(value) - min_value) / (max_value - min_value)

    def _saved_model_input(
        self, age, sleep_duration, activity_level, heart_rate, blood_pressure, gender,
        occupation='Accountant', quality_of_sleep=5, bmi_category='Normal',
        daily_steps=5000, sleep_disorder='None'
    ):
        try:
            systolic, diastolic = map(float, blood_pressure.split('/'))
        except Exception:
            systolic, diastolic = 120.0, 80.0

        row = self.saved_feature_defaults.copy()
        bmi_mapping = {
            'underweight': 0.0,
            'normal': 1.0 / 3.0,
            'overweight': 2.0 / 3.0,
            'obese': 1.0,
        }
        values = {
            'Gender': 1.0 if str(gender).lower() == 'male' else 0.0,
            'Age': self._scale_like_training_data('Age', age),
            'Sleep Duration': self._scale_like_training_data('Sleep Duration', sleep_duration),
            'Quality of Sleep': self._scale_like_training_data('Quality of Sleep', quality_of_sleep),
            'Physical Activity Level': self._scale_like_training_data('Physical Activity Level', activity_level),
            'BMI Category': bmi_mapping.get(str(bmi_category).lower(), 1.0 / 3.0),
            'Heart Rate': self._scale_like_training_data('Heart Rate', heart_rate),
            'Daily Steps': self._scale_like_training_data('Daily Steps', daily_steps),
            'Sleep Disorder': 0.0 if str(sleep_disorder).lower() in ('none', 'no', 'nan', '') else 1.0,
            'Systolic_BP': self._scale_like_training_data('Systolic_BP', systolic),
            'Diastolic_BP': self._scale_like_training_data('Diastolic_BP', diastolic),
        }

        for column, value in values.items():
            if column in row.index:
                row[column] = value

        occupation_column = f"Occupation_{occupation}"
        occupation_columns = [col for col in row.index if col.startswith('Occupation_')]
        for column in occupation_columns:
            row[column] = 1.0 if column == occupation_column else 0.0

        return row[self.saved_feature_columns].to_numpy(dtype=np.float32).reshape(1, -1), systolic, diastolic

    def _predict_with_saved_model(
        self, age, sleep_duration, activity_level, heart_rate, blood_pressure, gender,
        occupation='Accountant', quality_of_sleep=5, bmi_category='Normal',
        daily_steps=5000, sleep_disorder='None'
    ):
        input_features, systolic, diastolic = self._saved_model_input(
            age, sleep_duration, activity_level, heart_rate, blood_pressure, gender,
            occupation, quality_of_sleep, bmi_category, daily_steps, sleep_disorder
        )

        raw_output = np.asarray(self.saved_model.predict(input_features, verbose=0))[0].astype(float)
        if np.any(raw_output < 0) or not np.isclose(raw_output.sum(), 1.0, atol=1e-3):
            shifted = raw_output - np.max(raw_output)
            probabilities = np.exp(shifted) / np.sum(np.exp(shifted))
        else:
            probabilities = raw_output / raw_output.sum()

        stress_idx = int(np.argmax(probabilities))
        stress_level = float(stress_idx + 1)
        stress_category = self._categorize_stress_level(stress_level)
        confidence = float(probabilities[stress_idx])

        fig = self._generate_visualizations(
            age, sleep_duration, activity_level, heart_rate,
            systolic, diastolic, stress_level, stress_category
        )

        return {
            'stress_level': round(stress_level, 1),
            'stress_category': stress_category,
            'confidence': round(confidence * 100, 1),
            'probabilities': {
                f'Level {idx + 1}': round(float(prob) * 100, 1)
                for idx, prob in enumerate(probabilities)
            },
            'visualization': fig
        }

    def _categorize_stress_level(self, stress):
        if stress <= 3.5:
            return 'Low'
        if stress <= 6.5:
            return 'Medium'
        return 'High'
    
    def _load_data(self):
        try:
            df = pd.read_csv('Sleep_health_and_lifestyle_dataset.csv')
            required_columns = ['Gender', 'Age', 'Sleep Duration', 'Physical Activity Level',
                                'Blood Pressure', 'Heart Rate', 'Stress Level']
            if not all(col in df.columns for col in required_columns):
                raise ValueError("Missing required columns in dataset")
            return df[required_columns]
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return self._create_synthetic_data()
    
    def _create_synthetic_data(self, n_samples=1000):
        np.random.seed(42)
        data = {
            'Gender': np.random.choice(['Male', 'Female'], n_samples),
            'Age': np.random.randint(18, 70, n_samples),
            'Sleep Duration': np.random.uniform(4, 10, n_samples).round(1),
            'Physical Activity Level': np.random.randint(0, 180, n_samples),
            'Blood Pressure': [f"{np.random.randint(100, 140)}/{np.random.randint(60, 90)}" 
                             for _ in range(n_samples)],
            'Heart Rate': np.random.randint(55, 100, n_samples),
            'Stress Level': np.random.uniform(0, 10, n_samples).round(1)
        }
        return pd.DataFrame(data)
    
    def _preprocess_data(self):
        if self.df.empty:
            print("Warning: Empty dataframe, using default values")
            return
            
        self.df['Gender'] = self.df['Gender'].map({'Male': 0, 'Female': 1})
        
        try:
            bp_split = self.df['Blood Pressure'].str.split('/', expand=True)
            self.df[['Systolic BP', 'Diastolic BP']] = bp_split.astype(float)
        except Exception as e:
            print(f"Error processing blood pressure: {str(e)}")
            self.df['Systolic BP'] = np.random.randint(110, 130, len(self.df))
            self.df['Diastolic BP'] = np.random.randint(70, 85, len(self.df))
        
        self.df.drop(columns=['Blood Pressure'], inplace=True, errors='ignore')
        
        self.df['BP_Ratio'] = self.df['Systolic BP'] / self.df['Diastolic BP']
        self.df['Sleep_Quality_Index'] = self.df['Sleep Duration'] * 10  
        
        def categorize_stress(stress):
            if stress <= 3.5: return 'Low'
            elif stress <= 6.5: return 'Medium'
            else: return 'High'
        
        self.df['Stress Category'] = self.df['Stress Level'].apply(categorize_stress)
        
        self.df['Stress Category Encoded'] = self.label_encoder.fit_transform(self.df['Stress Category'])
        
        print(f"Data preprocessed: {len(self.df)} samples")
        print(f"   Features: {list(self.df.columns)}")
    
    def _build_and_train_nn(self):
        if self.df.empty:
            print("Warning: No data available for training, using dummy models")
            self.level_model = DummyRegressor(strategy='mean')
            self.category_model = DummyClassifier(strategy='most_frequent')
            return
        
        feature_columns = ['Gender', 'Age', 'Sleep Duration', 'Physical Activity Level',
                          'Heart Rate', 'Systolic BP', 'Diastolic BP', 'BP_Ratio', 'Sleep_Quality_Index']
        
        X = self.df[feature_columns].values
        y_level = self.df['Stress Level'].values
        y_category = self.df['Stress Category Encoded'].values
        
        X_train, X_test, y_level_train, y_level_test, y_category_train, y_category_test = train_test_split(
            X, y_level, y_category, test_size=0.2, random_state=42
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.level_model = self._build_regression_nn(X_train_scaled.shape[1])
        
        self.category_model = self._build_classification_nn(
            X_train_scaled.shape[1], 
            len(self.label_encoder.classes_)
        )
        
        print("\nTraining Stress Level Regression Network...")
        history_reg = self.level_model.fit(
            X_train_scaled, y_level_train,
            validation_data=(X_test_scaled, y_level_test),
            epochs=50, batch_size=32, verbose=0
        )
        
        print("Training Stress Category Classification Network...")
        history_clf = self.category_model.fit(
            X_train_scaled, y_category_train,
            validation_data=(X_test_scaled, y_category_test),
            epochs=50, batch_size=32, verbose=0
        )
        
        y_level_pred = self.level_model.predict(X_test_scaled).flatten()
        y_category_pred_proba = self.category_model.predict(X_test_scaled)
        y_category_pred = np.argmax(y_category_pred_proba, axis=1)
        
        self.regression_metrics = {
            'mae': mean_absolute_error(y_level_test, y_level_pred),
            'r2': self._calculate_r2(y_level_test, y_level_pred)
        }
        
        self.classification_metrics = {
            'accuracy': accuracy_score(y_category_test, y_category_pred),
            'report': classification_report(
                y_category_test, 
                y_category_pred, 
                target_names=self.label_encoder.classes_,
                output_dict=True
            )
        }
        
        self.feature_importance = self._approximate_feature_importance(feature_columns)
        
        print(f"\nNeural Networks trained successfully!")
        print(f"   Regression MAE: {self.regression_metrics['mae']:.3f}")
        print(f"   Regression R²: {self.regression_metrics['r2']:.3f}")
        print(f"   Classification Accuracy: {self.classification_metrics['accuracy']:.3f}")
    
    def _build_regression_nn(self, input_dim):
        model = models.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(32, activation='relu'),
            layers.Dense(16, activation='relu'),
            
            layers.Dense(1, activation='linear')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def _build_classification_nn(self, input_dim, num_classes):
        model = models.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(32, activation='relu'),
            layers.Dense(16, activation='relu'),
            
            layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _calculate_r2(self, y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / (ss_tot + 1e-8))

    def _approximate_feature_importance(self, feature_names):
        weights = self.level_model.layers[0].get_weights()[0]
        importance = np.mean(np.abs(weights), axis=1)
        importance = importance / np.sum(importance)
        return dict(zip(feature_names, importance))
    
    def predict_stress(
        self, age, sleep_duration, activity_level, heart_rate, blood_pressure, gender,
        occupation='Accountant', quality_of_sleep=5, bmi_category='Normal',
        daily_steps=5000, sleep_disorder='None'
    ):
        if self.saved_model is not None:
            return self._predict_with_saved_model(
                age, sleep_duration, activity_level, heart_rate, blood_pressure, gender,
                occupation, quality_of_sleep, bmi_category, daily_steps, sleep_disorder
            )

        try:
            systolic, diastolic = map(float, blood_pressure.split('/'))
        except:
            systolic, diastolic = 120, 80
        
        gender_encoded = 0 if gender.lower() == 'male' else 1
        bp_ratio = systolic / diastolic
        sleep_quality = sleep_duration * 10
        
        input_features = np.array([[
            gender_encoded, age, sleep_duration, activity_level,
            heart_rate, systolic, diastolic, bp_ratio, sleep_quality
        ]])
        
        input_scaled = self.scaler.transform(input_features)
        
        stress_level = self.level_model.predict(input_scaled, verbose=0)[0][0]
        stress_level = np.clip(stress_level, 0, 10)
        
        category_probs = self.category_model.predict(input_scaled, verbose=0)[0]
        category_idx = np.argmax(category_probs)
        stress_category = self.label_encoder.inverse_transform([category_idx])[0]
        
        confidence = float(category_probs[category_idx])
        
        fig = self._generate_visualizations(
            age, sleep_duration, activity_level, heart_rate,
            systolic, diastolic, stress_level, stress_category
        )
        
        return {
            'stress_level': round(stress_level, 1),
            'stress_category': stress_category,
            'confidence': round(confidence * 100, 1),
            'probabilities': {
                category: round(prob * 100, 1)
                for category, prob in zip(self.label_encoder.classes_, category_probs)
            },
            'visualization': fig
        }

    def _generate_visualizations(self, age, sleep, activity, hr, systolic, diastolic, 
                               stress_level, stress_category):
        print(f"\n[DEBUG] Generating visualizations with parameters:")
        print(f"Age: {age}, Sleep: {sleep}h, Activity: {activity}min")
        print(f"Heart Rate: {hr}bpm, BP: {systolic}/{diastolic}mmHg")
        print(f"Stress: {stress_level}/10 ({stress_category})")

        try:
            fig = plt.figure(figsize=(18, 12), constrained_layout=True)
            fig.suptitle('Stress Analysis Dashboard (Neural Network)', fontsize=16, y=1.02)
            
            ax1 = fig.add_subplot(2, 2, 1)
            stress_color = 'red' if stress_level > 7 else 'orange' if stress_level > 4 else 'green'
            ax1.barh(['Your Stress'], [stress_level], color=stress_color, height=0.6)
            ax1.set_xlim(0, 10)
            
            for x, color, label in [(3, 'green', 'Low'), 
                                (6, 'orange', 'Medium'), 
                                (7, 'red', 'High')]:
                ax1.axvline(x=x, color=color, linestyle='--', alpha=0.7, label=label)
                
            ax1.set_title('Stress Level Assessment', pad=20)
            ax1.set_xlabel('Stress Level (0-10 Scale)')
            ax1.legend(loc='lower right')
            ax1.grid(True, axis='x', alpha=0.3)

            ax2 = fig.add_subplot(2, 2, 2, polar=True)
            
            hr_score = max(0, 100 - (abs(hr - 70) * 2))
            bp_score = max(0, 100 - (abs(systolic - 115) + abs(diastolic - 75)))
            heart_health = (hr_score + bp_score) / 2
            
            metrics = ['Sleep', 'Activity', 'Heart Health']
            values = [sleep, activity, heart_health]
            ranges = [(3, 12), (0, 180), (0, 100)]
            
            norm_values = [(v - min_v) / (max_v - min_v) 
                        for v, (min_v, max_v) in zip(values, ranges)]
            
            angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
            angles += angles[:1]
            norm_values += norm_values[:1]
            
            ax2.plot(angles, norm_values, 'o-', linewidth=2, color='royalblue')
            ax2.fill(angles, norm_values, alpha=0.25, color='royalblue')
            ax2.set_xticks(angles[:-1])
            ax2.set_xticklabels(metrics)
            ax2.set_yticks([0.25, 0.5, 0.75, 1.0])
            ax2.set_yticklabels(["Low", "Fair", "Good", "Excellent"])
            ax2.set_ylim(0, 1)
            ax2.set_title('Health Metrics Overview', pad=20)

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
            
            ax3.bar(x - width, [f[1] for f in factors.values()], 
                    width, label='Healthy Min', color='lightgreen', edgecolor='darkgreen')
            ax3.bar(x + width, [f[2] for f in factors.values()], 
                    width, label='Healthy Max', color='lightgreen', edgecolor='darkgreen')
            
            colors = ['green' if low <= v <= high else 'red' 
                    for v, low, high in factors.values()]
            ax3.bar(x, [f[0] for f in factors.values()], 
                    width, label='Your Value', color=colors, edgecolor='black')
            
            ax3.set_title('Your Health Metrics vs Recommendations', pad=20)
            ax3.set_xticks(x)
            ax3.set_xticklabels(factors.keys(), rotation=45, ha='right')
            ax3.legend()
            ax3.grid(True, axis='y', alpha=0.3)

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
                    group_avg = 5.0
            except:
                group_avg = 5.0
                
            bars = ax4.bar(['Your Age Group Avg', 'Your Stress'], 
                        [group_avg, stress_level], 
                        color=['lightblue', stress_color])
            
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
    
    result = analyzer.predict_stress(
        age=25,
        sleep_duration=7.5,
        activity_level=45,
        heart_rate=72,
        blood_pressure="118/76",
        gender="Male"
    )
    
    print("\nPrediction Results:")
    print(f"Stress Level: {result['stress_level']}/10")
    print(f"Category: {result['stress_category']}")
    print(f"Confidence: {result['confidence']}%")
    print("Probabilities:", result['probabilities'])
    
    plt.show()
