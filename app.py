from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import pickle
import os
from werkzeug.utils import secure_filename
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables for models
clinical_model = None
scaler = None
label_encoder = None

def train_clinical_model():
    """Train the clinical data model"""
    global clinical_model, scaler, label_encoder
    
    # Create sample dataset (in real scenario, load from CSV)
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'Age': np.random.randint(30, 80, n_samples),
        'Gender': np.random.choice(['M', 'F'], n_samples),
        'Smoking': np.random.choice([0, 1, 2], n_samples),  # 0: Never, 1: Former, 2: Current
        'Yellow_Fingers': np.random.choice([0, 1], n_samples),
        'Anxiety': np.random.choice([0, 1], n_samples),
        'Peer_Pressure': np.random.choice([0, 1], n_samples),
        'Chronic_Disease': np.random.choice([0, 1], n_samples),
        'Fatigue': np.random.choice([0, 1], n_samples),
        'Allergy': np.random.choice([0, 1], n_samples),
        'Wheezing': np.random.choice([0, 1], n_samples),
        'Alcohol': np.random.choice([0, 1], n_samples),
        'Coughing': np.random.choice([0, 1], n_samples),
        'Shortness_of_Breath': np.random.choice([0, 1], n_samples),
        'Swallowing_Difficulty': np.random.choice([0, 1], n_samples),
        'Chest_Pain': np.random.choice([0, 1], n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Create target variable based on risk factors
    risk_score = (df['Age'] > 60).astype(int) * 2 + \
                 (df['Smoking'] == 2).astype(int) * 3 + \
                 df['Yellow_Fingers'] * 1 + \
                 df['Chronic_Disease'] * 2 + \
                 df['Coughing'] * 1 + \
                 df['Shortness_of_Breath'] * 2 + \
                 df['Chest_Pain'] * 2
    
    df['Lung_Cancer'] = (risk_score > 5).astype(int)
    
    # Encode categorical variables
    label_encoder = LabelEncoder()
    df['Gender'] = label_encoder.fit_transform(df['Gender'])
    
    # Prepare features and target
    X = df.drop('Lung_Cancer', axis=1)
    y = df['Lung_Cancer']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the model
    clinical_model = RandomForestClassifier(n_estimators=100, random_state=42)
    clinical_model.fit(X_train_scaled, y_train)
    
    # Save the model
    with open('models/clinical_model.pkl', 'wb') as f:
        pickle.dump(clinical_model, f)
    
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    with open('models/label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    
    # Evaluate model
    y_pred = clinical_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Clinical Model Accuracy: {accuracy:.4f}")

def load_models():
    """Load pre-trained models"""
    global clinical_model, scaler, label_encoder
    
    os.makedirs('models', exist_ok=True)
    
    try:
        with open('models/clinical_model.pkl', 'rb') as f:
            clinical_model = pickle.load(f)
        with open('models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('models/label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
    except FileNotFoundError:
        print("Models not found. Training new models...")
        train_clinical_model()

def predict_from_ct_scan(image_path):
    """Simple CT scan analysis (placeholder for actual deep learning model)"""
    try:
        # Load and preprocess image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img_resized = cv2.resize(img, (224, 224))
        
        # Simple feature extraction (in real scenario, use CNN)
        mean_intensity = np.mean(img_resized)
        std_intensity = np.std(img_resized)
        
        # Simple rule-based prediction (replace with actual CNN model)
        if mean_intensity < 50 and std_intensity > 30:
            probability = 0.75
        elif mean_intensity < 80 and std_intensity > 25:
            probability = 0.45
        else:
            probability = 0.25
        
        prediction = 1 if probability > 0.5 else 0
        
        return {
            'prediction': prediction,
            'probability': probability,
            'confidence': abs(probability - 0.5) * 2
        }
    except Exception as e:
        return {'error': str(e)}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/clinical')
def clinical():
    return render_template('clinical.html')

@app.route('/ct-scan')
def ct_scan():
    return render_template('ct_scan.html')

@app.route('/dataset')
def dataset():
    return render_template('dataset.html')

@app.route('/predict_clinical', methods=['POST'])
def predict_clinical():
    try:
        data = request.json
        
        # Prepare input data
        input_data = np.array([[
            data['age'],
            1 if data['gender'] == 'M' else 0,  # Assuming M=1, F=0 from training
            data['smoking'],
            data['yellow_fingers'],
            data['anxiety'],
            data['peer_pressure'],
            data['chronic_disease'],
            data['fatigue'],
            data['allergy'],
            data['wheezing'],
            data['alcohol'],
            data['coughing'],
            data['shortness_of_breath'],
            data['swallowing_difficulty'],
            data['chest_pain']
        ]])
        
        # Scale the input
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = clinical_model.predict(input_scaled)[0]
        probability = clinical_model.predict_proba(input_scaled)[0]
        
        risk_level = "High" if prediction == 1 else "Low"
        confidence = max(probability) * 100
        
        return jsonify({
            'prediction': int(prediction),
            'risk_level': risk_level,
            'probability': float(max(probability)),
            'confidence': f"{confidence:.1f}%"
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict_ct_scan', methods=['POST'])
def predict_ct_scan():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Predict from CT scan
            result = predict_from_ct_scan(filepath)
            
            if 'error' in result:
                return jsonify(result), 400
            
            risk_level = "High" if result['prediction'] == 1 else "Low"
            
            return jsonify({
                'prediction': result['prediction'],
                'risk_level': risk_level,
                'probability': result['probability'],
                'confidence': f"{result['confidence']*100:.1f}%"
            })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/analyze_dataset', methods=['POST'])
def analyze_dataset():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read CSV file
        df = pd.read_csv(file)
        
        # Basic analysis
        analysis = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'missing_values': df.isnull().sum().to_dict(),
            'basic_stats': df.describe().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else {}
        }
        
        # If there's a target column, perform prediction
        if 'Lung_Cancer' in df.columns or 'LUNG_CANCER' in df.columns:
            target_col = 'Lung_Cancer' if 'Lung_Cancer' in df.columns else 'LUNG_CANCER'
            predictions = []
            
            for _, row in df.iterrows():
                # Simple prediction based on available data
                risk_score = 0
                if 'Age' in row and row['Age'] > 60:
                    risk_score += 2
                if 'Smoking' in row and row['Smoking'] == 2:
                    risk_score += 3
                # Add more conditions based on available columns
                
                prediction = 1 if risk_score > 3 else 0
                predictions.append(prediction)
            
            analysis['predictions'] = predictions[:10]  # First 10 predictions
            analysis['prediction_summary'] = {
                'high_risk': sum(predictions),
                'low_risk': len(predictions) - sum(predictions),
                'total': len(predictions)
            }
        
        return jsonify(analysis)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    load_models()
    app.run(debug=True, port=5000)
