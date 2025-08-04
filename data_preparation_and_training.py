#!/usr/bin/env python3
"""
Lung Cancer Prediction - Data Preparation and Training
This script downloads Kaggle datasets, preprocesses them, and trains ML models.
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle
import warnings
from pathlib import Path
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.impute import SimpleImputer

# Deep Learning imports
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Suppress warnings
warnings.filterwarnings('ignore')

try:
    import kagglehub
except ImportError:
    print("Installing kagglehub...")
    os.system("pip install kagglehub")
    import kagglehub

class LungCancerDataProcessor:
    def __init__(self):
        self.data_dir = Path('./data')
        self.models_dir = Path('./models')
        self.data_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        
        self.clinical_model = None
        self.ct_scan_model = None
        self.scaler = None
        self.label_encoders = {}
        
    def download_datasets(self):
        """Download datasets from Kaggle using kagglehub"""
        print("Downloading datasets from Kaggle...")
        
        try:
            # 1. Cancer patient dataset
            print("Downloading cancer patient dataset...")
            path1 = kagglehub.dataset_download("thedevastator/cancer-patients-and-air-pollution-a-new-link")
            print(f"Cancer patient dataset downloaded to: {path1}")
            
            # 2. Chest CT-Scan images
            print("Downloading CT scan images...")
            path2 = kagglehub.dataset_download("mohamedhanyyy/chest-ctscan-images")
            print(f"CT scan images downloaded to: {path2}")
            
            # 3. Medical text dataset
            print("Downloading medical text dataset...")
            path3 = kagglehub.dataset_download("chaitanyakck/medical-text")
            print(f"Medical text dataset downloaded to: {path3}")
            
            return path1, path2, path3
            
        except Exception as e:
            print(f"Error downloading datasets: {e}")
            return None, None, None
    
    def preprocess_clinical_data(self, data_path):
        """Preprocess the clinical cancer patient dataset"""
        print("Preprocessing clinical data...")
        
        # Find CSV files in the downloaded path
        csv_files = list(Path(data_path).glob("*.csv"))
        if not csv_files:
            print("No CSV files found in the dataset")
            return None
            
        df = pd.read_csv(csv_files[0])
        print(f"Original dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Handle missing values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        categorical_columns = df.select_dtypes(include=['object']).columns
        
        # Fill missing values
        if len(numeric_columns) > 0:
            imputer_num = SimpleImputer(strategy='median')
            df[numeric_columns] = imputer_num.fit_transform(df[numeric_columns])
            
        if len(categorical_columns) > 0:
            imputer_cat = SimpleImputer(strategy='most_frequent')
            df[categorical_columns] = imputer_cat.fit_transform(df[categorical_columns])
        
        # Encode categorical variables
        for col in categorical_columns:
            if col.lower() not in ['patient_id', 'id']:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
        
        # Create target variable if not exists
        if 'Lung_Cancer' not in df.columns and 'LUNG_CANCER' not in df.columns:
            # Create synthetic target based on risk factors
            target_col = 'Lung_Cancer'
            df[target_col] = 0  # Default to no cancer
            
            # Simple heuristic for demonstration
            if 'Age' in df.columns:
                df.loc[df['Age'] > 60, target_col] = 1
        
        print(f"Processed dataset shape: {df.shape}")
        return df
    
    def train_clinical_model(self, df):
        """Train machine learning model on clinical data"""
        print("Training clinical prediction model...")
        
        # Identify target column
        target_cols = ['Lung_Cancer', 'LUNG_CANCER', 'lung_cancer']
        target_col = None
        for col in target_cols:
            if col in df.columns:
                target_col = col
                break
                
        if target_col is None:
            print("No target column found. Creating synthetic target.")
            target_col = 'Lung_Cancer'
            # Create synthetic target based on available features
            df[target_col] = np.random.randint(0, 2, size=len(df))
        
        # Prepare features and target
        X = df.drop([target_col], axis=1)
        y = df[target_col]
        
        # Remove non-numeric columns that couldn't be encoded
        X = X.select_dtypes(include=[np.number])
        
        print(f"Features shape: {X.shape}")
        print(f"Target distribution: {y.value_counts()}")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale the features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest model
        self.clinical_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        
        self.clinical_model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = self.clinical_model.predict(X_test_scaled)
        y_pred_proba = self.clinical_model.predict_proba(X_test_scaled)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        print(f"Clinical Model Performance:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"AUC Score: {auc_score:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Save model
        with open(self.models_dir / 'clinical_model.pkl', 'wb') as f:
            pickle.dump(self.clinical_model, f)
        with open(self.models_dir / 'scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        with open(self.models_dir / 'label_encoders.pkl', 'wb') as f:
            pickle.dump(self.label_encoders, f)
            
        return accuracy, auc_score
    
    def preprocess_ct_images(self, images_path):
        """Preprocess CT scan images"""
        print("Preprocessing CT scan images...")
        
        images_dir = Path(images_path)
        
        # Look for image directories
        image_dirs = [d for d in images_dir.iterdir() if d.is_dir()]
        
        if not image_dirs:
            print("No image directories found")
            return None, None
        
        print(f"Found image directories: {[d.name for d in image_dirs]}")
        
        # Create data generators
        datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            validation_split=0.2
        )
        
        try:
            train_generator = datagen.flow_from_directory(
                images_path,
                target_size=(224, 224),
                batch_size=32,
                class_mode='binary',
                subset='training'
            )
            
            validation_generator = datagen.flow_from_directory(
                images_path,
                target_size=(224, 224),
                batch_size=32,
                class_mode='binary',
                subset='validation'
            )
            
            return train_generator, validation_generator
            
        except Exception as e:
            print(f"Error creating data generators: {e}")
            return None, None
    
    def train_ct_model(self, train_gen, val_gen):
        """Train CNN model for CT scan analysis"""
        print("Training CT scan model...")
        
        # Build CNN model
        self.ct_scan_model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            
            Conv2D(64, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            
            Conv2D(128, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            
            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(256, activation='relu'),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])
        
        # Compile model
        self.ct_scan_model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        print(self.ct_scan_model.summary())
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        model_checkpoint = ModelCheckpoint(
            self.models_dir / 'ct_scan_model.h5',
            monitor='val_accuracy',
            save_best_only=True
        )
        
        # Train model
        history = self.ct_scan_model.fit(
            train_gen,
            epochs=20,
            validation_data=val_gen,
            callbacks=[early_stopping, model_checkpoint],
            verbose=1
        )
        
        # Plot training history
        self.plot_training_history(history)
        
        return history
    
    def plot_training_history(self, history):
        """Plot training history"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.show()
    
    def generate_analysis_report(self, clinical_df):
        """Generate comprehensive analysis report"""
        print("Generating analysis report...")
        
        report = {
            'dataset_info': {
                'shape': clinical_df.shape,
                'columns': list(clinical_df.columns),
                'missing_values': clinical_df.isnull().sum().to_dict(),
                'data_types': clinical_df.dtypes.to_dict()
            },
            'statistics': clinical_df.describe().to_dict(),
            'correlation_matrix': clinical_df.corr().to_dict() if len(clinical_df.select_dtypes(include=[np.number]).columns) > 1 else {}
        }
        
        # Save report
        with open('analysis_report.pkl', 'wb') as f:
            pickle.dump(report, f)
            
        return report
    
    def main(self):
        """Main execution function"""
        print("Starting Lung Cancer Prediction Data Processing and Training...")
        
        # Download datasets
        clinical_path, ct_path, medical_text_path = self.download_datasets()
        
        if clinical_path:
            # Process clinical data
            clinical_df = self.preprocess_clinical_data(clinical_path)
            
            if clinical_df is not None:
                # Train clinical model
                accuracy, auc = self.train_clinical_model(clinical_df)
                
                # Generate analysis report
                report = self.generate_analysis_report(clinical_df)
                
                print(f"\nClinical model trained successfully!")
                print(f"Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")
        
        if ct_path:
            # Process CT scan images
            train_gen, val_gen = self.preprocess_ct_images(ct_path)
            
            if train_gen is not None and val_gen is not None:
                # Train CT scan model
                history = self.train_ct_model(train_gen, val_gen)
                print("\nCT scan model trained successfully!")
        
        print("\nTraining completed! Models saved in ./models directory")
        print("Analysis report saved as analysis_report.pkl")

if __name__ == '__main__':
    processor = LungCancerDataProcessor()
    processor.main()

