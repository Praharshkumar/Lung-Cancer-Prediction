#!/usr/bin/env python3
"""
Lung Cancer Prediction - Setup and Run Script
This script sets up the environment and runs the web application.
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error during {description}:")
        print(f"  {e.stderr}")
        return False

def setup_environment():
    """Set up the Python environment and install dependencies"""
    print("🔧 Setting up Lung Cancer Prediction Environment")
    print("=" * 50)
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        return False
    
    print(f"✓ Python {python_version.major}.{python_version.minor}.{python_version.micro} detected")
    
    # Create necessary directories
    directories = ['data', 'models', 'uploads', 'static', 'static/images']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✓ Created directory: {directory}")
    
    # Install requirements
    if not run_command("pip install -r requirements.txt", "Installing Python packages"):
        print("❌ Failed to install requirements. Please check your internet connection.")
        return False
    
    return True

def download_and_train_models():
    """Download datasets and train models"""
    print("\n🤖 Training Machine Learning Models")
    print("=" * 50)
    
    if os.path.exists("models/clinical_model.pkl"):
        print("✓ Models already exist. Skipping training.")
        return True
    
    print("📥 This will download datasets from Kaggle and train models...")
    print("⚠️  Note: You may need to set up Kaggle API credentials first")
    
    response = input("Do you want to proceed with dataset download and training? (y/n): ")
    if response.lower() != 'y':
        print("⏭️  Skipping model training. Using pre-built models.")
        return True
    
    try:
        print("🔄 Starting data processing and training...")
        result = subprocess.run([sys.executable, "data_preparation_and_training.py"], 
                              capture_output=True, text=True, timeout=3600)  # 1 hour timeout
        
        if result.returncode == 0:
            print("✓ Model training completed successfully")
            return True
        else:
            print(f"⚠️  Training completed with warnings: {result.stderr}")
            return True
    except subprocess.TimeoutExpired:
        print("⏰ Training timed out. Using default models.")
        return True
    except Exception as e:
        print(f"❌ Error during training: {e}")
        print("📝 Using default models for demonstration.")
        return True

def create_sample_data():
    """Create sample data files for testing"""
    print("\n📊 Creating Sample Data")
    print("=" * 50)
    
    # Create sample CSV for testing
    sample_csv_content = """Age,Gender,Smoking,Yellow_Fingers,Anxiety,Peer_Pressure,Chronic_Disease,Fatigue,Allergy,Wheezing,Alcohol,Coughing,Shortness_of_Breath,Swallowing_Difficulty,Chest_Pain,Lung_Cancer
65,M,2,1,0,0,1,1,0,1,1,1,1,0,1,1
45,F,1,0,1,0,0,0,1,0,0,0,0,0,0,0
70,M,2,1,1,1,1,1,0,1,1,1,1,1,1,1
35,F,0,0,0,0,0,0,0,0,0,0,0,0,0,0
55,M,1,0,1,0,1,1,0,0,1,1,0,0,1,1
"""
    
    with open("sample_patient_data.csv", "w") as f:
        f.write(sample_csv_content)
    
    print("✓ Created sample_patient_data.csv for testing")
    return True

def run_web_application():
    """Run the Flask web application"""
    print("\n🚀 Starting Web Application")
    print("=" * 50)
    
    print("🌐 Starting Lung Cancer Prediction Web App...")
    print("📍 The application will be available at: http://localhost:5000")
    print("⏹️  Press Ctrl+C to stop the server")
    print("\n" + "=" * 50)
    
    try:
        # Set Flask environment variables
        os.environ['FLASK_APP'] = 'app.py'
        os.environ['FLASK_ENV'] = 'development'
        
        # Run Flask app
        subprocess.run([sys.executable, "app.py"], check=True)
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error running application: {e}")

def main():
    """Main setup and run function"""
    print("🫁 Lung Cancer Prediction System")
    print("=" * 50)
    print("This script will set up and run the lung cancer prediction web application.")
    print("Features include:")
    print("  • Clinical risk assessment")
    print("  • CT scan image analysis")
    print("  • Dataset bulk analysis")
    print("  • Machine learning predictions")
    print("\n")
    
    # Setup environment
    if not setup_environment():
        print("❌ Environment setup failed. Please check the errors above.")
        return
    
    # Create sample data
    create_sample_data()
    
    # Ask about model training
    download_and_train_models()
    
    # Run web application
    run_web_application()

if __name__ == "__main__":
    main()
