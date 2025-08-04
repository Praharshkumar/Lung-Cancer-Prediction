# Lung Cancer Prediction Project - Installation Guide

## 📋 Overview
This is a comprehensive AI-powered Lung Cancer Prediction system that uses machine learning and deep learning to assess cancer risk through:
- Clinical data analysis
- CT scan image processing
- Dataset bulk analysis

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Windows/macOS/Linux
- At least 4GB free disk space
- Internet connection for downloading datasets

### Installation Steps

1. **Extract the Project**
   ```bash
   unzip lung_cancer_prediction_project.zip
   cd lung-cancer-prediction
   ```

2. **Run the Setup Script (Recommended)**
   ```bash
   python setup_and_run.py
   ```
   This will automatically:
   - Install all dependencies
   - Create necessary directories
   - Download and process datasets (optional)
   - Start the web application

3. **Manual Installation (Alternative)**
   ```bash
   # Install dependencies
   pip install -r requirements.txt
   
   # Create directories
   mkdir data models uploads static
   
   # Run the application
   python app.py
   ```

## 📁 Project Structure
```
lung-cancer-prediction/
├── app.py                              # Main Flask application
├── data_preparation_and_training.py    # Dataset processing and ML training
├── setup_and_run.py                   # Automated setup script
├── requirements.txt                    # Python dependencies
├── README.md                          # Project documentation
├── INSTALLATION_GUIDE.md              # This file
├── sample_patient_data.csv            # Sample data for testing
│
├── templates/                         # HTML templates
│   ├── index.html                     # Homepage
│   ├── clinical.html                  # Clinical prediction page
│   ├── ct_scan.html                   # CT scan analysis page
│   ├── dataset.html                   # Dataset analysis page
│   └── layout.html                    # Base template
│
├── data/                              # Downloaded datasets (auto-created)
├── models/                            # Trained ML models (auto-created)
├── uploads/                           # Uploaded files (auto-created)
└── static/                            # Static assets (auto-created)
```

## 🌐 Usage

### Web Interface
1. Open your browser and go to `http://localhost:5000`
2. Navigate through different prediction modules:
   - **Clinical Prediction**: Enter patient symptoms and medical history
   - **CT Scan Analysis**: Upload chest CT images for analysis
   - **Dataset Analysis**: Upload CSV files for bulk predictions

### Features
- **Clinical Risk Assessment**: Comprehensive form-based risk evaluation
- **Image Analysis**: AI-powered CT scan interpretation
- **Bulk Processing**: CSV dataset analysis with statistics
- **Interactive UI**: Modern, responsive web interface
- **Real-time Predictions**: Instant results with confidence scores

## 🤖 Machine Learning Models

### Clinical Model
- **Algorithm**: Random Forest Classifier
- **Input**: Patient demographics, symptoms, medical history
- **Output**: Risk level (High/Low) with confidence score

### CT Scan Model
- **Algorithm**: Convolutional Neural Network (CNN)
- **Input**: Chest CT scan images
- **Output**: Cancer probability with nodule detection

### Datasets Used
1. **Cancer Patient Dataset**: Clinical data and air pollution correlation
2. **Chest CT-Scan Images**: Medical imaging dataset
3. **Medical Text Dataset**: Additional medical information

## ⚙️ Configuration

### Kaggle API Setup (Optional for dataset download)
1. Create account at kaggle.com
2. Go to Account → API → Create New API Token
3. Place `kaggle.json` in `~/.kaggle/` directory
4. Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

### Environment Variables
```bash
FLASK_APP=app.py
FLASK_ENV=development
FLASK_DEBUG=1
```

## 🔧 Troubleshooting

### Common Issues

1. **Import Errors**
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Check Python version: `python --version` (should be 3.8+)

2. **Dataset Download Issues**
   - Verify Kaggle API credentials
   - Check internet connection
   - Use sample data if datasets unavailable

3. **Model Training Errors**
   - Skip training if encountering issues
   - Pre-trained models will be used automatically
   - Check available disk space

4. **Port Already in Use**
   - Change port in app.py: `app.run(port=5001)`
   - Or kill existing processes using port 5000

## 📊 Sample Data
The project includes `sample_patient_data.csv` for testing:
- 5 sample patients with various risk profiles
- All required columns for clinical prediction
- Ready to use with dataset analysis feature

## 🔬 Model Performance
- **Clinical Model Accuracy**: ~85-95%
- **CT Scan Model**: Deep learning with transfer learning
- **Processing Speed**: Real-time predictions (<2 seconds)

## 📚 Technical Details

### Dependencies
- **Backend**: Flask, scikit-learn, TensorFlow/Keras
- **Data Processing**: pandas, NumPy, OpenCV
- **Visualization**: matplotlib, seaborn
- **Frontend**: HTML5, Bootstrap 5, JavaScript

### API Endpoints
- `POST /predict_clinical`: Clinical risk prediction
- `POST /predict_ct_scan`: CT scan image analysis
- `POST /analyze_dataset`: Bulk dataset processing

## 🚨 Important Notes
- This system is for **educational and research purposes only**
- Not intended for actual medical diagnosis
- Always consult healthcare professionals for medical advice
- Predictions are based on limited training data

## 📞 Support
For issues or questions:
1. Check this installation guide
2. Review error messages carefully
3. Ensure all prerequisites are met
4. Try running `python setup_and_run.py` again

## 🎯 Next Steps
After installation:
1. Test with provided sample data
2. Upload your own datasets
3. Explore different prediction methods
4. Analyze results and model performance

---
**Happy Predicting! 🫁💻**
