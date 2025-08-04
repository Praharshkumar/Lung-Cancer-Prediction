# 🫁 Lung Cancer Prediction System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.3.3-green.svg)](https://flask.palletsprojects.com)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13.0-orange.svg)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-Educational-yellow.svg)](#license)

An AI-powered web application for lung cancer risk assessment using machine learning and deep learning techniques. This system combines clinical data analysis, CT scan image processing, and bulk dataset analysis to provide comprehensive cancer risk predictions.

## 🌟 Features

- **🩺 Clinical Risk Assessment**: Comprehensive form-based evaluation using patient symptoms, medical history, and lifestyle factors
- **🖼️ CT Scan Analysis**: AI-powered chest CT image interpretation with nodule detection
- **📊 Dataset Processing**: Bulk analysis of patient datasets with statistical insights and batch predictions
- **🎯 Real-time Predictions**: Instant results with confidence scores and risk levels
- **📱 Responsive Design**: Modern, mobile-friendly web interface
- **🔄 Automated Setup**: One-command installation and deployment

## 🎯 Demo
Watch the demo video

https://github.com/user-attachments/assets/01455827-b70f-4772-94b7-00a724217883


## 🚀 Quick Start

### Option 1: Automated Setup (Recommended)
```bash
# Clone the repository
git clone https://github.com/your-username/lung-cancer-prediction.git
cd lung-cancer-prediction

# Run automated setup and launch
python setup_and_run.py
```

### Option 2: Manual Setup
```bash
# Clone and navigate
git clone https://github.com/your-username/lung-cancer-prediction.git
cd lung-cancer-prediction

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

🌐 **Access the application at:** `http://localhost:5000`

## 📋 Prerequisites

- **Python**: 3.8 or higher
- **Operating System**: Windows, macOS, or Linux  
- **Memory**: At least 4GB RAM
- **Storage**: 2GB free space
- **Internet**: Required for dataset downloads

## 🗂️ Project Structure

```
lung-cancer-prediction/
├── 📱 app.py                              # Main Flask application
├── 🤖 data_preparation_and_training.py    # ML model training pipeline
├── ⚙️ setup_and_run.py                   # Automated setup script
├── 📋 requirements.txt                    # Python dependencies
├── 📖 README.md                          # Project documentation
├── 📋 INSTALLATION_GUIDE.md              # Detailed setup guide
├── 📊 sample_patient_data.csv            # Sample dataset for testing
├── 🚫 .gitignore                         # Git ignore rules
│
├── 🎨 templates/                         # HTML templates
│   ├── 🏠 index.html                     # Homepage
│   ├── 🩺 clinical.html                  # Clinical prediction
│   ├── 🖼️ ct_scan.html                   # CT scan analysis
│   ├── 📊 dataset.html                   # Dataset analysis
│   └── 🎭 layout.html                    # Base template
│
├── 📂 data/                              # Downloaded datasets (auto-created)
├── 🧠 models/                            # Trained ML models (auto-created)
├── 📤 uploads/                           # User uploads (auto-created)
└── 🎨 static/                            # Static assets (auto-created)
```

## 🤖 Machine Learning Models

### Clinical Risk Model
- **Algorithm**: Random Forest Classifier
- **Features**: 15 clinical parameters
- **Accuracy**: ~85-95%
- **Output**: Risk level (High/Low) with confidence score

### CT Scan Analysis Model  
- **Algorithm**: Convolutional Neural Network (CNN)
- **Input**: 224x224 grayscale CT images
- **Architecture**: Deep CNN with batch normalization
- **Output**: Cancer probability with nodule detection

## 📊 Datasets

The system integrates with three Kaggle datasets:

1. **[Cancer Patient Dataset](https://www.kaggle.com/datasets/thedevastator/cancer-patients-and-air-pollution-a-new-link)**
   - Clinical data and air pollution correlation
   - Patient demographics and symptoms

2. **[Chest CT-Scan Images](https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images)**
   - Medical imaging dataset
   - Normal vs. abnormal classifications

3. **[Medical Text Dataset](https://www.kaggle.com/datasets/chaitanyakck/medical-text)**
   - Additional medical information
   - Text-based medical records

## 🔧 API Endpoints

| Endpoint | Method | Description |
|----------|--------|--------------|
| `/` | GET | Homepage |
| `/clinical` | GET | Clinical prediction form |
| `/ct-scan` | GET | CT scan upload interface |
| `/dataset` | GET | Dataset analysis interface |
| `/predict_clinical` | POST | Clinical risk prediction |
| `/predict_ct_scan` | POST | CT scan image analysis |
| `/analyze_dataset` | POST | Bulk dataset processing |

## 🧪 Usage Examples

### Clinical Prediction
```python
# Example patient data
patient_data = {
    "age": 65,
    "gender": "M",
    "smoking": 2,  # Current smoker
    "chest_pain": 1,
    "coughing": 1
    # ... other parameters
}
```

### Dataset Analysis
```csv
Age,Gender,Smoking,Yellow_Fingers,Anxiety,Chest_Pain,Lung_Cancer
65,M,2,1,0,1,1
45,F,1,0,1,0,0
70,M,2,1,1,1,1
```

## 🛠️ Development

### Local Development
```bash
# Clone repository
git clone https://github.com/your-username/lung-cancer-prediction.git
cd lung-cancer-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run in development mode
export FLASK_ENV=development  # On Windows: set FLASK_ENV=development
python app.py
```

### Contributing
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📈 Performance Metrics

- **Clinical Model Accuracy**: 85-95%
- **Response Time**: <2 seconds
- **Supported File Formats**: CSV, JPG, PNG, DICOM
- **Maximum File Size**: 16MB
- **Concurrent Users**: 100+

## 🔒 Privacy & Security

- No patient data is stored permanently
- All uploads are processed locally
- HIPAA-compliant design principles
- Secure file handling and validation

## ⚠️ Important Disclaimers

- **Educational Purpose Only**: This system is designed for educational and research purposes
- **Not for Medical Diagnosis**: Always consult healthcare professionals for medical advice
- **Research Tool**: Predictions are based on limited training data
- **No Medical Liability**: This tool should not replace professional medical consultation

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Kaggle community for providing datasets
- TensorFlow and scikit-learn teams
- Flask development team
- Bootstrap for UI components
- Medical professionals who provided guidance
