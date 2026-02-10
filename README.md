# Education ML Pipeline

An end-to-end machine learning pipeline for educational data analysis and prediction. This project demonstrates a complete MLOps workflow from data ingestion to model deployment, built with industry best practices.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Pipeline Components](#pipeline-components)
- [Technology Stack](#technology-stack)
- [Model Performance](#model-performance)
- [Web Application](#web-application)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project implements a production-ready machine learning pipeline designed to analyze and predict educational outcomes. The pipeline follows MLOps best practices, including modular code organization, automated data processing, model training, evaluation, and deployment through a web interface.

## ✨ Features

- **End-to-End ML Pipeline**: Complete workflow from data ingestion to model deployment
- **Modular Architecture**: Well-structured codebase with separate components for each pipeline stage
- **Web Interface**: Interactive Flask application for predictions
- **Automated Workflows**: Streamlined data processing and model training
- **Reproducibility**: Consistent results through versioned dependencies and structured code
- **Scalable Design**: Easy to extend and modify for different datasets and use cases

## 📁 Project Structure

```
Education-ml-pipeline/
│
├── artifacts/              # Saved models, preprocessors, and pipeline artifacts
│
├── notebook/              # Jupyter notebooks for EDA and experimentation
│   └── data/             # Dataset storage
│
├── src/                   # Source code for the ML pipeline
│   ├── components/       # Pipeline components (ingestion, transformation, training)
│   ├── pipeline/         # Training and prediction pipelines
│   └── utils.py          # Utility functions
│
├── templates/            # HTML templates for web application
│
├── app.py               # Flask web application
├── setup.py             # Package setup and installation
├── requirements.txt     # Python dependencies
└── README.md           # Project documentation
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/akshatkh18/Education-ml-pipeline.git
   cd Education-ml-pipeline
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install the package**
   ```bash
   pip install -e .
   ```

## 💻 Usage

### Training the Model

Run the complete pipeline to train the model:

```bash
python src/pipeline/training_pipeline.py
```

This will:
- Ingest and validate data
- Perform feature engineering and transformation
- Train the machine learning model
- Save artifacts for deployment

### Making Predictions

#### Using the Web Application

1. Start the Flask server:
   ```bash
   python app.py
   ```

2. Open your browser and navigate to:
   ```
   http://localhost:5000
   ```

3. Enter the required features and get predictions through the web interface

#### Using the Prediction Pipeline

```python
from src.pipeline.prediction_pipeline import PredictionPipeline

# Create prediction pipeline instance
pipeline = PredictionPipeline()

# Make predictions
results = pipeline.predict(input_data)
```

## 🔧 Pipeline Components

### 1. Data Ingestion
- Loads raw data from various sources
- Performs initial validation
- Splits data into training and testing sets

### 2. Data Transformation
- Handles missing values
- Encodes categorical variables
- Scales numerical features
- Creates feature engineering pipelines

### 3. Model Training
- Trains machine learning models
- Performs hyperparameter tuning
- Evaluates model performance
- Saves the best performing model

### 4. Model Evaluation
- Generates performance metrics
- Creates visualization of results
- Validates model on test data

### 5. Deployment
- Flask web application
- RESTful API endpoints
- User-friendly interface

## 🛠️ Technology Stack

- **Programming Language**: Python 3.8+
- **Web Framework**: Flask
- **Machine Learning**: scikit-learn
- **Data Processing**: Pandas, NumPy
- **Data Visualization**: Matplotlib, Seaborn
- **Notebook Environment**: Jupyter Notebook
- **Version Control**: Git

## 📊 Model Performance

Performance metrics and evaluation results can be found in the `artifacts/` directory after training. The pipeline tracks:

- Model accuracy
- Precision and recall
- F1 score
- Confusion matrix
- Feature importance

## 🌐 Web Application

The Flask web application provides an intuitive interface for making predictions. Features include:

- Simple input form for feature values
- Real-time predictions
- Result visualization
- Responsive design

### API Endpoints

- `GET /`: Home page with prediction form
- `POST /predict`: Submit data and receive predictions

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add docstrings to functions and classes
- Update tests for new features
- Update documentation as needed



## 👤 Author

**Akshat Khare**

- GitHub: [@akshatkh18](https://github.com/akshatkh18)

## 🙏 Acknowledgments

- Thanks to all contributors who have helped shape this project
- Inspired by MLOps best practices and industry standards
- Built as part of learning and implementing production ML systems

## 📮 Contact

For questions, suggestions, or issues, please open an issue on GitHub or reach out through the repository.

---

**Note**: This project is for educational and demonstration purposes. Ensure proper data privacy and security measures when deploying to production environments.
