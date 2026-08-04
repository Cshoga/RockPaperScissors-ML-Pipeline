# RockPaperScissors-ML-Pipeline
## Project Description

This project is a Machine Learning Classification Pipeline that predicts whether an uploaded image represents Rock, Paper, or Scissors.

The project was developed as part of the Machine Learning Pipeline assignment. It demonstrates the complete lifecycle of an ML model, from data acquisition to deployment and monitoring.

## Dataset

The dataset was obtained from Kaggle.

It contains three image classes:
- Rock
- Paper
- Scissors

Additional personal images were also included to improve model performance.

## Technologies Used
- Python
- TensorFlow / Keras
- FastAPI
- Streamlit
- Scikit-learn
- Matplotlib
- NumPy
- Pillow
- GitHub
  
## Features
- Image Classification
- Model Evaluation
- FastAPI REST API
- Streamlit User Interface
- Upload Images for Retraining
- Trigger Model Retraining
- Dataset Visualizations
- API Health Monitoring

## Repository Structure
RockPaperScissors-ML-Pipeline/
│
├── README.md
│
├── notebook/
│   └── RockPaperScissors_ML_Pipeline.ipynb
│
├── src/
│   ├── api.py
│   ├── model.py
│   ├── prediction.py
│   ├── preprocessing.py
│   └── retrain.py
│
├── ui/
│   └── app.py
│
├── data/
│   ├── train/
│   ├── test/
│   └── new_data/
│
├── images/
│   ├── class_distribution.png
│   ├── training_accuracy.png
│   ├── training_loss.png
│   └── confusion_matrix.png
│
├── models/
│   └── rock_paper_scissors_model.keras
│
├── requirements.txt
├── render.yaml
├── locust/
│   └── locustfile.py
│
└── .gitignore

## Installation
### Clone the repository

