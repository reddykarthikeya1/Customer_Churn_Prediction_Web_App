# Customer Churn Prediction Web App

A full-stack machine learning web application to predict customer churn for a telecom company. Built with Flask, scikit-learn, and Docker, this project demonstrates the end-to-end process of data analysis, model building, and deployment.

## 🚀 Features

- **Interactive Web Interface:** User-friendly form to input customer details and get instant churn predictions.
- **Robust ML Model:** Random Forest classifier trained with SMOTEENN to handle class imbalance.
- **Data Pipeline:** Modular preprocessing and feature engineering.
- **Dockerized:** Easily deployable anywhere with Docker.
- **Visualization:** Notebooks for EDA and model development included.

## 🗂️ Repository Structure

```
.
├── app.py                  # Flask application entry point
├── churn_model.pkl         # Trained machine learning model
├── cleaned_churn_data.csv  # Cleaned dataset used for modeling
├── requirements.txt        # Python dependencies
├── dockerfile              # Docker configuration
├── data_pipeline/          # Data preprocessing and prediction logic
│   ├── __init__.py
│   ├── config.py
│   ├── data_pipeline.py
│   ├── model_handler.py
│   └── predictor.py
├── templates/
│   └── index.html          # Web app HTML template
├── notebooks/              # Jupyter notebooks for EDA and modeling
│   ├── Churn_EDA.ipynb
│   └── Churn_Model.ipynb
├── data/                   # Raw data files
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
└── README.md
```

## 🏃‍♂️ Quickstart

### 1. Clone the repository

```bash
git clone https://github.com/reddykarthikeya1/Customer_Churn_Prediction_Web_App.git
cd Customer_Churn_Prediction_Web_App
```

### 2. Build and run with Docker

```bash
docker build -t churn-app .
docker run -p 5000:5000 churn-app
```

The app will be available at [http://localhost:5000](http://localhost:5000).

### 3. Run locally (without Docker)

```bash
pip install -r requirements.txt
python app.py
```

## 📝 Usage

- Fill in the customer details on the web form.
- Click **Predict Churn** to see if the customer is likely to churn, along with the model's confidence.

## 📊 Notebooks

- **Churn_EDA.ipynb:** Exploratory data analysis and visualization.
- **Churn_Model.ipynb:** Model training, evaluation, and export.

## 🛠️ Tech Stack

- Python, Flask, scikit-learn, pandas, imbalanced-learn
- HTML/CSS (Jinja2 templating)
- Docker

## 📦 Model Details

- **Algorithm:** Random Forest Classifier
- **Imbalance Handling:** SMOTEENN
- **Features:** All major customer attributes, one-hot encoded as needed

## 🧑‍💻 Author

Developed by [Karthikeya Reddy](https://github.com/reddykarthikeya1)

## ⭐️ Star this repo if you found it useful!

---

**License:** MIT
