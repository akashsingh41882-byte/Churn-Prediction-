# Customer Churn Prediction

Predicts whether a telecom customer will churn (cancel their subscription) using account, billing, and service data. End-to-end ML project: data preprocessing, model training/evaluation, and an interactive Streamlit app for live predictions.

## Why this project

Churn prediction is one of the most common real-world ML use cases (telecom, SaaS, banking). This repo demonstrates a full pipeline you can point to in interviews or a portfolio: clean data handling, a reproducible training script, saved model artifacts, and a deployable app — not just a notebook.

## Dataset

IBM Telco Customer Churn dataset (~7,000 customers, 20 features: tenure, contract type, monthly charges, internet service, payment method, etc.). The training script downloads it automatically from a public source, so no manual download is needed.

## Project structure

```
churn-prediction/
├── data/                  # dataset lands here after first run
├── models/                # trained model + encoders saved here
├── src/
│   ├── preprocess.py      # data cleaning & feature encoding
│   ├── train.py           # trains and evaluates the model
│   └── predict.py         # batch predictions from a CSV
├── app.py                 # Streamlit web app for live predictions
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

```bash
git clone https://github.com/akashsingh41882-byte/churn-prediction.git
cd churn-prediction
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

**1. Train the model**
```bash
python src/train.py
```
Downloads the dataset, preprocesses it, trains a Random Forest classifier, prints evaluation metrics, and saves `models/churn_model.pkl` + `models/encoders.pkl`.

**2. Run the web app**
```bash
streamlit run app.py
```
Opens a browser UI where you enter a customer's details and get a churn probability instantly.

**3. Batch predict on a CSV**
```bash
python src/predict.py --input data/new_customers.csv --output data/predictions.csv
```

## Results

Random Forest on the held-out test set (typical run — exact numbers vary slightly by seed):

| Metric    | Score |
|-----------|-------|
| Accuracy  | ~0.80 |
| Precision | ~0.66 |
| Recall    | ~0.51 |
| F1        | ~0.58 |
| ROC-AUC   | ~0.84 |

Top churn predictors: contract type (month-to-month), tenure, and monthly charges.

## Next steps / ideas to extend

- Try XGBoost or LightGBM and compare
- Add SHAP values for per-prediction explainability in the app
- Handle class imbalance with SMOTE
- Containerize with Docker and deploy the Streamlit app

## License

MIT
