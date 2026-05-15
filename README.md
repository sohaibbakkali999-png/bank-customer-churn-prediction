# Bank Customer Churn Prediction

This project is a Machine Learning web application that predicts whether a bank customer is likely to leave the bank or stay.

## Project Description

The application uses customer information such as credit score, country, gender, age, balance, number of products, credit card status, activity status, and estimated salary to predict customer churn.

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Joblib
- Streamlit

## Machine Learning Model

The best model used in this project is Random Forest.

## Files

- `app.py`: Streamlit web application
- `best_churn_model_with_scalerr.pkl`: saved machine learning model and scaler
- `feature_columns.pkl`: saved feature columns used during training
- `requirements.txt`: required Python libraries

## How to Run the App Locally

```bash
pip install -r requirements.txt
streamlit run app.py