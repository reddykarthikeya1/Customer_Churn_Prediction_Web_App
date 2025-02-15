import pandas as pd
from data_pipeline.config import COLUMNS

# Mapping for binary variables
binary_map = {"Yes": 1, "No": 0}

def preprocess_input(form_data):
    """
    Extracts and converts form data into a dictionary.
    """
    input_data = {
        "SeniorCitizen": int(form_data.get('query1', 0)),
        "MonthlyCharges": float(form_data.get('query2', 0)),
        "TotalCharges": float(form_data.get('query3', 0)),
        "gender": form_data.get('query4', ''),
        "Partner": form_data.get('query5', 'No'),
        "Dependents": form_data.get('query6', 'No'),
        "PhoneService": form_data.get('query7', 'No'),
        "MultipleLines": form_data.get('query8', ''),
        "InternetService": form_data.get('query9', ''),
        "OnlineSecurity": form_data.get('query10', ''),
        "OnlineBackup": form_data.get('query11', ''),
        "DeviceProtection": form_data.get('query12', ''),
        "TechSupport": form_data.get('query13', ''),
        "StreamingTV": form_data.get('query14', ''),
        "StreamingMovies": form_data.get('query15', ''),
        "Contract": form_data.get('query16', ''),
        "PaperlessBilling": form_data.get('query17', 'No'),
        "PaymentMethod": form_data.get('query18', ''),
        "tenure": int(form_data.get('query19', 0))
    }
    
    # Convert binary responses to 0/1
    for key in ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]:
        input_data[key] = binary_map.get(input_data[key], 0)
    
    return input_data

def create_input_dataframe(input_data):
    """
    Creates a DataFrame with the proper columns and fills in the values.
    """
    df_input = pd.DataFrame([[0] * len(COLUMNS)], columns=COLUMNS)
    
    # Set numeric columns
    df_input.at[0, "SeniorCitizen"] = input_data["SeniorCitizen"]
    df_input.at[0, "MonthlyCharges"] = input_data["MonthlyCharges"]
    df_input.at[0, "TotalCharges"] = input_data["TotalCharges"]
    df_input.at[0, "tenure"] = input_data["tenure"]
    
    # Dummy coding for categorical variables
    
    # Gender
    if input_data["gender"] == "Female":
        df_input.at[0, "gender_Female"] = 1
        df_input.at[0, "gender_Male"] = 0
    elif input_data["gender"] == "Male":
        df_input.at[0, "gender_Female"] = 0
        df_input.at[0, "gender_Male"] = 1

    # MultipleLines
    if input_data["MultipleLines"] == "No":
        df_input.at[0, "MultipleLines_No"] = 1
    elif input_data["MultipleLines"] == "No phone service":
        df_input.at[0, "MultipleLines_No phone service"] = 1
    elif input_data["MultipleLines"] == "Yes":
        df_input.at[0, "MultipleLines_Yes"] = 1

    # InternetService
    if input_data["InternetService"] == "DSL":
        df_input.at[0, "InternetService_DSL"] = 1
    elif input_data["InternetService"] == "Fiber optic":
        df_input.at[0, "InternetService_Fiber optic"] = 1
    elif input_data["InternetService"] == "No":
        df_input.at[0, "InternetService_No"] = 1

    # Contract
    if input_data["Contract"] == "Month-to-month":
        df_input.at[0, "Contract_Month-to-month"] = 1
    elif input_data["Contract"] == "One year":
        df_input.at[0, "Contract_One year"] = 1
    elif input_data["Contract"] == "Two year":
        df_input.at[0, "Contract_Two year"] = 1

    # PaymentMethod
    if input_data["PaymentMethod"] == "Bank transfer (automatic)":
        df_input.at[0, "PaymentMethod_Bank transfer (automatic)"] = 1
    elif input_data["PaymentMethod"] == "Credit card (automatic)":
        df_input.at[0, "PaymentMethod_Credit card (automatic)"] = 1
    elif input_data["PaymentMethod"] == "Electronic check":
        df_input.at[0, "PaymentMethod_Electronic check"] = 1
    elif input_data["PaymentMethod"] == "Mailed check":
        df_input.at[0, "PaymentMethod_Mailed check"] = 1

    # You can add additional dummy coding for:
    # OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies
    # following the same structure as above.

    return df_input

def make_prediction(model, df_input):
    """
    Uses the provided model to make a prediction and return both the prediction and its probability.
    """
    prediction = model.predict(df_input)
    proba = model.predict_proba(df_input)[:, 1]
    return prediction[0], proba[0]
