import pandas as pd
import joblib
from flask import Flask, request, render_template

app = Flask(__name__)

# List of expected columns (make sure this exactly matches your training features)
columns = [
    "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService", "PaperlessBilling", 
    "MonthlyCharges", "TotalCharges", "gender_Female", "gender_Male", "MultipleLines_No", 
    "MultipleLines_No phone service", "MultipleLines_Yes", "InternetService_DSL", 
    "InternetService_Fiber optic", "InternetService_No", "OnlineSecurity_No", 
    "OnlineSecurity_No internet service", "OnlineSecurity_Yes", "OnlineBackup_No", 
    "OnlineBackup_No internet service", "OnlineBackup_Yes", "DeviceProtection_No", 
    "DeviceProtection_No internet service", "DeviceProtection_Yes", "TechSupport_No", 
    "TechSupport_No internet service", "TechSupport_Yes", "StreamingTV_No", 
    "StreamingTV_No internet service", "StreamingTV_Yes", "StreamingMovies_No", 
    "StreamingMovies_No internet service", "StreamingMovies_Yes", "Contract_Month-to-month", 
    "Contract_One year", "Contract_Two year", "PaymentMethod_Bank transfer (automatic)", 
    "PaymentMethod_Credit card (automatic)", "PaymentMethod_Electronic check", 
    "PaymentMethod_Mailed check"
]

# Load the trained model (saved with joblib)
model = joblib.load("churn_model.pkl")

@app.route("/")
def loadPage():
    return render_template('index.html')

@app.route("/", methods=['POST'])
def predict():
    # Retrieve form inputs (all as strings)
    # For example, assume the form fields are named query1, query2, ..., query19
    # and they correspond to: SeniorCitizen, MonthlyCharges, TotalCharges, gender, Partner, 
    # Dependents, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, 
    # DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling,
    # PaymentMethod, tenure
    input_data = {
        "SeniorCitizen": request.form['query1'],  # e.g. "0" or "1"
        "MonthlyCharges": request.form['query2'],  # e.g. "70.5"
        "TotalCharges": request.form['query3'],    # e.g. "850.0"
        "gender": request.form['query4'],          # e.g. "Male" or "Female"
        "Partner": request.form['query5'],         # "Yes" or "No"
        "Dependents": request.form['query6'],      # "Yes" or "No"
        "PhoneService": request.form['query7'],      # "Yes" or "No"
        "MultipleLines": request.form['query8'],     # e.g. "Yes", "No", or "No phone service"
        "InternetService": request.form['query9'],   # e.g. "DSL", "Fiber optic", "No"
        "OnlineSecurity": request.form['query10'],   # e.g. "Yes", "No", "No internet service"
        "OnlineBackup": request.form['query11'],     # similar to OnlineSecurity
        "DeviceProtection": request.form['query12'], # similar options
        "TechSupport": request.form['query13'],      # similar options
        "StreamingTV": request.form['query14'],      # similar options
        "StreamingMovies": request.form['query15'],  # similar options
        "Contract": request.form['query16'],         # e.g. "Month-to-month", "One year", "Two year"
        "PaperlessBilling": request.form['query17'], # "Yes" or "No"
        "PaymentMethod": request.form['query18'],    # e.g. "Electronic check", etc.
        "tenure": request.form['query19']            # e.g. "12"
    }

    # Convert the numeric values appropriately
    input_data["SeniorCitizen"] = int(input_data["SeniorCitizen"])
    input_data["MonthlyCharges"] = float(input_data["MonthlyCharges"])
    input_data["TotalCharges"] = float(input_data["TotalCharges"])
    input_data["tenure"] = int(input_data["tenure"])

    # Map binary variables ("Yes"/"No") to 1/0
    binary_map = {"Yes": 1, "No": 0}
    for key in ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]:
        input_data[key] = binary_map.get(input_data[key], input_data[key])
    
    # Create an empty DataFrame with all expected columns, initialized to zero
    df_input = pd.DataFrame([[0]*len(columns)], columns=columns)
    
    # Now fill in the numeric columns directly:
    df_input.at[0, "SeniorCitizen"] = input_data["SeniorCitizen"]
    df_input.at[0, "MonthlyCharges"] = input_data["MonthlyCharges"]
    df_input.at[0, "TotalCharges"] = input_data["TotalCharges"]
    # Optionally, if you want to keep the original 'tenure' value, set it here:
    df_input.at[0, "tenure"] = input_data["tenure"]
    
    # Next, fill in the dummy variables for categorical fields.
    # For each categorical feature, set the corresponding column to 1.
    # Example for gender:
    if input_data["gender"] == "Female":
        df_input.at[0, "gender_Female"] = 1
        df_input.at[0, "gender_Male"] = 0
    elif input_data["gender"] == "Male":
        df_input.at[0, "gender_Female"] = 0
        df_input.at[0, "gender_Male"] = 1

    # Similarly for MultipleLines
    if input_data["MultipleLines"] == "No":
        df_input.at[0, "MultipleLines_No"] = 1
        df_input.at[0, "MultipleLines_No phone service"] = 0
        df_input.at[0, "MultipleLines_Yes"] = 0
    elif input_data["MultipleLines"] == "No phone service":
        df_input.at[0, "MultipleLines_No phone service"] = 1
        df_input.at[0, "MultipleLines_No"] = 0
        df_input.at[0, "MultipleLines_Yes"] = 0
    elif input_data["MultipleLines"] == "Yes":
        df_input.at[0, "MultipleLines_Yes"] = 1
        df_input.at[0, "MultipleLines_No"] = 0
        df_input.at[0, "MultipleLines_No phone service"] = 0

    # Do the same for InternetService:
    if input_data["InternetService"] == "DSL":
        df_input.at[0, "InternetService_DSL"] = 1
        df_input.at[0, "InternetService_Fiber optic"] = 0
        df_input.at[0, "InternetService_No"] = 0
    elif input_data["InternetService"] == "Fiber optic":
        df_input.at[0, "InternetService_Fiber optic"] = 1
        df_input.at[0, "InternetService_DSL"] = 0
        df_input.at[0, "InternetService_No"] = 0
    elif input_data["InternetService"] == "No":
        df_input.at[0, "InternetService_No"] = 1
        df_input.at[0, "InternetService_DSL"] = 0
        df_input.at[0, "InternetService_Fiber optic"] = 0

    # For Contract:
    if input_data["Contract"] == "Month-to-month":
        df_input.at[0, "Contract_Month-to-month"] = 1
        df_input.at[0, "Contract_One year"] = 0
        df_input.at[0, "Contract_Two year"] = 0
    elif input_data["Contract"] == "One year":
        df_input.at[0, "Contract_One year"] = 1
        df_input.at[0, "Contract_Month-to-month"] = 0
        df_input.at[0, "Contract_Two year"] = 0
    elif input_data["Contract"] == "Two year":
        df_input.at[0, "Contract_Two year"] = 1
        df_input.at[0, "Contract_Month-to-month"] = 0
        df_input.at[0, "Contract_One year"] = 0

    # For PaymentMethod:
    if input_data["PaymentMethod"] == "Bank transfer (automatic)":
        df_input.at[0, "PaymentMethod_Bank transfer (automatic)"] = 1
        df_input.at[0, "PaymentMethod_Credit card (automatic)"] = 0
        df_input.at[0, "PaymentMethod_Electronic check"] = 0
        df_input.at[0, "PaymentMethod_Mailed check"] = 0
    elif input_data["PaymentMethod"] == "Credit card (automatic)":
        df_input.at[0, "PaymentMethod_Credit card (automatic)"] = 1
        df_input.at[0, "PaymentMethod_Bank transfer (automatic)"] = 0
        df_input.at[0, "PaymentMethod_Electronic check"] = 0
        df_input.at[0, "PaymentMethod_Mailed check"] = 0
    elif input_data["PaymentMethod"] == "Electronic check":
        df_input.at[0, "PaymentMethod_Electronic check"] = 1
        df_input.at[0, "PaymentMethod_Bank transfer (automatic)"] = 0
        df_input.at[0, "PaymentMethod_Credit card (automatic)"] = 0
        df_input.at[0, "PaymentMethod_Mailed check"] = 0
    elif input_data["PaymentMethod"] == "Mailed check":
        df_input.at[0, "PaymentMethod_Mailed check"] = 1
        df_input.at[0, "PaymentMethod_Bank transfer (automatic)"] = 0
        df_input.at[0, "PaymentMethod_Credit card (automatic)"] = 0
        df_input.at[0, "PaymentMethod_Electronic check"] = 0

    # (Repeat similar assignments for other categorical features like OnlineSecurity, 
    # OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies)
    # You'll need to fill in these based on how they were dummy-encoded during training.
    
    # For brevity, let's assume you have filled in all necessary dummy columns.

    # Now predict using the model
    prediction = model.predict(df_input)
    proba = model.predict_proba(df_input)[:, 1]
    
    if prediction[0] == 1:
        output1 = "This customer is likely to be churned!!"
    else:
        output1 = "This customer is likely to continue!!"
    output2 = "Confidence: {:.2f}%".format(proba[0] * 100)
    
    return render_template('index.html', output1=output1, output2=output2)

if __name__ == '__main__':
    app.run()
