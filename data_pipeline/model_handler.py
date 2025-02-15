import joblib

def load_model(model_path="churn_model.pkl"):
    """
    Loads and returns the trained churn prediction model.
    """
    model = joblib.load(model_path)
    return model
