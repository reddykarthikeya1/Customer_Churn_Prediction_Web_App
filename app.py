# app.py

from flask import Flask, request, render_template
from data_pipeline.model_handler import load_model
from data_pipeline.predictor import preprocess_input, create_input_dataframe, make_prediction

app = Flask(__name__)

# Load the trained model once when the application starts
model = load_model("churn_model.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        form_data = request.form
        input_data = preprocess_input(form_data)
        df_input = create_input_dataframe(input_data)
        prediction, proba = make_prediction(model, df_input)
        
        if prediction == 1:
            output1 = "This customer is likely to be churned!!"
        else:
            output1 = "This customer is likely to continue!!"
        output2 = "Confidence: {:.2f}%".format(proba * 100)
        
        return render_template('index.html', output1=output1, output2=output2)
    
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
