
from flask import Flask, request, jsonify
import pandas as pd
import joblib

model = joblib.load("gb_pipeline.joblib")
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    input_data = request.get_json()
    df = pd.DataFrame([input_data])
    prediction = model.predict(df)
    return jsonify({"prediction": float(prediction[0])})

@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    input_data = request.get_json()
    df = pd.DataFrame(input_data)
    predictions = model.predict(df)
    return jsonify({"predictions": predictions.tolist()})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
