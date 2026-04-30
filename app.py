from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("model.pkl")

@app.route("/")
def home():
    return "Model is running 🚀"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    input_data = np.array(data["input"]).reshape(1, -1)
    prediction = model.predict(input_data)

    return jsonify({"prediction": prediction.tolist()})
