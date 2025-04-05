from flask import Flask, request, jsonify
import numpy as np
import pickle
from huggingface_hub import hf_hub_download

app = Flask(__name__)

# Constants
MODEL_REPO = "PAyus77/house-price-model"
GBR_MODEL_FILE = "gbr_house_model.pkl"
ENCODER_FILE = "label_house_encoders.pkl"

# Load Model & Encoder
gbr_model_path = hf_hub_download(repo_id=MODEL_REPO, filename=GBR_MODEL_FILE)
encoder_path = hf_hub_download(repo_id=MODEL_REPO, filename=ENCODER_FILE)

with open(gbr_model_path, 'rb') as f:
    model = pickle.load(f)

with open(encoder_path, 'rb') as f:
    label_encoders = pickle.load(f)


def preprocess_input(data):
    processed = data.copy()
    for col, encoder in label_encoders.items():
        if col in processed:
            processed[col] = encoder.transform([processed[col]])[0]
    return np.array(list(processed.values())).reshape(1, -1)


@app.route("/", methods=["GET"])
def home():
    return "House Price Prediction API Running 🚀"


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    try:
        processed = preprocess_input(data)
        prediction = model.predict(processed)
        return jsonify({"Predicted Price": float(prediction[0])})
    except Exception as e:
        return jsonify({"Error": str(e)})


if __name__ == "__main__":
    app.run(debug=True)
