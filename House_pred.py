from flask import Flask, request, jsonify
import pandas as pd
import pickle
import os

app = Flask(__name__)

# Load the LabelEncoders
with open(os.path.join('model', 'label_house_encoders.pkl'), 'rb') as f:
    label_encoders = pickle.load(f)

# Load the trained model
with open(os.path.join('model', 'gbr_house_model.pkl'), 'rb') as f:
    gb = pickle.load(f)

# Function to transform user input to match the model's input
def transform_input(user_input, label_encoders):
    user_input_encoded = user_input.copy()
    for col, encoder in label_encoders.items():
        if col in user_input.columns:
            user_input_encoded[col] = encoder.transform(user_input[[col]].values.ravel())
    return user_input_encoded


@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "Welcome to the House Price Prediction API. Use POST /predict to get predictions."
    })


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        required_features = [
            'POSTED_BY', 'UNDER_CONSTRUCTION', 'RERA', 'BHK_NO.',
            'BHK_OR_RK', 'SQUARE_FT', 'READY_TO_MOVE', 'RESALE',
            'LONGITUDE', 'LATITUDE', 'CITY'
        ]

        missing = [feature for feature in required_features if feature not in data]
        if missing:
            return jsonify({'error': f'Missing features: {", ".join(missing)}'}), 400

        user_input = pd.DataFrame([data])

        user_input_transformed = transform_input(user_input, label_encoders)

        prediction = gb.predict(user_input_transformed.values.reshape(1, -1))[0]

        return jsonify({'predicted_price_lakh': round(prediction, 2)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
