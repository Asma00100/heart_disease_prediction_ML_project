from flask import Flask, render_template, request, jsonify
import joblib  # For loading a pre-trained model and scaler
import numpy as np
import os

app = Flask(__name__)

# Load your pre-trained ML model and scaler
MODEL_PATH = 'm.pkl'
SCALER_PATH = 's.pkl'

if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    raise FileNotFoundError("Model or scaler file not found. Please check 'm.pkl' and 's.pkl'.")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract data from the form
        data = {
            'gender': int(request.form.get('gender', 0)),
            'height': float(request.form.get('height', 0)),
            'weight': float(request.form.get('weight', 0)),
            'ap_hi': int(request.form.get('ap_hi', 0)),
            'ap_lo': int(request.form.get('ap_lo', 0)),
            'cholesterol': int(request.form.get('cholesterol', 0)),
            'gluc': int(request.form.get('gluc', 0)),
            'smoke': int(request.form.get('smoke', 0)),
            'alco': int(request.form.get('alco', 0)),
            'active': int(request.form.get('active', 0)),
            'age_years': int(request.form.get('age_years', 0))
        }

        # Convert data to the format required by the model (list/array)
        features = np.array([[
            data['gender'], data['height'], data['weight'], data['ap_hi'], data['ap_lo'],
            data['cholesterol'], data['gluc'], data['smoke'], data['alco'],
            data['active'], data['age_years']
        ]])

        # Scale the input features
        scaled_features = scaler.transform(features)

        # Predict using the ML model
        prediction = model.predict(scaled_features)

        # Return the prediction as JSON
        result = {'prediction': int(prediction[0])}  # Ensure JSON-serializable output
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
