from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app, origins=["http://localhost:3000", "http://localhost:3002", "http://localhost:3001"])

# Load model and encoders
model = joblib.load("model/churn_model.pkl")
encoders = joblib.load("model/encoders.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        print("Received request:", request.json)
        data = request.json

        # Define the order of features as expected by the model
        feature_names = [
            "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
            "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
            "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
            "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
            "MonthlyCharges", "TotalCharges"
        ]

        # Prepare input data
        input_data = []
        for feature in feature_names:
            value = data[feature]
            
            # Handle categorical features (encode them)
            if feature in encoders:
                try:
                    # Convert to string and encode
                    encoded_value = encoders[feature].transform([str(value)])[0]
                    input_data.append(encoded_value)
                    print(f"Encoded {feature}: '{value}' -> {encoded_value}")
                except ValueError as e:
                    print(f"Warning: Unknown category '{value}' for {feature}. Using most frequent category.")
                    # Handle unseen categories by using the most frequent one
                    encoded_value = 0  # Default to first category
                    input_data.append(encoded_value)
            else:
                # Handle numeric features
                input_data.append(float(value))

        print("Processed input data:", input_data)
        input_array = np.array(input_data).reshape(1, -1)
        prediction = model.predict(input_array)[0]
        print("Prediction:", prediction)

        return jsonify({
            "churn": "YES" if prediction == 1 else "NO"
        })
    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
