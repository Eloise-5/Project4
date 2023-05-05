import pandas as pd
import joblib
from flask import Flask, request, jsonify

# Load the trained model
model = joblib.load('model.pkl')

# Create the Flask app
app = Flask(__name__)

# Define the endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data
    data = request.json
    df = pd.DataFrame(data, index=[0])
    
    # Make the prediction
    prediction = model.predict(df)[0]
    
    # Return the prediction as a JSON object
    return jsonify({'prediction': prediction})

# Start the Flask app
if __name__ == '__main__':
    app.run(debug=True)

