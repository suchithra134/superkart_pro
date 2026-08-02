
# Import necessary libraries
import numpy as np
import joblib  # For loading the serialized model
import pandas as pd  # For data manipulation
from flask import Flask, request, jsonify  # For creating the Flask API

# Initialize the Flask application
superkart_sales_predictor_api = Flask("SuperKart Sales Predictor")

# Load the trained machine learning model and preprocessor
model = joblib.load("superkart_random_forest_model.joblib")
preprocessor = joblib.load("preprocessor.joblib")

# Define a route for the home page (GET request)
@superkart_sales_predictor_api.get('/')
def home():
    """
    This function handles GET requests to the root URL ('/') of the API.
    It returns a simple welcome message.
    """
    return "Welcome to the SuperKart Sales Prediction API!"

# Define an endpoint for single property prediction (POST request)
@superkart_sales_predictor_api.post('/v1/predict')
def predict_sales():
    """
    This function handles POST requests to the '/v1/predict' endpoint.
    It expects a JSON payload containing product and store details and returns
    the predicted sales as a JSON response.
    """
    # Get the JSON data from the request body
    input_data_json = request.get_json()

    # Extract relevant features from the JSON data
    # Ensure features match those used in model training
    sample = {
        'Product_Weight': input_data_json['Product_Weight'],
        'Product_Sugar_Content': input_data_json['Product_Sugar_Content'],
        'Product_Allocated_Area': input_data_json['Product_Allocated_Area'],
        'Product_MRP': input_data_json['Product_MRP'],
        'Store_Size': input_data_json['Store_Size'],
        'Store_Location_City_Type': input_data_json['Store_Location_City_Type'],
        'Store_Type': input_data_json['Store_Type'],
        'Product_Id_char': input_data_json['Product_Id_char'],
        'Store_Age_Years': input_data_json['Store_Age_Years'],
        'Product_Type_Category': input_data_json['Product_Type_Category']
    }

    # Convert the extracted data into a Pandas DataFrame
    input_df = pd.DataFrame([sample])

    # Preprocess the input data using the loaded preprocessor
    input_processed = preprocessor.transform(input_df)

    # Make prediction
    prediction = model.predict(input_processed)[0]

    # Convert predicted_price to Python float and round it
    predicted_sales = round(float(prediction), 2)

    # Return the predicted sales
    return jsonify({'Predicted Product_Store_Sales_Total': predicted_sales})


# Define an endpoint for batch prediction (POST request)
@superkart_sales_predictor_api.post('/v1/predictbatch')
def predict_sales_batch():
    """
    This function handles POST requests to the '/v1/predictbatch' endpoint.
    It expects a CSV file containing product details for multiple products
    and returns the predicted sales as a dictionary in the JSON response.
    """
    # Get the uploaded CSV file from the request
    file = request.files['file']

    # Read the CSV file into a Pandas DataFrame
    input_df_batch = pd.read_csv(file)

    # Preprocess the batch input data using the loaded preprocessor
    input_processed_batch = preprocessor.transform(input_df_batch)

    # Make predictions for all products in the DataFrame
    predicted_sales_batch = model.predict(input_processed_batch).tolist()

    # Round predictions and convert to Python floats
    predicted_sales_batch = [round(float(sales), 2) for sales in predicted_sales_batch]

    # If there's an 'id' column, use it for keys, otherwise use indices
    if 'id' in input_df_batch.columns:
        product_ids = input_df_batch['id'].tolist()
        output_dict = dict(zip(product_ids, predicted_sales_batch))
    else:
        output_dict = {str(i): sales for i, sales in enumerate(predicted_sales_batch)}

    # Return the predictions dictionary as a JSON response
    return output_dict

# Run the Flask application in debug mode if this script is executed directly
if __name__ == '__main__':
    superkart_sales_predictor_api.run(debug=True)
