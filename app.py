from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib  # Import joblib for loading the model

app = Flask(__name__)
 

# Correct path to your model file
model_path = r'C:\Users\Asus\PROJECT CSE\Breast Cancer Detection\breast_cancer_detector.pickle'

# Load the model
breast_cancer_detector = joblib.load(model_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve form data
    input_features = [float(request.form[field]) for field in [
        'Mean Radius', 'Mean Texture', 'Mean Perimeter', 'Mean Area', 'Mean Smoothness',
        'Mean Compactness', 'Mean Concavity', 'Mean Concave Points', 'Mean Symmetry',
        'Mean Fractal Dimension', 'radius_error', 'texture_error', 'perimeter_error',
        'area_error', 'smoothness_error', 'compactness_error', 'concavity_error',
        'concave_points_error', 'symmetry_error', 'fractal_dimension_error',
        'worst_radius', 'worst_texture', 'worst_perimeter', 'worst_area', 'worst_smoothness',
        'worst_compactness', 'worst_concavity', 'worst_concave_points', 'worst_symmetry',
        'worst_fractal_dimension'
    ]]
    features_value = [np.array(input_features)]

    # Define feature names
    features_name = [
        'mean radius', 'mean texture', 'mean perimeter', 'mean area',
        'mean smoothness', 'mean compactness', 'mean concavity',
        'mean concave points', 'mean symmetry', 'mean fractal dimension',
        'radius error', 'texture error', 'perimeter error', 'area error',
        'smoothness error', 'compactness error', 'concavity error',
        'concave points error', 'symmetry error', 'fractal dimension error',
        'worst radius', 'worst texture', 'worst perimeter', 'worst area',
        'worst smoothness', 'worst compactness', 'worst concavity',
        'worst concave points', 'worst symmetry', 'worst fractal dimension'
    ]

    # Convert input features into a DataFrame
    df = pd.DataFrame(features_value, columns=features_name)
    
    # Make prediction
    output = breast_cancer_detector.predict(df)

    # Render appropriate HTML page based on the prediction
    if output[0] == 1:
        return render_template('yes.html')
    else:
        return render_template('no.html')

if __name__ == '__main__':
    app.run(debug=True)
