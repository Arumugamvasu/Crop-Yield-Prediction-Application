# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 23:57:13 2025

@author: Arumugam
"""
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import base64
from io import BytesIO

app = Flask(__name__)

# Load the model, scaler, and feature information
MODEL_PATH = 'crop_yield_model.pkl'
SCALER_PATH = 'scaler.pkl'
SELECTED_FEATURES_PATH = 'selected_features.pkl'
FEATURE_LIST_PATH = 'feature_list.pkl'

def load_model_components():
    """Load all required model components"""
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    selected_features = joblib.load(SELECTED_FEATURES_PATH)
    feature_list = joblib.load(FEATURE_LIST_PATH)
    return model, scaler, selected_features, feature_list

# Sample crop types and their characteristics for the form
CROP_TYPES = [
    'wheat', 'rice', 'corn', 'barley', 'soybean', 'potato', 'sugarcane', 'cotton'
]

SOIL_TYPES = [
    'clay', 'silt', 'sand', 'loam', 'rocky', 'chalky', 'peat'
]

SEASONS = [
    'spring', 'summer', 'fall', 'winter'
]

FERTILIZER_TYPES = [
    'nitrogen', 'phosphorus', 'potassium', 'mixed', 'organic', 'none'
]

IRRIGATION_METHODS = [
    'drip', 'sprinkler', 'flood', 'subsurface', 'manual', 'none'
]

REGIONS = [
    'north', 'south', 'east', 'west', 'northeast', 'northwest', 'southeast', 'southwest'
]

def create_prediction_plots(input_data, prediction):
    """Create visualization plots for the prediction"""
    plots = {}
    
    # Create a comparison chart with predicted yield vs average for crop type
    crop_averages = {
        'wheat': 5.0, 'rice': 6.5, 'corn': 8.5, 'barley': 4.5, 
        'soybean': 3.0, 'potato': 22.5, 'sugarcane': 75.0, 'cotton': 1.2
    }
    
    # Get the crop type from input data
    crop_type = input_data.get('crop_type', 'wheat')
    
    # Comparison bar chart
    plt.figure(figsize=(8, 6))
    bars = plt.bar(['Predicted Yield', f'Average {crop_type.title()} Yield'], 
                  [prediction[0], crop_averages.get(crop_type, 5.0)],
                  color=['#2C7BB6', '#D7191C'])
    plt.axhline(y=crop_averages.get(crop_type, 5.0), color='gray', linestyle='--', alpha=0.6)
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f}',
                ha='center', va='bottom', fontweight='bold')
    
    plt.ylabel('Yield (tons per hectare)')
    plt.title(f'Predicted Yield vs Average for {crop_type.title()}')
    plt.tight_layout()
    
    # Convert plot to base64 string
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()
    
    plots['comparison'] = base64.b64encode(image_png).decode('utf-8')
    
    # Create feature importance plot (simplified)
    plt.figure(figsize=(10, 6))
    plt.bar(['Temperature', 'Rainfall', 'Soil pH', 'Irrigation', 'Fertilizer'], 
            [0.3, 0.25, 0.2, 0.15, 0.1],
            color='#2C7BB6')
    plt.title('Simplified Feature Importance')
    plt.ylabel('Importance')
    plt.tight_layout()
    
    # Convert plot to base64 string
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()
    
    plots['importance'] = base64.b64encode(image_png).decode('utf-8')
    
    return plots


@app.route('/')
def home():
    """Render the home page"""
    return render_template('index.html', 
                          crop_types=CROP_TYPES,
                          soil_types=SOIL_TYPES,
                          seasons=SEASONS,
                          fertilizer_types=FERTILIZER_TYPES,
                          irrigation_methods=IRRIGATION_METHODS,
                          regions=REGIONS)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        # Get form data
        form_data = request.form.to_dict()
        
        # Load model components
        model, scaler, selected_features, feature_list = load_model_components()
        
        # Create a sample dataframe to get feature columns
        dummy_data = {feature: 0 for feature in feature_list}
        input_data = pd.DataFrame([dummy_data])
        
        # Fill in the values from the form (simplified for demonstration)
        # Basic numeric features
        numeric_features = [
            'temperature_avg', 'rainfall_mm', 'humidity_pct', 'sunlight_hours',
            'field_size_hectares', 'plant_density', 'irrigation_frequency',
            'soil_pH', 'nitrogen_level', 'phosphorus_level', 'potassium_level'
        ]
        
        for feature in numeric_features:
            if feature in form_data:
                if feature in feature_list:
                    input_data[feature] = float(form_data[feature])
        
        # One-hot encode categorical features if present in the model's feature list
        crop_type = form_data.get('crop_type')
        if crop_type and f'crop_type_{crop_type}' in feature_list:
            input_data[f'crop_type_{crop_type}'] = 1
            
        soil_type = form_data.get('soil_type')
        if soil_type and f'soil_type_{soil_type}' in feature_list:
            input_data[f'soil_type_{soil_type}'] = 1
            
        season = form_data.get('season')
        if season and f'season_{season}' in feature_list:
            input_data[f'season_{season}'] = 1
            
        # Make prediction
        # 1. Scale the input data
        input_scaled = scaler.transform(input_data)
        
        # 2. Select only the features that were used during training
        feature_indices = [feature_list.index(feature) for feature in selected_features if feature in feature_list]
        selected_input = input_scaled[:, feature_indices]
        
        # 3. Make prediction with the model
        prediction = model.predict(selected_input)
        
        # Create plots
        plots = create_prediction_plots(form_data, prediction)
        
        # Prepare output data
        output = {
            'prediction': float(prediction[0]),
            'input_summary': {
                'crop_type': crop_type,
                'soil_type': soil_type,
                'season': season,
                'temperature': float(form_data.get('temperature_avg', 0)),
                'rainfall': float(form_data.get('rainfall_mm', 0)),
                'field_size': float(form_data.get('field_size_hectares', 0))
            },
            'plots': plots
        }
        
        return render_template('result.html', output=output)
    
    except Exception as e:
        import traceback
        error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
        return render_template('error.html', error=error_msg)


if __name__ == '__main__':
    # Ensure model files exist
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file {MODEL_PATH} not found. Please train the model first.")
        print("Run the model training script to create the necessary model files.")
        exit(1)
    
    # Run the Flask app
    app.run(debug=True)