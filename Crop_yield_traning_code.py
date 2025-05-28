# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 23:55:57 2025

@author: Arumugam
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class CropYieldPredictor:
    """
    A class for building, training, and evaluating machine learning models 
    for crop yield prediction.
    """
    
    def __init__(self, data_path='crop_yield_dataset.csv'):
        """
        Initialize the predictor with dataset path.
        
        Parameters:
        -----------
        data_path : str
            Path to the crop yield dataset CSV file
        """
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.scaler = None
        self.selected_features = None
        self.feature_list = None
        
    def load_data(self):
        """Load the dataset from CSV file"""
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"Dataset loaded with shape: {self.df.shape}")
            return self
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}. Make sure the file {self.data_path} exists.")
    
    def preprocess_data(self, test_size=0.2, random_state=42):
        """
        Preprocess the data and split into training and testing sets.
        """
        print("\n=== Data Preprocessing ===")
        
        # Check for missing values
        missing_values = self.df.isnull().sum().sum()
        print(f"Missing values in dataset: {missing_values}")
        
        if missing_values > 0:
            # Fill missing values if any
            self.df = self.df.fillna(self.df.mean())
        
        # Split features and target
        X = self.df.drop('yield_tons_per_ha', axis=1)
        y = self.df['yield_tons_per_ha']
        
        # Save original feature names
        self.feature_list = X.columns.tolist()
        
        # Split into train and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Scale features
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        print(f"Data preprocessed and split into training ({self.X_train.shape[0]} samples) "
              f"and testing ({self.X_test.shape[0]} samples) sets")
        
        return self
    
    def feature_selection(self, n_features=50):
        """
        Select the most important features using a Random Forest model.
        
        Parameters:
        -----------
        n_features : int
            Number of features to select
        """
        print(f"\n=== Feature Selection (Top {n_features} features) ===")
        
        # Create a temporary Random Forest to get feature importances
        temp_rf = RandomForestRegressor(n_estimators=100, random_state=42)
        temp_rf.fit(self.X_train, self.y_train)
        
        # Get feature importances
        importances = temp_rf.feature_importances_
        
        # Create DataFrame with feature names and importance values
        feature_importance = pd.DataFrame({
            'Feature': self.feature_list,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        
        # Select top n_features
        self.selected_features = feature_importance['Feature'][:n_features].tolist()
        
        # Get indices of selected features
        selected_indices = [self.feature_list.index(feature) for feature in self.selected_features]
        
        # Filter X_train and X_test to include only selected features
        self.X_train = self.X_train[:, selected_indices]
        self.X_test = self.X_test[:, selected_indices]
        
        print(f"Selected {len(self.selected_features)} important features")
        print(f"Top 10 features: {self.selected_features[:10]}")
        
        return self
    
    def train_model(self, model_type='random_forest'):
        """
        Train a machine learning model for crop yield prediction.
        """
        print(f"\n=== Training {model_type.replace('_', ' ').title()} Model ===")
        
        if model_type == 'random_forest':
            # Grid search for Random Forest
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [None, 20],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
            base_model = RandomForestRegressor(random_state=42)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Perform grid search with simpler parameters for faster results
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=3,  # Reduced from 5 to 3 for faster execution
            scoring='neg_mean_squared_error',
            verbose=1,
            n_jobs=-1
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        self.model = grid_search.best_estimator_
        print(f"Best parameters: {grid_search.best_params_}")
        
        return self
    
    def evaluate_model(self):
        """
        Evaluate the trained model on test data.
        """
        print("\n=== Model Evaluation ===")
        
        # Predictions on train and test data
        train_pred = self.model.predict(self.X_train)
        test_pred = self.model.predict(self.X_test)
        
        # Calculate metrics
        train_mse = mean_squared_error(self.y_train, train_pred)
        test_mse = mean_squared_error(self.y_test, test_pred)
        train_rmse = np.sqrt(train_mse)
        test_rmse = np.sqrt(test_mse)
        train_mae = mean_absolute_error(self.y_train, train_pred)
        test_mae = mean_absolute_error(self.y_test, test_pred)
        train_r2 = r2_score(self.y_train, train_pred)
        test_r2 = r2_score(self.y_test, test_pred)
        
        # Print metrics
        print(f"Training RMSE: {train_rmse:.4f}")
        print(f"Testing RMSE: {test_rmse:.4f}")
        print(f"Training MAE: {train_mae:.4f}")
        print(f"Testing MAE: {test_mae:.4f}")
        print(f"Training R²: {train_r2:.4f}")
        print(f"Testing R²: {test_r2:.4f}")
        
        return self
    
    def save_model(self, model_path='crop_yield_model.pkl', scaler_path='scaler.pkl', 
                   features_path='selected_features.pkl', feature_list_path='feature_list.pkl'):
        """
        Save the trained model, scaler, and selected features for later use.
        """
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        joblib.dump(self.selected_features, features_path)
        joblib.dump(self.feature_list, feature_list_path)
        
        print(f"\nModel saved to {model_path}")
        print(f"Scaler saved to {scaler_path}")
        print(f"Selected features saved to {features_path}")
        print(f"Feature list saved to {feature_list_path}")
        
        return self
    
    def make_prediction(self, input_data):
        """
        Make predictions on new data.
        
        Parameters:
        -----------
        input_data : DataFrame
            Input data to predict crop yield
            
        Returns:
        --------
        array
            Predicted crop yields
        """
        # Ensure input has all required features
        if isinstance(input_data, dict):
            input_data = pd.DataFrame([input_data])
            
        # First get the indices of the selected features from the original feature list
        selected_indices = [self.feature_list.index(feature) for feature in self.selected_features]
        
        # Scale the input data using the scaler
        scaled_input = self.scaler.transform(input_data)
        
        # Select only the features that were used in training
        selected_input = scaled_input[:, selected_indices]
        
        # Make prediction
        predictions = self.model.predict(selected_input)
        
        return predictions


def generate_sample_data():
    """Generate a sample dataset for demonstration"""
    # Check if file already exists
    if os.path.exists('crop_yield_dataset.csv'):
        print("Dataset file already exists. Using existing file.")
        return
        
    print("Generating sample dataset...")
    # Number of samples and features
    n_samples = 1000
    n_features = 300
    
    # Create base DataFrame with random data
    np.random.seed(42)
    data = {}
    
    # Add key features
    data['temperature_avg'] = np.random.normal(25, 5, n_samples)
    data['rainfall_mm'] = np.random.gamma(2, 10, n_samples)
    data['humidity_pct'] = np.random.uniform(30, 90, n_samples)
    data['soil_pH'] = np.random.normal(6.5, 0.7, n_samples)
    
    # Add many more features to reach 300
    for i in range(1, n_features - 3):
        data[f'feature_{i}'] = np.random.normal(0, 1, n_samples)
    
    # Create target variable with some correlation to features
    base_yield = 5 + 0.1 * data['temperature_avg'] + 0.05 * data['rainfall_mm'] - 0.2 * (data['soil_pH'] - 6.5)**2
    noise = np.random.normal(0, 1, n_samples)
    data['yield_tons_per_ha'] = base_yield + noise
    
    # Save to CSV
    pd.DataFrame(data).to_csv('crop_yield_dataset.csv', index=False)
    print("Sample dataset generated and saved to crop_yield_dataset.csv")


def main():
    """Main function to run the crop yield prediction pipeline"""
    try:
        # Check if dataset exists, if not generate sample data
        if not os.path.exists('crop_yield_dataset.csv'):
            generate_sample_data()
        
        # Create predictor instance
        predictor = CropYieldPredictor()
        
        # Load data and train model
        (predictor
         .load_data()
         .preprocess_data()
         .feature_selection(n_features=50)  # Use 50 features
         .train_model(model_type='random_forest')
         .evaluate_model()
         .save_model())
        
        print("\nCrop yield prediction model training completed!")
        
        # Example of making predictions
        print("\n=== Making Predictions on New Data ===")
        
        # Load small sample for demonstration
        sample_data = pd.read_csv('crop_yield_dataset.csv').head(5)
        sample_X = sample_data.drop('yield_tons_per_ha', axis=1)
        sample_y = sample_data['yield_tons_per_ha']
        
        # Make predictions
        predictions = predictor.make_prediction(sample_X)
        
        # Compare predictions with actual values
        results = pd.DataFrame({
            'Actual': sample_y.values,
            'Predicted': predictions,
            'Difference': sample_y.values - predictions
        })
        
        print(results)
        
        return predictor
        
    except Exception as e:
        print(f"An error occurred: {e}")
        print("\nTroubleshooting steps:")
        print("1. Make sure the dataset file 'crop_yield_dataset.csv' exists")
        print("2. Ensure all required packages are installed")
        print("3. Check if you have sufficient disk space for model training")
        print("4. If model files exist but are corrupted, delete them and run again to retrain")
        
        raise


if __name__ == "__main__":
    main()