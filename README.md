# Crop Yield Prediction Flask Application

A comprehensive machine learning web application for predicting crop yields using various environmental and agricultural parameters. Built with Python, Flask, and scikit-learn, featuring a Random Forest regression model with feature selection and interactive visualizations.

## 🌾 Features

- **Machine Learning Model**: Random Forest Regressor with hyperparameter tuning
- **Feature Selection**: Automated selection of top 50 most important features from 300+ available features
- **Interactive Web Interface**: User-friendly Flask web application
- **Data Visualization**: Dynamic charts showing predictions vs. average yields and feature importance
- **Multiple Crop Support**: Supports 8 different crop types (wheat, rice, corn, barley, soybean, potato, sugarcane, cotton)
- **Environmental Factors**: Considers temperature, rainfall, humidity, soil conditions, and more

## 🚀 Demo

The application provides:
- Input form for agricultural parameters
- Real-time yield predictions
- Comparison charts with average crop yields
- Feature importance visualization
- Detailed prediction analysis

## 📊 Model Performance

- **Training R²**: 0.9890
- **Testing R²**: 0.9341
- **Training RMSE**: 3.0464
- **Testing RMSE**: 7.0703
- **Dataset Size**: 1,000 samples with 333 features

## 🛠️ Installation

### Prerequisites

- Python 3.7+
- pip package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/crop-yield-prediction.git
   cd crop-yield-prediction
   ```

2. **Create virtual environment**
   ```bash
   python -m venv crop_yield_env
   source crop_yield_env/bin/activate  # On Windows: crop_yield_env\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn flask joblib
   ```

4. **Train the model (first time only)**
   ```bash
   python Crop_yield_training_code.py
   ```

5. **Run the Flask application**
   ```bash
   python crop_yield_app.py
   ```

6. **Access the application**
   Open your browser and navigate to `http://localhost:5000`

## 📁 Project Structure

```
crop-yield-prediction/
│
├── Crop_yield_training_code.py    # Model training script
├── crop_yield_app.py              # Flask web application
├── crop_yield_dataset.csv         # Generated dataset (created automatically)
├── crop_yield_model.pkl           # Trained model (created after training)
├── scaler.pkl                     # Feature scaler (created after training)
├── selected_features.pkl          # Selected features list
├── feature_list.pkl               # Complete feature list
├── templates/                     # HTML templates for Flask
│   ├── index.html
│   ├── result.html
│   └── error.html
└── README.md
```

## 🔧 Usage

### Training the Model

1. Run the training script to create the machine learning model:
   ```bash
   python Crop_yield_training_code.py
   ```

This will:
- Generate a sample dataset (if not exists)
- Preprocess the data
- Select top 50 features using Random Forest feature importance
- Train the model with hyperparameter tuning
- Save the trained model and preprocessing components

### Running the Web Application

1. Start the Flask server:
   ```bash
   python crop_yield_app.py
   ```

2. Fill in the prediction form with:
   - Crop type (wheat, rice, corn, etc.)
   - Environmental conditions (temperature, rainfall, humidity)
   - Soil conditions (pH, type, nutrients)
   - Field characteristics (size, plant density)
   - Agricultural practices (irrigation, fertilizer)

3. View predictions with visualizations

## 🧠 Model Details

### Algorithm
- **Primary Model**: Random Forest Regressor
- **Feature Selection**: Based on feature importance ranking
- **Preprocessing**: StandardScaler for feature normalization
- **Hyperparameter Tuning**: GridSearchCV with 3-fold cross-validation

### Key Features Used
- Temperature averages
- Rainfall measurements
- Soil pH and moisture
- Crop type (one-hot encoded)
- Irrigation and fertilizer data
- Genetic and environmental indicators

### Parameters Optimized
- Number of estimators: [100, 200]
- Maximum depth: [None, 20]
- Minimum samples split: [2, 5]
- Minimum samples leaf: [1, 2]

## 📈 API Endpoints

- `GET /` - Home page with input form
- `POST /predict` - Submit prediction request and get results

## 🎯 Input Parameters

### Required Fields
- **Crop Type**: wheat, rice, corn, barley, soybean, potato, sugarcane, cotton
- **Temperature**: Average temperature (°C)
- **Rainfall**: Rainfall amount (mm)
- **Humidity**: Humidity percentage (%)
- **Soil pH**: Soil acidity level
- **Field Size**: Field size in hectares

### Optional Fields
- Sunlight hours
- Plant density
- Irrigation frequency
- Nutrient levels (N, P, K)
- Soil type
- Season
- Region

## 📊 Output

The application provides:
- **Predicted Yield**: Estimated crop yield in tons per hectare
- **Comparison Chart**: Predicted vs. average yield for the crop type
- **Feature Importance**: Visualization of key factors affecting prediction
- **Input Summary**: Overview of provided parameters

## 🔍 Troubleshooting

### Common Issues

1. **Model files not found**
   ```
   Error: Model file crop_yield_model.pkl not found
   ```
   **Solution**: Run `python Crop_yield_training_code.py` first to train the model

2. **Missing dependencies**
   ```
   ModuleNotFoundError: No module named 'sklearn'
   ```
   **Solution**: Install required packages using pip

3. **Port already in use**
   ```
   Address already in use
   ```
   **Solution**: Change the port in `crop_yield_app.py` or kill the existing process


## 📝 License

This project is licensed under the MIT License -

## 👨‍💻 Author

**Arumugam**
- GitHub: [@Arumugamvasu](https://github.com/Arumugamvasu)
- Email: arumugamece57@gmail.com

## 🙏 Acknowledgments

- scikit-learn for machine learning capabilities
- Flask for web framework
- matplotlib/seaborn for visualizations
- Agricultural research community for domain knowledge

## 📚 Future Enhancements

- [ ] Add more crop types and varieties
- [ ] Implement real-time weather data integration
- [ ] Add historical yield tracking
- [ ] Mobile-responsive design improvements
- [ ] API documentation with Swagger
- [ ] Docker containerization
- [ ] Cloud deployment (AWS/Azure/GCP)
- [ ] Database integration for user data storage
