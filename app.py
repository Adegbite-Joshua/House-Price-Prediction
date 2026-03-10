from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import json
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PowerTransformer, RobustScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def iqr_cap(X):
    X = X.copy()
    Q1 = np.percentile(X, 25, axis=0)
    Q3 = np.percentile(X, 75, axis=0)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return np.clip(X, lower_bound, upper_bound)


class NumericalTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, skew_threshold=0.5):
        self.skew_threshold = skew_threshold

    def fit(self, X, y=None):
        X = pd.DataFrame(X)

        self.numeric_cols_ = X.select_dtypes(include=[np.number]).columns
        skewness = X[self.numeric_cols_].skew()

        self.high_skew_cols_ = skewness[abs(skewness) > self.skew_threshold].index.tolist()
        self.low_skew_cols_ = skewness[abs(skewness) <= self.skew_threshold].index.tolist()

        transformers = []

        if self.high_skew_cols_:
            transformers.append(
                ('high_skew',
                 Pipeline([
                     ('cap', FunctionTransformer(iqr_cap)),
                     ('power', PowerTransformer(method='yeo-johnson')),
                     ('scale', RobustScaler())
                 ]),
                 self.high_skew_cols_)
            )

        if self.low_skew_cols_:
            transformers.append(
                ('low_skew',
                 Pipeline([
                     ('cap', FunctionTransformer(iqr_cap)),
                     ('scale', RobustScaler())
                 ]),
                 self.low_skew_cols_)
            )

        self.column_transformer_ = ColumnTransformer(
            transformers,
            remainder='passthrough'
        )

        self.column_transformer_.fit(X)

        return self

    def transform(self, X):
        return self.column_transformer_.transform(X)
    
    
class BasementTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["has_basement"] = (X["sqft_basement"] > 0).astype(int)
        X["total_sqft"] = X["sqft_lot"] + X["sqft_living"]
        X["is_renovated"] = (X["yr_renovated"] > 0).astype(int)
        X["bedrooms"] = X["bedrooms"].astype(int)
        X["house_age"] = 2026 - X["yr_built"]
        X["years_since_renovation"] = np.where(
            X["yr_renovated"] == 0, X["house_age"], 2026 - X["yr_renovated"]
        )
        X["is_modern"] = (X["yr_built"] > 2000).astype(int)
        X['luxury_bathrooms'] = (X['bathrooms'] >= 3).astype(int)
        X['bath_per_bedroom'] = X['bathrooms'] / X['bedrooms']
        X['extra_bathrooms'] = X['bathrooms'] - X['bedrooms']
        X['needs_renovation'] = (X['condition'] <= 2).astype(int)
        X['has_master_suite'] = (X['bathrooms'] >= X['bedrooms']).astype(int)
        X['multi_story'] = (X['floors'] > 1).astype(int)
        X['is_split_level'] = (X['floors'] % 1 != 0).astype(int)
        X['floor_category'] = pd.cut(
            X['floors'],
            bins=[0, 1, 2, 3, 4],
            labels=['Single', 'Two', 'Three', 'Four+']
        )
        X['premium_view'] = (X['view'] >= 3).astype(int)
        X['good_view'] = (X['view'] >= 2).astype(int)
        return X
    

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes


import sys

sys.modules['__main__'].BasementTransformer = BasementTransformer
sys.modules['__main__'].NumericalTransformer = NumericalTransformer

# Load the model package
try:
    model_package = joblib.load('complete_model_package.joblib')
    model = model_package['model']
    print(f"Model loaded successfully!")
    print(f"Best params: {model_package['metadata']['best_params']}")
    print(f"Best score: {model_package['metadata']['best_score']}")
except FileNotFoundError:
    print("Model file not found. Please ensure 'complete_model_package.joblib' exists.")
    model = None
    model_package = None

@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint for making predictions"""
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Extract features from request
        # Note: We're ignoring 'street' as specified
        features = {
            'bedrooms': float(data.get('bedrooms', 0)),
            'bathrooms': float(data.get('bathrooms', 0)),
            'sqft_living': float(data.get('sqft_living', 0)),
            'sqft_lot': float(data.get('sqft_lot', 0)),
            'floors': float(data.get('floors', 0)),
            'waterfront': int(data.get('waterfront', 0)),
            'view': int(data.get('view', 0)),
            'condition': int(data.get('condition', 0)),
            'sqft_above': float(data.get('sqft_above', 0)),
            'sqft_basement': float(data.get('sqft_basement', 0)),
            'yr_built': int(data.get('yr_built', 1900)),
            'yr_renovated': int(data.get('yr_renovated', 0)),
            # 'street' is ignored as requested
            'city': data.get('city', ''),
            'statezip': data.get('statezip', ''),
        }
        
        # Convert to DataFrame for prediction
        # The order of features should match what the model expects
        # Adjust this based on your model's feature order
        feature_columns = [
            'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
            'waterfront', 'view', 'condition', 'sqft_above', 'sqft_basement',
            'yr_built', 'yr_renovated', 'city', 'statezip'
        ]
        
        input_df = pd.DataFrame([features])[feature_columns]
        
        # Make prediction
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        prediction_log = model.predict(input_df)[0]
        prediction = np.expm1(prediction_log)
        
        response = {
            'success': True,
            'prediction': float(prediction),
            'formatted_price': f"${float(prediction):,.2f}",
            'input_features': features,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/model-info', methods=['GET'])
def model_info():
    """API endpoint to get model information"""
    if model_package:
        return jsonify({
            'model_type': model_package['metadata'].get('model_type', 'Unknown'),
            'best_score': model_package['metadata'].get('best_score', None),
            'best_params': model_package['metadata'].get('best_params', {}),
            'training_date': model_package['metadata'].get('training_date', None)
        })
    else:
        return jsonify({'error': 'Model information not available'}), 404

if __name__ == '__main__':
    app.run(debug=True, port=5000)