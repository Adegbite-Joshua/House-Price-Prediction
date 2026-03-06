# 🏠 House Price Prediction Flask App

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Flask](https://img.shields.io/badge/Flask-2.3%2B-green)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

A machine learning-powered web application that predicts house prices based on property features like location, size, condition, and amenities. Built with Flask and scikit-learn, this app provides instant price estimates through an intuitive web interface.

## 📋 Table of Contents
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Model Information](#-model-information)
- [API Endpoints](#-api-endpoints)
- [Deployment](#-deployment)
- [Contributing](#-contributing)
- [License](#-license)

## ✨ Features

- **Interactive Web Interface**: Clean, modern UI with dropdowns for easy input selection
- **Real-time Predictions**: AJAX-powered API calls for instant price estimates without page reload
- **ML Model Integration**: Pre-trained regression model for accurate predictions
- **Comprehensive Inputs**: 15+ property features including:
  - 📍 Location (City, ZIP code)
  - 📏 Physical attributes (sq ft, bedrooms, bathrooms)
  - ⭐ Quality metrics (condition, view, waterfront)
  - 📅 Year built/renovated
- **Responsive Design**: Works seamlessly on desktop, tablet, and mobile devices
- **Model Transparency**: Displays model type, training date, and performance metrics

## 🚀 Tech Stack

### Backend
- **Flask** - Python web framework
- **scikit-learn** - Machine learning library
- **joblib** - Model serialization
- **pandas** - Data manipulation
- **numpy** - Numerical operations

### Frontend
- **HTML5** - Structure
- **CSS3** - Styling with modern gradients and animations
- **JavaScript** - Dynamic interactions and API calls
- **Fetch API** - Asynchronous requests

## 💻 Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Git

### Steps

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/house-price-prediction.git
cd house-price-prediction