# 🏠 King County House Price Prediction with CI/CD

[![CI](https://github.com/YOUR_USERNAME/house-price-ml/workflows/Continuous%20Integration/badge.svg)](https://github.com/Saoudyahya/house-price-ml/actions)
[![CD](https://github.com/YOUR_USERNAME/house-price-ml/workflows/Continuous%20Deployment/badge.svg)](https://github.com/Saoudyahya/house-price-ml/actions)

An end-to-end machine learning project for predicting house prices in King County, Washington with automated CI/CD pipelines using GitHub Actions.

## 📊 Dataset

**King County House Sales Dataset** from [Kaggle](https://www.kaggle.com/datasets/shree1992/housedata)

- **21,613 house sales** records
- **21 features** including:
  - Location: latitude, longitude, zipcode
  - Size: sqft_living, sqft_lot, sqft_above, sqft_basement
  - Quality: grade, condition, view
  - Structure: bedrooms, bathrooms, floors
  - Temporal: yr_built, yr_renovated
  - Neighbors: sqft_living15, sqft_lot15

## 🎯 Project Overview

This project implements a **Random Forest Regression** model to predict house prices with:
- ✅ Automated model training
- ✅ Performance evaluation and reporting
- ✅ CI/CD pipeline with GitHub Actions
- ✅ Automatic deployment to Hugging Face Spaces
- ✅ Interactive Gradio web interface

## 🏗️ Architecture

```
├── Data/
│   └── data.csv                 # King County housing dataset
├── Model/
│   └── model_pipeline.skops     # Trained model (auto-generated)
├── Results/
│   ├── metrics.txt              # Model metrics (auto-generated)
│   └── model_results.png        # Visualizations (auto-generated)
├── App/
│   ├── app.py                   # Gradio interface
│   └── requirements.txt         # App dependencies
├── .github/workflows/
│   ├── ci.yml                   # Continuous Integration
│   └── cd.yml                   # Continuous Deployment
├── train.py                     # Training script
├── Makefile                     # Automation commands
└── requirements.txt             # Project dependencies
```

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/house-price-ml.git
cd house-price-ml
```

### 2. Install Dependencies
```bash
make install
```

### 3. Download Dataset
Download the dataset from [Kaggle](https://www.kaggle.com/datasets/shree1992/housedata) and place `data.csv` in the `Data/` folder.

### 4. Train the Model
```bash
make train
```

### 5. Run the Web App Locally
```bash
cd App
python app.py
```

## 🤖 CI/CD Pipeline

### Continuous Integration (CI)
Triggered on every push to `main` branch:
1. ✅ Install dependencies
2. ✅ Format code with Black
3. ✅ Train the model
4. ✅ Generate performance report with CML
5. ✅ Commit results back to repository

### Continuous Deployment (CD)
Triggered after successful CI:
1. ✅ Deploy to Hugging Face Spaces
2. ✅ Make the app publicly accessible


Performance metrics are automatically updated in `Results/metrics.txt` after each training run.

## 🛠️ Technologies Used

- **ML Framework**: scikit-learn
- **Model Persistence**: skops
- **Web Interface**: Gradio
- **CI/CD**: GitHub Actions
- **MLOps**: CML (Continuous Machine Learning)
- **Deployment**: Hugging Face Spaces
- **Code Formatting**: Black

## 🔧 Configuration

### GitHub Secrets
Set up the following secrets in your GitHub repository:

1. **For CI:**
   - `USER_NAME`: Your GitHub username
   - `USER_EMAIL`: Your GitHub email

2. **For CD:**
   - `HF_TOKEN`: Your Hugging Face API token

### Hugging Face Deployment
Update the deployment command in `Makefile`:
```makefile
deploy:
    # Replace YOUR_USERNAME with your Hugging Face username
    python -c "from huggingface_hub import HfApi; ..."
```

## 📝 Usage

### Training
```bash
# Full pipeline
make all

# Individual steps
make install    # Install dependencies
make format     # Format code
make train      # Train model
make eval       # Generate evaluation report
```

### Running the App
```bash
cd App
python app.py
```

Then open your browser to `http://localhost:7860`

### Making Predictions

**Example 1: Modest Family Home**
- 3 bedrooms, 2 bathrooms
- 2000 sqft living area
- Built in 1995
- Predicted: ~$450,000

**Example 2: Luxury Waterfront**
- 4 bedrooms, 3 bathrooms
- 3000 sqft living area
- Waterfront property
- Grade 9, Excellent view
- Predicted: ~$900,000

## 📊 Feature Importance

The most important features for price prediction:
1. **sqft_living** - Living area size
2. **grade** - Construction quality
3. **sqft_above** - Above ground area
4. **lat/long** - Location
5. **sqft_living15** - Neighbor house sizes

## 🧪 Model Details

### Preprocessing
- **Numerical features**: Median imputation + Standard scaling
- **Categorical features**: One-hot encoding

### Algorithm
- **Model**: Random Forest Regressor
- **Estimators**: 100 trees
- **Random State**: 125 (for reproducibility)

### Train/Test Split
- **Training**: 70% (15,129 samples)
- **Testing**: 30% (6,484 samples)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License.


---

⭐ Star this repo if you find it helpful!
