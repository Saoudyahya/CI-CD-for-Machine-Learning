import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import skops.io as sio
import os

# Create directories if they don't exist
os.makedirs("Model", exist_ok=True)
os.makedirs("Results", exist_ok=True)

# Load and shuffle data
# King County Housing Dataset from Kaggle
df = pd.read_csv("Data/data.csv")
df = df.sample(frac=1, random_state=42)

print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Data preprocessing
# Drop unnecessary columns
columns_to_drop = ['id', 'date']
df = df.drop(columns=columns_to_drop, errors='ignore')

# Handle missing values in yr_renovated (0 means never renovated)
if 'yr_renovated' in df.columns:
    df['yr_renovated'] = df['yr_renovated'].fillna(0)

# Remove any rows with missing target values
df = df.dropna(subset=['price'])

print(f"Dataset shape after preprocessing: {df.shape}")

# Define feature columns
TARGET_COLUMN = "price"

# Separate features into categorical and numerical
categorical_features = ['waterfront', 'view', 'condition', 'grade']
numerical_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
                     'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated',
                     'zipcode', 'lat', 'long', 'sqft_living15', 'sqft_lot15']

# Remove features that don't exist in the dataset
categorical_features = [f for f in categorical_features if f in df.columns]
numerical_features = [f for f in numerical_features if f in df.columns]

all_features = categorical_features + numerical_features

print(f"Categorical features: {categorical_features}")
print(f"Numerical features: {numerical_features}")

# Create features and target
X = df[all_features]
y = df[TARGET_COLUMN].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=125
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numerical_features),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]), categorical_features)
    ]
)

# Create full pipeline with Random Forest Regressor
pipe = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("model", RandomForestRegressor(n_estimators=100, random_state=125, n_jobs=-1)),
    ]
)

# Train model
print("Training model...")
pipe.fit(X_train, y_train)
print("Model training complete!")

# Make predictions
predictions = pipe.predict(X_test)

# Calculate regression metrics
rmse = np.sqrt(mean_squared_error(y_test, predictions))
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"\n=== Model Performance ===")
print(f"RMSE: ${rmse:,.2f}")
print(f"MAE: ${mae:,.2f}")
print(f"R² Score: {r2:.4f}")

# Save metrics
with open("Results/metrics.txt", "w") as outfile:
    outfile.write(f"Model Performance Metrics\n")
    outfile.write(f"=" * 50 + "\n")
    outfile.write(f"RMSE (Root Mean Squared Error): ${rmse:,.2f}\n")
    outfile.write(f"MAE (Mean Absolute Error): ${mae:,.2f}\n")
    outfile.write(f"R² Score: {r2:.4f}\n")
    outfile.write(f"\nInterpretation:\n")
    outfile.write(f"- On average, predictions are off by ${mae:,.2f}\n")
    outfile.write(f"- The model explains {r2*100:.2f}% of the variance in house prices\n")

# Create visualization: Predicted vs Actual
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Plot 1: Predicted vs Actual
axes[0].scatter(y_test, predictions, alpha=0.5, s=10)
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0].set_xlabel('Actual Price ($)', fontsize=12)
axes[0].set_ylabel('Predicted Price ($)', fontsize=12)
axes[0].set_title('Predicted vs Actual House Prices', fontsize=14)
axes[0].grid(True, alpha=0.3)

# Plot 2: Residuals
residuals = y_test - predictions
axes[1].scatter(predictions, residuals, alpha=0.5, s=10)
axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
axes[1].set_xlabel('Predicted Price ($)', fontsize=12)
axes[1].set_ylabel('Residuals ($)', fontsize=12)
axes[1].set_title('Residual Plot', fontsize=14)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("Results/model_results.png", dpi=120, bbox_inches='tight')
plt.close()

# Save model
print("\nSaving model...")
sio.dump(pipe, "Model/model_pipeline.skops")
print("Model saved successfully!")

print("\n=== Training Complete ===")
print(f"Model saved to: Model/model_pipeline.skops")
print(f"Metrics saved to: Results/metrics.txt")
print(f"Visualizations saved to: Results/model_results.png")