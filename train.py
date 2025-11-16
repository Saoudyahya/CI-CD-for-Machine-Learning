import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import skops.io as sio
import os

# Create directories if they don't exist
os.makedirs("Model", exist_ok=True)
os.makedirs("Results", exist_ok=True)

# Load and shuffle data
# REPLACE THIS with your own dataset path
df = pd.read_csv("Data/your_data.csv")
df = df.sample(frac=1, random_state=42)

print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# CONFIGURE THESE based on your dataset
TARGET_COLUMN = "target"  # Replace with your target column name
CATEGORICAL_COLUMNS = [1, 2, 3]  # Indices of categorical columns
NUMERICAL_COLUMNS = [0, 4]  # Indices of numerical columns

# Create features and target
X = df.drop(TARGET_COLUMN, axis=1).values
y = df[TARGET_COLUMN].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=125
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Create preprocessing pipeline
transform = ColumnTransformer(
    [
        ("encoder", OrdinalEncoder(), CATEGORICAL_COLUMNS),
        ("num_imputer", SimpleImputer(strategy="median"), NUMERICAL_COLUMNS),
        ("num_scaler", StandardScaler(), NUMERICAL_COLUMNS),
    ]
)

# Create full pipeline
pipe = Pipeline(
    steps=[
        ("preprocessing", transform),
        ("model", RandomForestClassifier(n_estimators=100, random_state=125)),
    ]
)

# Train model
print("Training model...")
pipe.fit(X_train, y_train)
print("Model training complete!")

# Make predictions
predictions = pipe.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions, average="macro")

print(f"Accuracy: {round(accuracy, 2) * 100}%")
print(f"F1 Score: {round(f1, 2)}")

# Save metrics
with open("Results/metrics.txt", "w") as outfile:
    outfile.write(f"\nAccuracy = {round(accuracy, 2)}, F1 Score = {round(f1, 2)}.")

# Create and save confusion matrix
cm = confusion_matrix(y_test, predictions, labels=pipe.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=pipe.classes_)
disp.plot()
plt.savefig("Results/model_results.png", dpi=120)
plt.close()

# Save model
print("Saving model...")
sio.dump(pipe, "Model/model_pipeline.skops")
print("Model saved successfully!")

print("\n=== Training Complete ===")
print(f"Model saved to: Model/model_pipeline.skops")
print(f"Metrics saved to: Results/metrics.txt")
print(f"Confusion matrix saved to: Results/model_results.png")