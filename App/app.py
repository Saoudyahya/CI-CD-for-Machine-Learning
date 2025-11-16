import gradio as gr
import skops.io as sio
import pandas as pd

# Load the trained pipeline with proper security handling
model_path = "../Model/model_pipeline.skops"
unknown_types = sio.get_untrusted_types(file=model_path)
pipe = sio.load(model_path, trusted=unknown_types)


def predict(bedrooms, bathrooms, sqft_living, sqft_lot, floors, waterfront, view,
            condition, grade, sqft_above, sqft_basement, yr_built, yr_renovated,
            zipcode, lat, long, sqft_living15, sqft_lot15):
    """
    Make house price predictions based on input features.
    """
    # Create a DataFrame with the input features
    input_data = pd.DataFrame({
        'bedrooms': [bedrooms],
        'bathrooms': [bathrooms],
        'sqft_living': [sqft_living],
        'sqft_lot': [sqft_lot],
        'floors': [floors],
        'waterfront': [waterfront],
        'view': [view],
        'condition': [condition],
        'grade': [grade],
        'sqft_above': [sqft_above],
        'sqft_basement': [sqft_basement],
        'yr_built': [yr_built],
        'yr_renovated': [yr_renovated],
        'zipcode': [zipcode],
        'lat': [lat],
        'long': [long],
        'sqft_living15': [sqft_living15],
        'sqft_lot15': [sqft_lot15]
    })

    # Make prediction
    prediction = pipe.predict(input_data)[0]

    return f"${prediction:,.2f}"


# Configure inputs
inputs = [
    gr.Slider(1, 10, step=1, value=3, label="Bedrooms"),
    gr.Slider(0.5, 8, step=0.5, value=2, label="Bathrooms"),
    gr.Slider(500, 10000, step=100, value=2000, label="Living Area (sqft)"),
    gr.Slider(500, 50000, step=500, value=5000, label="Lot Size (sqft)"),
    gr.Slider(1, 3.5, step=0.5, value=1, label="Floors"),
    gr.Radio([0, 1], value=0, label="Waterfront (0=No, 1=Yes)"),
    gr.Slider(0, 4, step=1, value=0, label="View Quality (0-4)"),
    gr.Slider(1, 5, step=1, value=3, label="Condition (1-5)"),
    gr.Slider(1, 13, step=1, value=7, label="Grade (1-13)"),
    gr.Slider(500, 8000, step=100, value=1500, label="Above Ground Area (sqft)"),
    gr.Slider(0, 3000, step=100, value=0, label="Basement Area (sqft)"),
    gr.Slider(1900, 2015, step=1, value=1990, label="Year Built"),
    gr.Slider(0, 2015, step=1, value=0, label="Year Renovated (0=Never)"),
    gr.Slider(98001, 98199, step=1, value=98103, label="Zipcode"),
    gr.Slider(47.0, 47.8, step=0.01, value=47.5, label="Latitude"),
    gr.Slider(-122.5, -121.5, step=0.01, value=-122.2, label="Longitude"),
    gr.Slider(500, 6000, step=100, value=1800, label="Neighbor Living Area (sqft)"),
    gr.Slider(500, 20000, step=500, value=5000, label="Neighbor Lot Size (sqft)")
]

outputs = gr.Textbox(label="Predicted House Price")

# Example inputs
examples = [
    [3, 2.5, 2000, 5000, 1, 0, 0, 3, 7, 1500, 500, 1995, 0, 98103, 47.5, -122.2, 1800, 5000],
    [4, 3, 3000, 7500, 2, 1, 3, 4, 9, 2500, 500, 2005, 0, 98112, 47.6, -122.3, 2200, 6000],
    [2, 1, 1200, 3000, 1, 0, 0, 3, 6, 1200, 0, 1980, 2010, 98125, 47.7, -122.3, 1500, 4000]
]

# App details
title = "🏠 King County House Price Predictor"
description = """
Enter house features to predict the price of a home in King County, Washington.
This model uses Random Forest Regression trained on historical house sales data.
"""
article = """
### About the Model
- **Dataset**: King County House Sales (21,000+ houses)
- **Algorithm**: Random Forest Regressor
- **Features**: 18 house characteristics including location, size, and quality
- **Deployment**: Automated CI/CD with GitHub Actions

### Feature Guide
- **Grade**: 1-3 (Poor) | 4-6 (Below Average) | 7-9 (Average) | 10-13 (High Quality)
- **Condition**: 1 (Poor) to 5 (Excellent)
- **View**: 0 (No view) to 4 (Excellent view)
- **Waterfront**: 1 if property has waterfront view, 0 otherwise
"""

# Launch app
gr.Interface(
    fn=predict,
    inputs=inputs,
    outputs=outputs,
    examples=examples,
    title=title,
    description=description,
    article=article,
    theme=gr.themes.Soft(),
).launch()