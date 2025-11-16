import gradio as gr
import skops.io as sio

# Load the trained pipeline
pipe = sio.load("./Model/model_pipeline.skops", trusted=True)


def predict(age, sex, blood_pressure, cholesterol, na_to_k_ratio):
    """
    Make predictions based on input features.

    MODIFY THIS FUNCTION based on your dataset features!
    """
    features = [age, sex, blood_pressure, cholesterol, na_to_k_ratio]
    prediction = pipe.predict([features])[0]

    label = f"Predicted: {prediction}"
    return label


# CONFIGURE YOUR INPUTS based on your dataset
inputs = [
    gr.Slider(15, 74, step=1, label="Age"),
    gr.Radio(["M", "F"], label="Sex"),
    gr.Radio(["HIGH", "LOW", "NORMAL"], label="Blood Pressure"),
    gr.Radio(["HIGH", "NORMAL"], label="Cholesterol"),
    gr.Slider(6.2, 38.2, step=0.1, label="Na_to_K"),
]

outputs = [gr.Label(num_top_classes=5)]

# CONFIGURE EXAMPLE INPUTS
examples = [
    [30, "M", "HIGH", "NORMAL", 15.4],
    [35, "F", "LOW", "NORMAL", 8],
    [50, "M", "HIGH", "HIGH", 34],
]

# CUSTOMIZE YOUR APP DETAILS
title = "ML Model Prediction"
description = "Enter the details to get predictions from your trained model"
article = "This app uses CI/CD with GitHub Actions for automated training and deployment."

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