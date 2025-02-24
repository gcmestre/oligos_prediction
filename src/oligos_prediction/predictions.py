from .utils import generate_features_for_sequences, load_model, load_transformers, make_prediction, calculate_prediction_intervals
import pandas as pd
import numpy as np
import os

file_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(file_dir, "artifacts/saved_models")
MODEL_INFO_PATH = os.path.join(MODEL_PATH, "model_info.json")
TRANSFORMER_PATH = os.path.join(MODEL_PATH, "scaler.pkl")
TARGET_TRANSFORMER_PATH = os.path.join(MODEL_PATH, "target_transformer.pkl")
MODEL = os.path.join(MODEL_PATH, "XGBoost.pkl")
RESIDUAL = os.path.join(MODEL_PATH, "residuals.csv")


def run_predictions(sequence_data):
    # Generate features for the input sequence and synthesis scale
    residuals_df = pd.read_csv(RESIDUAL)
    residuals = residuals_df['XGBoost']
    features_list = generate_features_for_sequences(sequence_data)
    data = pd.DataFrame(features_list)
    data = data.select_dtypes(include=[np.number])
    

    # Load the scaler
    scaler, target_transformer = load_transformers(transformer_path= TRANSFORMER_PATH, target_transformer_path= TARGET_TRANSFORMER_PATH)


     # Load the model
    model = load_model(model_path= MODEL)


    # Make the prediction
    prediction = make_prediction(data, model, scaler, target_transformer)

    lower_bound, upper_bound = calculate_prediction_intervals(prediction= prediction[0], residuals= residuals)


    # Print the prediction result
    return prediction[0], lower_bound, upper_bound
