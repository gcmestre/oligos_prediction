from oligos_prediction.artifacts.utils import generate_features_for_sequences, load_model, load_transformers, make_prediction
import pandas as pd
import numpy as np
import os

MODEL_PATH = "src/artifacts/saved_models/"
MODEL_INFO_PATH = os.path.join(MODEL_PATH, "model_info.json")
TRANSFORMER_PATH = os.path.join(MODEL_PATH, "scaler.pkl")
TARGET_TRANSFORMER_PATH = os.path.join(MODEL_PATH, "target_transformer.pkl")
MODEL = os.path.join(MODEL_PATH, "XGBoost.pkl")


def run_predictions(sequence_data):
    # Generate features for the input sequence and synthesis scale
    features_list = generate_features_for_sequences(sequence_data)
    data = pd.DataFrame(features_list)
    data = data.select_dtypes(include=[np.number])
    

    # Load the scaler
    scaler, target_transformer = load_transformers(transformer_path= TRANSFORMER_PATH, target_transformer_path= TARGET_TRANSFORMER_PATH)


     # Load the model
    model = load_model(model_path= MODEL)


    # Make the prediction
    prediction = make_prediction(data, model, scaler, target_transformer)


    # Print the prediction result
    return prediction[0]
