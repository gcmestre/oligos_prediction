import joblib
import csv
from src.oligos_prediction.generate_features import OligoFeatureCalculator
from itertools import product
import pandas as pd


ALL_DINUCLEOTIDES = [''.join(pair) for pair in product('ATGC', repeat=2)]


def load_transformers(transformer_path, target_transformer_path):
    """Load the saved feature scaler and target transformer."""
    scaler = joblib.load(transformer_path)
    target_transformer = joblib.load(target_transformer_path)
    return scaler, target_transformer

def load_model(model_path):
    """Load the saved best model."""
    model = joblib.load(model_path)  # Ensure this path matches where you saved your model
    return model

def make_prediction(input_data, model, scaler, target_transformer):
    """
    Make prediction using the trained model, scaler, and target transformer.
    
    input_data: pandas DataFrame (input features to predict yield)
    model: Trained model (e.g., RandomForestRegressor, XGBoost, etc.)
    scaler: Feature scaler (RobustScaler used in training)
    target_transformer: Target transformer (PowerTransformer used in training)
    
    Returns: Predicted yield (inverse-transformed)
    """
    # Step 1: Apply the feature scaling (same scaling used during training)
    input_data_scaled = scaler.transform(input_data)
    
    # Step 2: Make the prediction
    prediction_scaled = model.predict(input_data_scaled)
    
    # Step 3: Inverse transform the predicted values to get actual yield
    prediction = target_transformer.inverse_transform(prediction_scaled.reshape(-1, 1)).ravel()
    
    return prediction



def read_sequences_from_csv(filename):
    """
    Read sequences, synthesis scale and yield from CSV file
    """
    sequences_data = []
    with open(filename, mode='r') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            sequences_data.append({
                'sequence': row['sequence'],
                'synthesis_scale': float(row['synthesis_scale']),
                'Count_of_f': row['Count_of_f'],
                'Count_of_m': row['Count_of_m'],
                # 'yield': log(float(row['yield_percent']) + 1)
                'yield': round(float(row['yield']), 2)
            })
    return sequences_data


def flatten_features(features: dict) -> dict:
    # Flatten the terminal_bases and end_stability fields
    flattened = features.copy()  # Start with a copy of the main features dictionary

    # Flatten dinucleotide_count
    # Flatten dinucleotide_count, ensuring all dinucleotides are present
    for dinucleotide in ALL_DINUCLEOTIDES:
        flattened[f"{dinucleotide}_count"] = flattened["dinucleotides_count"].get(dinucleotide, 0)
    del flattened["dinucleotides_count"]  # Remove the original dictionary field

    # Flatten terminal_bases
    flattened["5_prime_terminal"] = flattened["terminal_bases"].get("5_prime", "")
    flattened["3_prime_terminal"] = flattened["terminal_bases"].get("3_prime", "")
    del flattened["terminal_bases"]  # Remove the original dictionary field

    # Flatten base_run_lengths
    for base, length in flattened["base_run_lengths"].items():
        flattened[f"{base}_run_length"] = length
    del flattened["base_run_lengths"]  # Remove the original dictionary field

    # Flatten end_stability
    flattened["5_prime_end_stability"] = flattened["end_stability"].get("5_prime", 0.0)
    flattened["3_prime_end_stability"] = flattened["end_stability"].get("3_prime", 0.0)
    del flattened["end_stability"]  # Remove the original dictionary field


    return flattened

def generate_features_for_sequences(sequences_data):
    calculator = OligoFeatureCalculator()
    features_list = []
    

    c = 1
    for data in sequences_data:
        sequence = data['sequence']
        synthesis_scale = data['synthesis_scale']
        # modification_f = data['Count_of_f']
        # modification_m = data['Count_of_m']
        # yield_value = data['yield']

        features = calculator.calculate_all_features(sequence, synthesis_scale)
        
        features_dict = {
            "sequence": sequence,
            "length": features.length,
            "synthesis_scale": features.synthesis_scale,
            # "modification_f" : modification_f,
            # "modification_m": modification_m,
            "gc_content": features.gc_content,
            "melting_temp": features.melting_temp,
            "dinucleotides_count": features.dinucleotides_count,
            "max_gc_runs": features.max_gc_runs,
            "terminal_bases": features.terminal_bases,
            "purine_pyrimidine_ratio": features.purine_pyrimidine_ratio,
            "hairpin_score": features.hairpin_score,
            "difficult_coupling_steps": features.difficult_coupling_steps,
            "sequence_complexity": features.sequence_complexity,
            "repeating_motifs": features.repeating_motifs,
            "base_run_lengths": features.base_run_lengths,
            "hydrophobicity_score": features.hydrophobicity_score,
            "coupling_efficiency": features.coupling_efficiency,
            "deprotection_sensitivity": features.deprotection_sensitivity,
            "purification_difficulty": features.purification_difficulty,
            "g_quadruplex_potential": features.g_quadruplex_potential,
            "end_stability": features.end_stability,
            "molecular_weight": features.molecular_weight,
            "charge_density": features.charge_density,
            "gc_scale_product": features.gc_scale_product,
            # "yield": yield_value
        }

        # Flatten nested dictionaries
        flattened_features = flatten_features(features_dict)
        features_list.append(flattened_features)
        c += 1
    
    return features_list

def transform_data(features_list, scaler):
    """Transform the feature data using the saved scaler."""
    features_df = pd.DataFrame(features_list)
    features_scaled = scaler.transform(features_df)
    scaled_df = pd.DataFrame(features_scaled, columns=features_df.columns)
    return scaled_df