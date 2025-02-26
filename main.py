from oligos_prediction.predictions import run_predictions

if __name__ == "__main__":
    # Example input sequence and synthesis scale
    # sequence = "ATGCATGCATGC"  # Replace with your input sequence
    sequence = "ACUUTATUCCAAAGGGCAGCUGA"  # Replace with your input sequence
    # sequence = "CGAAGCGCCCTACTCCACTCCUGGACAUUCAGAACAAGAA"  # Replace with your input sequence
    synthesis_scale = 20  # Replace with the synthesis scale (if applicable)

    sequence_data = [{'sequence': sequence,
                    'synthesis_scale': synthesis_scale}]
    
    
    # Run the main function
    predicted_yield, lower_bound, upper_bound = run_predictions(sequence_data)
    # print(f"predicted Yield: {predicted_yield}, 95% confidence interval Lower Bound: {lower_bound}, Upper Bound: {upper_bound}")