import os
import pandas as pd
from skopt import load

# --- CONFIGURATION ---
CHECKPOINT_FILE = "optimization_checkpoint.pkl"
OUTPUT_CSV_FILE = "optimization_results.csv"
FAILURE_PENALTY = 1000.0 # Must match the value in your optimizer script

def read_and_display_results():
    """
    Loads the optimization checkpoint file, processes the data,
    and prints it to the console and saves it as a CSV.
    """
    if not os.path.exists(CHECKPOINT_FILE):
        print(f"Error: Checkpoint file '{CHECKPOINT_FILE}' not found.")
        print("Please run the optimization script first to generate a checkpoint.")
        return

    print(f"✓ Loading results from '{CHECKPOINT_FILE}'...")
    result = load(CHECKPOINT_FILE)

    # Extract the data from the result object
    param_names = [dim.name for dim in result.space]
    params_tried = result.x_iters
    
    # --- THIS IS THE CORRECTED LINE ---
    scores_received = result.func_vals 

    # Process the scores to convert them back into distances
    distances = []
    for score in scores_received:
        if score == FAILURE_PENALTY:
            distances.append("Failed")
        else:
            # Convert the negative score back to a positive distance
            distances.append(-score)

    # Use pandas to create a clean, organized table (DataFrame)
    data = {
        'Trial #': [i+1 for i in range(len(scores_received))],
        'Distance (cm)': distances,
        'Score': scores_received,
    }
    # Add each parameter as its own column
    for i, name in enumerate(param_names):
        data[name] = [params[i] for params in params_tried]

    df = pd.DataFrame(data)

    # Reorder columns to be more logical
    column_order = ['Trial #'] + param_names + ['Distance (cm)', 'Score']
    df = df[column_order]

    print("\n" + "="*80)
    print("Optimization Run History:")
    print("="*80)
    # Print the full table to the console
    print(df.to_string())

    # Save the DataFrame to a CSV file, which can be opened in Excel
    df.to_csv(OUTPUT_CSV_FILE, index=False)
    print("\n" + "="*80)
    print(f"✓ Results have been successfully saved to '{OUTPUT_CSV_FILE}'")
    print("You can now open this file in Excel or any spreadsheet program.")

if __name__ == "__main__":
    # You might need to install pandas: pip3 install pandas
    read_and_display_results()