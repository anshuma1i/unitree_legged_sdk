import os
import pandas as pd
import cma

# --- CONFIGURATION ---
CHECKPOINT_FILE_CMA = "optimization_checkpoint_cma.pkl"
OUTPUT_CSV_FILE_CMA = "optimization_results_cmaes.csv"
FAILURE_PENALTY = 1000.0 # Must match the value in your optimizer script
param_names = ['tau_boost', 'z_offset', 'amp_push', 'push_ratio', 'swing_time']

def read_cmaes_results():
    """
    Loads the CMA-ES checkpoint file, processes the data,
    and prints it to the console and saves it as a CSV.
    """
    if not os.path.exists(CHECKPOINT_FILE_CMA):
        print(f"Error: Checkpoint file '{CHECKPOINT_FILE_CMA}' not found.")
        return

    print(f"✓ Loading results from '{CHECKPOINT_FILE_CMA}'...")
    es = cma.load(CHECKPOINT_FILE_CMA)

    # CMA-ES stores its history in a logger. We can access it directly.
    # The data is stored in columns, so we need to transpose it.
    all_solutions = es.logger.x.T
    all_scores = es.logger.f.T[0] # The logger stores scores in a nested list

    # Process the scores to convert them back into distances
    distances = []
    for score in all_scores:
        if score == FAILURE_PENALTY:
            distances.append("Failed")
        else:
            distances.append(-score)

    # Use pandas to create a clean DataFrame
    data = {
        'Trial #': [i+1 for i in range(len(all_scores))],
        'Distance (cm)': distances,
        'Score': all_scores,
    }
    for i, name in enumerate(param_names):
        data[name] = all_solutions[i]

    df = pd.DataFrame(data)

    # Reorder columns
    column_order = ['Trial #'] + param_names + ['Distance (cm)', 'Score']
    df = df[column_order]

    print("\n" + "="*80)
    print("CMA-ES Optimization Run History:")
    print("="*80)
    print(df.to_string())

    # Save the DataFrame to a CSV file
    df.to_csv(OUTPUT_CSV_FILE_CMA, index=False)
    print("\n" + "="*80)
    print(f"✓ Results have been successfully saved to '{OUTPUT_CSV_FILE_CMA}'")

if __name__ == "__main__":
    read_cmaes_results()