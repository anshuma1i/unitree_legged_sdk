import subprocess
import os
import cma
from cma.evolution_strategy import CMAEvolutionStrategy
import numpy as np
import sys
import pickle
import pandas as pd

# --- CONFIGURATION ---
CHECKPOINT_FILE_CMA = "optimization_checkpoint_cma.pkl"
# This is our new human-readable log file
CSV_LOG_FILE = "optimization_log_cmaes.csv" 
TOTAL_TRIALS = 50
FAILURE_PENALTY = 1000.0

# --- DEFINE THE SEARCH SPACE & STARTING POINT ---
x0 = [10.0, 0.25, 0.55, 0.75, 0.75] # [tau_boost, z_offset, amp_push, push_ratio, swing_time]
sigma0 = 0.5
param_names = ['tau_boost', 'z_offset', 'amp_push', 'push_ratio', 'swing_time']

# --- NEW FUNCTION: To save our human-readable log ---
def save_history_to_csv(history, filename):
    """Takes our trial history and saves it to a clean CSV file."""
    if not history:
        return # Do nothing if history is empty
    
    all_solutions = [item['parameters'] for item in history]
    all_scores = [item['score'] for item in history]
    
    distances = []
    for score in all_scores:
        distances.append("Failed" if score == FAILURE_PENALTY else -score)

    data_for_df = {'Trial #': [i+1 for i in range(len(all_scores))]}
    for i, name in enumerate(param_names):
        data_for_df[name] = [sol[i] for sol in all_solutions]
    
    data_for_df['Distance (cm)'] = distances
    data_for_df['Score'] = all_scores

    df = pd.DataFrame(data_for_df)
    column_order = ['Trial #'] + param_names + ['Distance (cm)', 'Score']
    df = df[column_order]
    
    df.to_csv(filename, index=False)
    print(f"✓ Human-readable log updated: '{filename}'")

# --- THE OBJECTIVE FUNCTION (This is correct and robust) ---
def objective(params):
    """
    Runs a single physical trial on the robot and gets the score from the user.
    """
    params_dict = dict(zip(param_names, params))
    
    print("\n" + "="*50)
    print(f"Executing new trial with parameters:")
    for key, val in params_dict.items():
        print(f"  {key}: {val:.4f}")
    print("="*50)

    command = ['python3', 'push_board_hind.py']
    for key, val in params_dict.items():
        command.append(f'--{key}')
        command.append(str(val))

    try:
        print("Place the robot on the starting line and press Enter to begin the trial...")
        print("(Or, press Ctrl+C to end the trial)")
        input()
        subprocess.run(command)
    except KeyboardInterrupt:
        print("\nTrial concluded by user. Proceeding to measurement...")

    while True:
        success_input = input("Did the run complete successfully (y/n)?: (or type 'q' to quit): ").lower()
        if success_input in ['y', 'n']:
            break
        elif success_input in ['q', 'quit']:
            print("Quit command received. Exiting optimization run.")
            sys.exit(0)
        print("Invalid input. Please enter 'y', 'n', or 'q'.")

    if success_input == 'y':
        while True:
            try:
                distance_str = input("Please enter the distance traveled in cm : ")
                score = -float(distance_str)
                print(f"SUCCESS -> Distance: {-score:.2f}cm, Score: {score:.2f}")
                break
            except ValueError:
                print("Invalid input. Please enter a number.")
    else:
        score = FAILURE_PENALTY
        print(f"FAILURE -> Applying penalty score: {score:.2f}")
    
    return score

# --- MAIN SCRIPT WITH DUAL LOGGING AND ALL BUG FIXES ---
if __name__ == "__main__":
    if os.path.exists(CHECKPOINT_FILE_CMA):
        print(f"✓ Found existing checkpoint. Resuming optimization from '{CHECKPOINT_FILE_CMA}'...")
        with open(CHECKPOINT_FILE_CMA, 'rb') as f:
            saved_data = pickle.load(f)
            es = saved_data['optimizer_state']
            trial_history = saved_data['trial_history']
    else:
        print("i No checkpoint found. Starting a new optimization run...")
        bounds = [[5.0, 0.1, 0.4, 0.5, 0.5], [15.0, 0.4, 0.8, 0.9, 2.0]]
        es = CMAEvolutionStrategy(x0, sigma0, {'bounds': bounds, 'seed':123})
        trial_history = []

    print(f"Optimizer state: {len(trial_history)} evaluations completed so far.")
    
    budget_met = False
    while len(trial_history) < TOTAL_TRIALS:
        solutions = es.ask()
        scores = []
        for i, sol in enumerate(solutions):
            if len(trial_history) >= TOTAL_TRIALS:
                print("\nTrial budget reached. Concluding this generation early.")
                budget_met = True
                break
            
            print(f"\n--- Starting Trial {len(trial_history) + 1} of {TOTAL_TRIALS} (Generation {es.countiter}, Candidate {i+1}/{len(solutions)}) ---")
            score = objective(sol)
            scores.append(score)
            trial_history.append({'parameters': sol, 'score': score})
            
        solutions_ran = solutions[:len(scores)]
        if not solutions_ran:
            break

        try:
            es.tell(solutions_ran, scores)
        except ValueError as e:
            print(f"\nWARNING: Could not update optimizer with incomplete generation: {e}")

        # Save both the optimizer state (for resuming) AND our human-readable log
        with open(CHECKPOINT_FILE_CMA, 'wb') as f:
            pickle.dump({'optimizer_state': es, 'trial_history': trial_history}, f)
        save_history_to_csv(trial_history, CSV_LOG_FILE)
        
        print(f"\n✓ Generation {es.countiter} complete. Progress saved. Safe to interrupt.")
        es.disp()

        if budget_met:
            break

    print("\n" + "="*50)
    print("Optimization finished!")
    es.result_pretty()