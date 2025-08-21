import subprocess
import os
import cma
from cma.evolution_strategy import CMAEvolutionStrategy
import numpy as np
import sys # Import the sys library to allow for a clean exit

# --- CONFIGURATION ---
CHECKPOINT_FILE_CMA = "optimization_checkpoint_cma.pkl"
TOTAL_TRIALS = 50

# A constant to represent a very bad score for failed runs.
FAILURE_PENALTY = 1000.0

# 1. --- DEFINE THE SEARCH SPACE & STARTING POINT ---
x0 = [10.0, 0.25, 0.55, 0.75, 0.75] # [tau_boost, z_offset, amp_push, push_ratio, swing_time]
sigma0 = 0.5 
param_names = ['tau_boost', 'z_offset', 'amp_push', 'push_ratio', 'swing_time']

# --- THE OBJECTIVE FUNCTION ---
def objective(params):
    """
    Runs a single trial on the robot and returns the score.
    Accepts a list of parameter values.
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
        # --- THIS IS THE CORRECTED SECTION ---
        success_input = input("Did the run complete successfully (y/n)?: (or type 'q' to quit): ").lower()
        if success_input in ['y', 'n']:
            break
        # Add a new condition to check for the quit command
        elif success_input in ['q', 'quit']:
            print("Quit command received. Exiting optimization run.")
            # sys.exit() is a clean way to terminate the entire script
            sys.exit(0)
        print("Invalid input. Please enter 'y', 'n', or 'q'.")

    if success_input == 'y':
        while True:
            try:
                distance_str = input("Please enter the distance traveled in cm : ")
                distance = float(distance_str)
                break
            except ValueError:
                print("Invalid input. Please enter a number.")
        score = -distance
        print(f"SUCCESS -> Distance: {distance:.2f}cm, Score: {score:.2f}")
    else:
        score = FAILURE_PENALTY
        print(f"FAILURE -> Applying penalty score: {score:.2f}")
    
    return score

# --- MAIN SCRIPT WITH CHECKPOINTING ---
if __name__ == "__main__":
    if os.path.exists(CHECKPOINT_FILE_CMA):
        print(f"✓ Found existing checkpoint. Resuming optimization from '{CHECKPOINT_FILE_CMA}'...")
        es = cma.load(CHECKPOINT_FILE_CMA)
    else:
        print("i No checkpoint found. Starting a new optimization run...")
        bounds = [[5.0, 0.1, 0.4, 0.5, 0.5], [15.0, 0.4, 0.8, 0.9, 2.0]]
        es = CMAEvolutionStrategy(x0, sigma0, {'bounds': bounds, 'seed':123})

    print(f"Optimizer state: {es.countevals} evaluations completed so far.")
    
    # The main optimization loop
    while es.countevals < TOTAL_TRIALS:
        solutions = es.ask()
        
        scores = []
        for i, sol in enumerate(solutions):
            if es.countevals + len(scores) >= TOTAL_TRIALS:
                print("\nTrial budget reached. Concluding this generation early.")
                break

            print(f"\n--- Starting Trial {es.countevals + len(scores) + 1} of {TOTAL_TRIALS} (Generation {es.countiter}, Candidate {i+1}/{len(solutions)}) ---")
            
            score = objective(sol)
            scores.append(score)
            
        solutions_ran = solutions[:len(scores)]
        if not solutions_ran:
            break

        es.tell(solutions_ran, scores)
        
        cma.pickle_dump(es, CHECKPOINT_FILE_CMA)
        print(f"\n✓ Generation {es.countiter} complete. Progress saved to '{CHECKPOINT_FILE_CMA}'. Safe to interrupt.")
        es.disp()

    print("\n" + "="*50)
    print("Optimization finished!")
    es.result_pretty()