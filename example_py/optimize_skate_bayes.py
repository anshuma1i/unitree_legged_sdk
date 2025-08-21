import subprocess
import os
from skopt import Optimizer, dump, load
from skopt.space import Real

# --- CONFIGURATION ---
CHECKPOINT_FILE = "optimization_checkpoint.pkl"
TOTAL_CALLS = 50

# A constant to represent a very bad score for failed runs.
FAILURE_PENALTY = 1000.0

# 1. --- DEFINE THE SEARCH SPACE ---
search_space = [
    Real(5.0, 15.0, name='tau_boost'),
    Real(0.1, 0.4, name='z_offset'),
    Real(0.4, 0.8, name='amp_push'),
    Real(0.5, 0.9, name='push_ratio'),
    Real(0.5, 2.0, name='swing_time')
]
# Create a list of the parameter names for convenience
param_names = [dim.name for dim in search_space]

# --- THE OBJECTIVE FUNCTION (Slightly modified to accept a list) ---
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

    command = ['python3', 'push_board_hind.py'] # Using python3 explicitly
    for key, val in params_dict.items():
        command.append(f'--{key}')
        command.append(str(val))

    print("Place the robot on the starting line and press Enter to begin the trial...")
    input()
    
    # --- THIS IS THE CORRECTED SECTION ---
    try:
        subprocess.run(command)
    except KeyboardInterrupt:
        # This block will now catch the Ctrl+C in THIS script,
        # preventing it from crashing the whole optimization.
        print("\nTrial concluded by user. Proceeding to measurement...")

    # --- The rest of the script continues as normal ---
    while True:
        success_input = input("Did the run complete successfully (y/n)?: ").lower()
        if success_input in ['y', 'n']:
            break
        print("Invalid input. Please enter 'y' or 'n'.")

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
    if os.path.exists(CHECKPOINT_FILE):
        print(f"✓ Found existing checkpoint. Resuming optimization from '{CHECKPOINT_FILE}'...")
        opt = load(CHECKPOINT_FILE)
    else:
        print("i No checkpoint found. Starting a new optimization run...")
        opt = Optimizer(
            dimensions=search_space,
            random_state=123,
            n_initial_points=5
        )

    # The main optimization loop
    for i in range(TOTAL_CALLS):
        completed_trials = len(opt.yi)
        if completed_trials >= TOTAL_CALLS:
            print("All trials completed.")
            break
        
        print(f"\n--- Starting Trial {completed_trials + 1} of {TOTAL_CALLS} ---")
        
        suggested_params = opt.ask()
        score = objective(suggested_params)
        result = opt.tell(suggested_params, score)
        
        dump(result, CHECKPOINT_FILE)
        print(f"✓ Progress saved to '{CHECKPOINT_FILE}'. Safe to interrupt.")

    print("\n" + "="*50)
    print("Optimization finished!")
    best_score = result.fun
    best_params_list = result.x
    
    print(f"Best score achieved (negative distance): {best_score:.4f}")
    print("Best parameters found:")
    best_params = dict(zip(param_names, best_params_list))
    for key, val in best_params.items():
        print(f"  {key}: {val:.4f}")
    print("="*50)