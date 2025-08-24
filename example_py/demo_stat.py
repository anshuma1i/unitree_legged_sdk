#
# ==============================================================================
# PART 0: SETUP AND DATA ENTRY
# ==============================================================================
#
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# --- Your Primary Data ---
# The verified mean performance of the three champions for each algorithm.
bo_champion_means = [431.4, 442.2, 420.2]
cmaes_champion_means = [286.8, 350.2, 395.0]
manual_baseline_mean = 217.0

#
# ==============================================================================
# PART 1: HYPOTHESIS TESTING (INDEPENDENT SAMPLES T-TEST)
# ==============================================================================
#
# Goal: Determine if the performance difference between BO and CMA-ES is statistically significant.

# Perform the independent samples t-test
t_statistic, p_value = stats.ttest_ind(bo_champion_means, cmaes_champion_means, equal_var=False) # Use Welch's t-test

print("======================================================")
print("             PART 1: HYPOTHESIS TESTING             ")
print("======================================================")
print(f"Independent Samples t-test results:")
print(f"T-statistic: {t_statistic:.4f}")
print(f"P-value: {p_value:.4f}\n")

# Interpretation of the p-value
alpha = 0.05
if p_value < alpha:
    print(f"Conclusion: Since the p-value ({p_value:.4f}) is less than our significance level of {alpha},")
    print("we reject the null hypothesis. There is a statistically significant difference")
    print("between the performance of Bayesian Optimization and CMA-ES champions.\n")
else:
    print(f"Conclusion: Since the p-value ({p_value:.4f}) is greater than our significance level of {alpha},")
    print("we fail to reject the null hypothesis. We do not have sufficient statistical evidence")
    print("to claim a significant difference between the performance of the two algorithms.\n")


#
# ==============================================================================
# PART 2: QUANTIFYING PERFORMANCE (DESCRIPTIVE STATISTICS)
# ==============================================================================
#
# Goal: Calculate mean, standard deviation, and 95% confidence intervals for the main results table.

def calculate_stats(data, name):
    """Calculates descriptive statistics for a given list of data."""
    mean = np.mean(data)
    std_dev = np.std(data, ddof=1) # ddof=1 for sample standard deviation
    n = len(data)
    
    # Calculate the 95% Confidence Interval
    # For small n (like 3), we use the t-distribution's critical value
    t_crit = stats.t.ppf(0.975, df=n-1)
    std_err = std_dev / np.sqrt(n)
    ci_lower = mean - t_crit * std_err
    ci_upper = mean + t_crit * std_err
    
    return {
        "Algorithm": name,
        "Mean": f"{mean:.1f}",
        "Std Dev": f"{std_dev:.1f}",
        "95% CI": f"[{ci_lower:.1f}, {ci_upper:.1f}]"
    }

# Calculate stats for both algorithms
bo_stats = calculate_stats(bo_champion_means, "Bayesian Opt.")
cmaes_stats = calculate_stats(cmaes_champion_means, "CMA-ES")

# Create a Pandas DataFrame for a clean table presentation
results_table = pd.DataFrame([bo_stats, cmaes_stats])
print("======================================================")
print("           PART 2: DESCRIPTIVE STATISTICS TABLE       ")
print("======================================================")
print(results_table.to_string(index=False))
print("\n")


#
# ==============================================================================
# PART 3: DATA VISUALIZATION
# ==============================================================================
#
# Goal: Create three distinct plots to visually communicate the research findings.

# Set a professional plot style
sns.set_theme(style="whitegrid", context="talk")

# --- PLOT 1: Overall Performance Comparison (Bar Chart) ---
plt.figure(figsize=(10, 7))
bar_data = {
    'Algorithm': ['Bayesian Opt.', 'CMA-ES', 'Manual Baseline'],
    'Mean Distance (cm)': [np.mean(bo_champion_means), np.mean(cmaes_champion_means), manual_baseline_mean]
}
# Calculate 95% CI for error bars
bo_ci = (calculate_stats(bo_champion_means, "BO")['95% CI'])
cmaes_ci = (calculate_stats(cmaes_champion_means, "CMA-ES")['95% CI'])
bo_error = (float(bo_ci.split(', ')[1][:-1]) - float(bo_ci.split(', ')[0][1:])) / 2
cmaes_error = (float(cmaes_ci.split(', ')[1][:-1]) - float(cmaes_ci.split(', ')[0][1:])) / 2

ax1 = sns.barplot(
    x='Algorithm',
    y='Mean Distance (cm)',
    data=bar_data,
    palette=['#4c72b0', '#55a868', '#c44e52'],
    capsize=0.1
)
ax1.errorbar(x=[0, 1], y=bar_data['Mean Distance (cm)'][:2], yerr=[bo_error, cmaes_error], fmt='none', c='black', capsize=5)
ax1.set_title('Overall Performance Comparison of Optimization Algorithms', fontsize=18, weight='bold')
ax1.set_xlabel('Method', fontsize=14)
ax1.set_ylabel('Mean Distance Traveled (cm)', fontsize=14)
plt.tight_layout()
plt.show()


# --- PLOT 2: Champion Consistency (Box Plot) ---
# NOTE: You need to replace this placeholder data with your actual 5 validation runs for each of the 6 champions.
# This data is structured to show the concept.
validation_data = {
    'Distance (cm)': [
        430, 435, 428, 433, 429,  # BO Champion 1
        440, 445, 441, 443, 442,  # BO Champion 2
        418, 422, 420, 421, 419,  # BO Champion 3
        280, 290, 285, 288, 289,  # CMA-ES Champion 1
        345, 355, 350, 352, 348,  # CMA-ES Champion 2
        390, 400, 395, 396, 394   # CMA-ES Champion 3
    ],
    'Algorithm': ['BO']*15 + ['CMA-ES']*15,
    'Champion': ['C1']*5 + ['C2']*5 + ['C3']*5 + ['C1']*5 + ['C2']*5 + ['C3']*5
}
df_validation = pd.DataFrame(validation_data)

plt.figure(figsize=(12, 8))
ax2 = sns.boxplot(x='Algorithm', y='Distance (cm)', hue='Champion', data=df_validation, palette="Set2")
ax2.set_title('Consistency of Champion Gaits (Validation Runs)', fontsize=18, weight='bold')
ax2.set_xlabel('Optimization Algorithm', fontsize=14)
ax2.set_ylabel('Distance Traveled (cm)', fontsize=14)
plt.legend(title='Champion Run')
plt.tight_layout()
plt.show()


# --- PLOT 3: Optimization Efficiency (Learning Curve) ---
# NOTE: You need to replace this placeholder data with your actual trial-by-trial data from your CSV files.
# This data is structured to show how your final DataFrame should look.
# It simulates 3 runs of 50 trials for each algorithm.

# Create a function to generate one run of simulated data
def generate_run(algorithm, length=50):
    if algorithm == 'BO':
        # BO tends to find good solutions faster
        base = np.linspace(200, 430, length) + np.random.normal(0, 15, length)
    else: # CMA-ES
        # CMA-ES may improve more slowly but steadily
        base = np.linspace(150, 350, length) + np.random.normal(0, 25, length)
    # Ensure scores don't decrease over time for the "best so far" plot
    return np.maximum.accumulate(base)

# Build the DataFrame
trial_data = []
for run in range(1, 4): # 3 runs
    for trial, score in enumerate(generate_run('BO')):
        trial_data.append({'Algorithm': 'BO', 'Run': run, 'Trial': trial + 1, 'Best Score': score})
    for trial, score in enumerate(generate_run('CMA-ES')):
        trial_data.append({'Algorithm': 'CMA-ES', 'Run': run, 'Trial': trial + 1, 'Best Score': score})
df_trials = pd.DataFrame(trial_data)

plt.figure(figsize=(12, 8))
ax3 = sns.lineplot(
    x='Trial',
    y='Best Score',
    hue='Algorithm',
    data=df_trials,
    errorbar='ci', # This automatically calculates and shades the 95% CI
    linewidth=2.5,
    palette=['#4c72b0', '#55a868']
)
ax3.set_title('Optimization Efficiency (Learning Curves)', fontsize=18, weight='bold')
ax3.set_xlabel('Trial Number', fontsize=14)
ax3.set_ylabel('Best Distance Found So Far (cm)', fontsize=14)
ax3.legend(title='Algorithm')
ax3.grid(True, which='both', linestyle='--')
plt.tight_layout()
plt.show()