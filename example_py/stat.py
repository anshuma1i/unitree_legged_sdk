# ==============================================================================
# ANALYSIS SCRIPT 
# ==============================================================================

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats
from itertools import combinations
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

# Optional (mixed-effects)
try:
    import statsmodels.formula.api as smf
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

# -------------------------------------
# Styling
# -------------------------------------
sns.set_theme(style="whitegrid", context="talk")
PALETTE = {'BO': '#4c72b0', 'CMA-ES': '#55a868', 'Baseline': '#c44e52'}
ALGO_COLOR_MAP = {'Bayesian': PALETTE['BO'], 'CMA-ES': PALETTE['CMA-ES']}

# -------------------------------------
# Load data (exact filenames as provided)
# -------------------------------------
print("=" * 80)
print("PART 1: LOADING AND PREPARING DATA")
print("=" * 80)

try:
    df_validation_raw = pd.read_csv("best_fitness_Sheet1.csv")
    bo_run1 = pd.read_csv("optimization_results_BO_1.csv")
    bo_run2 = pd.read_csv("optimization_results_BO_2.csv")
    bo_run3 = pd.read_csv("optimization_results_BO_3.csv")
    cmaes_run1 = pd.read_csv("optimization_log_cmaes_1.csv")
    cmaes_run2 = pd.read_csv("optimization_log_cmaes_2.csv")
    cmaes_run3 = pd.read_csv("optimization_log_cmaes_3.csv")
    df_params = pd.read_csv("champion_parameters_ranked.csv")
    print("All data files loaded successfully.")
except FileNotFoundError as e:
    print(f"ERROR: Could not find a required data file: {e.filename}")
    raise SystemExit

# -------------------------------------
# Validation dataframe (3 champions × 5 runs each per algorithm)
# -------------------------------------
bo_vals = df_validation_raw['BO Re-run'].dropna().values.astype(float)
cma_vals = df_validation_raw['CMAES Re-run'].dropna().values.astype(float)

validation_rows = []
for i in range(3):
    for j in range(5):
        validation_rows.append({"Algorithm": "BO", "Champion": f"C{i+1}", "Distance (cm)": bo_vals[i*5 + j]})
        validation_rows.append({"Algorithm": "CMA-ES", "Champion": f"C{i+1}", "Distance (cm)": cma_vals[i*5 + j]})
df_validation = pd.DataFrame(validation_rows)

chk = df_validation.groupby(['Algorithm', 'Champion'])['Distance (cm)'].agg(['count', 'mean', 'std'])
assert (chk['count'] == 5).all(), "Each champion must have exactly 5 validation runs."
print("\n--- Champion validation sanity check ---")
print(chk)

# -------------------------------------
# Robust "Distance (cm)" for trial logs
# -------------------------------------
def to_distance(df):
    s = pd.to_numeric(df['Score'], errors='coerce') if 'Score' in df.columns else pd.Series(np.nan, index=df.index)
    d = pd.to_numeric(df['Distance (cm)'], errors='coerce') if 'Distance (cm)' in df.columns else pd.Series(np.nan, index=df.index)
    dist = d.copy()
    dist = dist.where(dist.notna(), -s)      # when Distance missing, use -Score (Score is negative distance, 1000=fail)
    dist = dist.fillna(0.0)
    if 'Score' in df.columns:
        dist[s == 1000.0] = 0.0
    return dist

bo_run1_dist = to_distance(bo_run1)
bo_run2_dist = to_distance(bo_run2)
bo_run3_dist = to_distance(bo_run3)
cma1_dist = to_distance(cmaes_run1)
cma2_dist = to_distance(cmaes_run2)
cma3_dist = to_distance(cmaes_run3)

all_trials_raw = pd.concat([
    bo_run1.assign(Algorithm='BO'), bo_run2.assign(Algorithm='BO'), bo_run3.assign(Algorithm='BO'),
    cmaes_run1.assign(Algorithm='CMA-ES'), cmaes_run2.assign(Algorithm='CMA-ES'), cmaes_run3.assign(Algorithm='CMA-ES'),
], ignore_index=True)
all_trials_raw['Distance (cm)'] = to_distance(all_trials_raw)

# -------------------------------------
# Learning-curve dataset
# -------------------------------------
def process_run(df, algorithm_name, run_id, truncate=None):
    dist = to_distance(df)
    if truncate is not None:
        dist = dist.iloc[:truncate]
    best_so_far = dist.cummax().reset_index(drop=True)
    return pd.DataFrame({
        'Trial': np.arange(1, len(best_so_far) + 1),
        'Best So Far': best_so_far.values,
        'Algorithm': algorithm_name,
        'Run ID': run_id
    })

all_runs = pd.concat([
    process_run(bo_run1, 'BO', 1),
    process_run(bo_run2, 'BO', 2),
    process_run(bo_run3, 'BO', 3),
    process_run(cmaes_run1, 'CMA-ES', 1, truncate=48),
    process_run(cmaes_run2, 'CMA-ES', 2, truncate=48),
    process_run(cmaes_run3, 'CMA-ES', 3, truncate=48),
], ignore_index=True)

# -------------------------------------
# Hypothesis testing & effect sizes (champion means)
# -------------------------------------
print("\n" + "=" * 80)
print("PART 2: PRIMARY HYPOTHESIS TESTING (Unit of Analysis: Champion)")
print("=" * 80)

bo_means = df_validation.query("Algorithm=='BO'").groupby('Champion')['Distance (cm)'].mean().to_numpy()
cma_means = df_validation.query("Algorithm=='CMA-ES'").groupby('Champion')['Distance (cm)'].mean().to_numpy()

all_means = np.concatenate([bo_means, cma_means])
obs_diff = bo_means.mean() - cma_means.mean()

perms = 0; ge = 0; two_sided = 0
for idx in combinations(range(6), 3):
    perms += 1
    d = all_means[list(idx)].mean() - np.delete(all_means, list(idx)).mean()
    if d >= obs_diff:
        ge += 1
    if abs(d) >= abs(obs_diff) - 1e-12:
        two_sided += 1

p_one = ge / perms
p_two = two_sided / perms

def hedges_g(x, y):
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    pool_sd = np.sqrt(((nx-1)*np.var(x, ddof=1) + (ny-1)*np.var(y, ddof=1)) / dof)
    g = (np.mean(x) - np.mean(y)) / pool_sd
    j = 1 - (3 / (4*dof - 1))
    return g * j

def cliffs_delta(x, y):
    return (sum(xi > yj for xi in x for yj in y) - sum(xi < yj for xi in x for yj in y)) / (len(x) * len(y))

rng = np.random.default_rng(123)
def boot_g(bo, cma, B=10000):
    vals = []
    for _ in range(B):
        bx = rng.choice(bo, size=len(bo), replace=True)
        by = rng.choice(cma, size=len(cma), replace=True)
        if np.std(bx) == 0 and np.std(by) == 0:
            continue
        vals.append(hedges_g(bx, by))
    if len(vals) == 0:
        return (np.nan, np.nan)
    return np.percentile(vals, [2.5, 97.5])

g_main = hedges_g(bo_means, cma_means)
g_lo, g_hi = boot_g(bo_means, cma_means)
delta = cliffs_delta(bo_means, cma_means)

print(f"Observed Mean Difference (BO - CMA-ES): {obs_diff:.2f} cm")
print(f"Permutation p-value (one-sided) = {p_one:.4f}")
print(f"Permutation p-value (two-sided) = {p_two:.4f}")
t_stat, p_welch = stats.ttest_ind(bo_vals, cma_vals, equal_var=False, alternative='greater')
print(f"\n--- Effect Sizes (Champion level) ---")
print(f"Hedges' g: {g_main:.2f}  (95% bootstrap CI: [{g_lo:.2f}, {g_hi:.2f}])")
print(f"Cliff's Delta: {delta:.2f} (In this sample, BO champions dominate CMA-ES champions)")
print(f"\n--- Supporting Welch's t-test (15 vs 15 individual validation runs) ---")
print(f"T = {t_stat:.3f}, one-sided p = {p_welch:.4f}  [treat with caution; repeated measures]")

# -------------------------------------
# Robustness, reliability, failure rates
# -------------------------------------
print("\n" + "=" * 80)
print("PART 3: ROBUSTNESS, RELIABILITY & SUCCESS RATE")
print("=" * 80)

def count_failures(df):
    if 'Score' in df.columns:
        s = pd.to_numeric(df['Score'], errors='coerce')
        return int((s == 1000.0).sum())
    if 'Distance (cm)' in df.columns:
        d = pd.to_numeric(df['Distance (cm)'], errors='coerce')
        return int((d <= 0).sum())
    return 0

bo_runs = [bo_run1, bo_run2, bo_run3]
cma_runs = [cmaes_run1, cmaes_run2, cmaes_run3]
bo_fail_counts = [count_failures(df) for df in bo_runs]
cma_fail_counts = [count_failures(df) for df in cma_runs]
bo_total = sum(bo_fail_counts); cma_total = sum(cma_fail_counts)
bo_total_trials = sum(len(df) for df in bo_runs)
cma_total_trials = sum(len(df) for df in cma_runs)

print("\n--- Search Success Rate (Full 150-trial budgets per algorithm) ---")
print(f"BO total failures in {bo_total_trials} trials: {bo_total} ({bo_total/bo_total_trials:.1%})")
print(f"CMA-ES total failures in {cma_total_trials} trials: {cma_total} ({cma_total/cma_total_trials:.1%})")

# Reliability across champions (std of champion means)
bo_std_across = np.std(bo_means, ddof=1)
cma_std_across = np.std(cma_means, ddof=1)
print("\n--- Reliability across champions (Std of champion means) ---")
print(f"BO champion means std:  {bo_std_across:.2f} cm")
print(f"CMA-ES champion means std: {cma_std_across:.2f} cm")

# -------------------------------------
# Parameter table + boundary hits
# -------------------------------------
print("\n" + "=" * 80)
print("PART 4: MOTION & PARAMETER ANALYSIS")
print("=" * 80)

# Normalize df_params column names
unit_to_bare = {
    'tau_boost (Nm)': 'tau_boost', 'z_offset (rad)': 'z_offset', 'amp_push (rad)': 'amp_push',
    'push_ratio': 'push_ratio', 'swing_time (s)': 'swing_time',
}
df_params_bare = df_params.rename(columns=unit_to_bare)
print("\n--- Champion Parameter Table ---")
print(df_params_bare.set_index('Algorithm').to_string())

param_ranges = {
    'tau_boost': [5, 15],
    'z_offset': [0.1, 0.4],
    'amp_push': [0.4, 0.8],
    'push_ratio': [0.5, 0.9],
    'swing_time': [0.5, 2.0]
}

print("\n--- Parameters Hitting Search Space Boundaries ---")
for _, row in df_params_bare.iterrows():
    if row['Algorithm'] == 'Baseline (Manual)':
        continue
    for p, lim in param_ranges.items():
        if abs(row[p] - lim[0]) < 1e-3 or abs(row[p] - lim[1]) < 1e-3:
            algo = row['Algorithm']
            print(f"Insight: {algo} ({row['Source Run']}) hit '{p}' boundary with value {row[p]}.")

# -------------------------------------
# Mixed-effects (optional)
# -------------------------------------
print("\n" + "=" * 80)
print("PART 5: ADVANCED ANALYSIS (MIXED-EFFECTS MODEL)")
print("=" * 80)
if STATSMODELS_AVAILABLE:
    df_val_long = df_validation.copy()
    df_val_long['champ_id'] = df_val_long['Algorithm'] + "_" + df_val_long['Champion']
    md = smf.mixedlm("Q('Distance (cm)') ~ Algorithm", df_val_long, groups=df_val_long["champ_id"])
    mfit = md.fit()
    print(mfit.summary())
else:
    print("Install 'statsmodels' to run the mixed-effects model (pip install statsmodels).")

# ==============================================================================
# VISUALIZATION SUITE
# ==============================================================================
print("\n" + "=" * 80)
print("GENERATING THE FINAL PLOTS")
print("=" * 80)

# =========================
# Plot 1.1 & 1.2 — Single-panel efficiency plots
# =========================
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

def efficiency_single_plot(algo_name, color, panel_title, outfile):
    sub = all_runs.query("Algorithm == @algo_name").copy()
    grp = sub.groupby('Trial')['Best So Far']
    med = grp.median()
    lo  = grp.min()
    hi  = grp.max()

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.suptitle(panel_title, fontsize=20, fontweight='bold', y=0.98)

    # Individual runs (thin) with inline labels R1/R2/R3
    for rid, r in sub.groupby('Run ID'):
        ax.plot(
            r['Trial'], r['Best So Far'],
            drawstyle='steps-post',
            color=color, alpha=0.38, lw=1.4, zorder=1, label='_nolegend_'
        )
        # inline label near last point
        x_last = r['Trial'].iloc[-1]
        y_last = r['Best So Far'].iloc[-1]
        ax.text(
            x_last - 0.8, y_last, f'R{rid}',
            fontsize=10, ha='right', va='center',
            color=color, alpha=0.9, zorder=4
        )

    # Min–max envelope (full range across runs)
    ax.fill_between(
        med.index, lo.values, hi.values,
        step='post', color=color, alpha=0.18,
        linewidth=0, zorder=1, label='_nolegend_'
    )

    # Median curve (thick)
    ax.plot(
        med.index, med.values,
        drawstyle='steps-post',
        color=color, lw=3.0, zorder=3, label='_nolegend_'
    )

    # Baseline
    ax.axhline(217.0, color=PALETTE['Baseline'], ls='--', lw=2, zorder=0)

    # Axes cosmetics
    ax.set_xlim(1, int(med.index.max()))
    ypad = max(10.0, 0.05 * float(hi.max()))
    ax.set_ylim(bottom=0, top=float(hi.max()) + ypad)
    ax.grid(True, which='both', linestyle='--', alpha=0.35)
    ax.set_xlabel('Trial Number', fontsize=14)
    ax.set_ylabel('Best-so-far Distance (cm)', fontsize=14)

    # Subtle backdrop: lighten spines
    for spine in ['top','right','left','bottom']:
        ax.spines[spine].set_alpha(0.3)

    # Custom legend with your preferred placement
    legend_handles = [
        Line2D([0],[0], color=color, lw=3, label=f'{algo_name} median'),
        Patch(facecolor=color, alpha=0.18, label='Min–max range across runs'),
        Line2D([0],[0], color=color, lw=1.4, alpha=0.6, label='Individual runs'),
        Line2D([0],[0], color=PALETTE['Baseline'], lw=2, ls='--', label='Manual Baseline'),
    ]
    ax.legend(
        handles=legend_handles,
        title='Algorithm',
        loc='center',
        bbox_to_anchor=(0.49, 0.20),
        frameon=True
    )

    plt.tight_layout(rect=(0, 0, 1, 0.96))

    # ===== Save BEFORE showing =====
    fig.savefig(outfile, dpi=300, bbox_inches='tight')

    plt.show()


# ---- Plot 1.1 (BO) ----
efficiency_single_plot(
    'BO', PALETTE['BO'],
    'Plot 1.1 — Optimization Efficiency (BO)',
    'plot_1_1_efficiency_BO.png'
)

# ---- Plot 1.2 (CMA-ES) ----
# If you truncated CMA-ES to 48 trials when building all_runs, it will carry through here.
efficiency_single_plot(
    'CMA-ES', PALETTE['CMA-ES'],
    'Plot 1.2 — Optimization Efficiency (CMA-ES)',
    'plot_1_2_efficiency_CMAES.png'
)




# ---------- Plot 2: Corner Plot (Parameter Space Exploration) ----------
# Prepare data + quantiles
param_cols = ['tau_boost', 'z_offset', 'amp_push', 'push_ratio', 'swing_time']
plot_data_pair = all_trials_raw.dropna(subset=param_cols + ['Distance (cm)']).copy()
# quantile labels
labels = ['Very low', 'Low', 'Mid', 'High', 'Very high']
plot_data_pair['Perf Quantile'] = pd.qcut(
    plot_data_pair['Distance (cm)'], q=[0, .2, .4, .6, .8, 1.0],
    labels=labels, duplicates='drop'
)
present_labels = list(plot_data_pair['Perf Quantile'].cat.categories)
colors = sns.color_palette('plasma', n_colors=len(present_labels))
pal = dict(zip(present_labels, colors))

# PairGrid (corner)
g = sns.PairGrid(
    data=plot_data_pair,
    vars=param_cols,
    hue='Perf Quantile',
    palette=pal,
    corner=True,
    diag_sharey=False
)
g.map_lower(sns.scatterplot, s=12, alpha=0.6, linewidth=0)
g.map_diag(sns.kdeplot, fill=True, alpha=0.3)

# Axis limits, ticks, boundaries, overlays
custom_ticks = {
    'tau_boost': [5, 10, 15],
    'z_offset': [0.1, 0.25, 0.4],
    'amp_push': [0.4, 0.6, 0.8],
    'push_ratio': [0.5, 0.7, 0.9],
    'swing_time': [0.5, 1.25, 2.0]
}

# champions/baseline series
df_params_bare_idx = df_params_bare.set_index('Algorithm')
champs = df_params_bare[df_params_bare['Algorithm'].isin(['Bayesian', 'CMA-ES'])]
baseline_row = df_params_bare_idx.loc['Baseline (Manual)']

for i, yvar in enumerate(g.y_vars):
    for j, xvar in enumerate(g.x_vars):
        ax = g.axes[i, j]
        if ax is None:
            continue
        # exact limits (no margins, avoids "outside bounds" visual)
        ax.set_xlim(param_ranges[xvar])
        if yvar != xvar:
            ax.set_ylim(param_ranges[yvar])

        # custom ticks bottom row / left column
        if i == len(g.y_vars) - 1:
            ax.set_xticks(custom_ticks[xvar])
        if j == 0 and yvar != xvar:
            ax.set_yticks(custom_ticks[yvar])

        # boundary lines
        ax.axvline(param_ranges[xvar][0], color='black', linestyle=':', linewidth=1.2)
        ax.axvline(param_ranges[xvar][1], color='black', linestyle=':', linewidth=1.2)
        if yvar != xvar:
            ax.axhline(param_ranges[yvar][0], color='black', linestyle=':', linewidth=1.2)
            ax.axhline(param_ranges[yvar][1], color='black', linestyle=':', linewidth=1.2)
            # overlay champions (stars) + baseline (X)
            for _, row in champs.iterrows():
                c = ALGO_COLOR_MAP.get(row['Algorithm'], 'k')
                ax.scatter(row[xvar], row[yvar], marker='*', s=220, c=c, edgecolor='white', zorder=10)
            ax.scatter(baseline_row[xvar], baseline_row[yvar], marker='X', s=220,
                       c=PALETTE['Baseline'], edgecolor='white', zorder=11)

g.fig.suptitle('Parameter Space Exploration', y=0.98, fontsize=22, weight='bold')
g.fig.subplots_adjust(top=0.92, bottom=0.20, left=0.08, right=0.98)

# Build one clean legend (quantiles + markers)
quant_handles = [mpatches.Patch(color=pal[l], label=l) for l in present_labels]
marker_handles = [
    Line2D([0], [0], marker='*', color='w', label='BO Champion',
           markerfacecolor=PALETTE['BO'], markeredgecolor='white', markersize=14),
    Line2D([0], [0], marker='*', color='w', label='CMA-ES Champion',
           markerfacecolor=PALETTE['CMA-ES'], markeredgecolor='white', markersize=14),
    Line2D([0], [0], marker='X', color='w', label='Manual Baseline',
           markerfacecolor=PALETTE['Baseline'], markeredgecolor='white', markersize=12)
]
g.fig.legend(handles=quant_handles + marker_handles, ncol=3, frameon=True,
             loc='center left', bbox_to_anchor=(0.39, 0.86))
plt.show()



# ==============================================================================
# Plot 3 : Hardware Robustness of Champion Gaits
# ==============================================================================

# --- Create the figure with two subplots (facets) that share a y-axis ---
fig, axes = plt.subplots(1, 2, figsize=(12, 8), sharey=True)

# --- Iterate over the two algorithms to create each facet ---
for ax, algo in zip(axes, ['BO', 'CMA-ES']):
    # Select the data for the current algorithm
    sub_df = df_validation[df_validation['Algorithm'] == algo]

    # --- FIX: Calculate the correct champion order FOR THIS ALGORITHM ---
    # This is the key change. It ensures each panel has its own correct ordering.
    algo_champion_order = (
        sub_df.groupby('Champion')['Distance (cm)']
        .mean().sort_values(ascending=False).index
    )
    # --------------------------------------------------------------------

    # 1. Draw the box plots with the CORRECT order
    sns.boxplot(
        data=sub_df, x='Champion', y='Distance (cm)',
        order=algo_champion_order, # Use the algorithm-specific order
        color=PALETTE[algo],
        width=0.8,
        linewidth=1.5,
        fliersize=0,
        ax=ax
    )

    # 2. Overlay the raw data points with the CORRECT order
    sns.stripplot(
        data=sub_df, x='Champion', y='Distance (cm)',
        order=algo_champion_order, # Use the algorithm-specific order
        ax=ax,
        jitter=0.1,
        alpha=0.9,
        size=8,
        color=PALETTE[algo],
        edgecolor='black',
        linewidth=0.6
    )

    # 3. Manually plot outliers with the CORRECT order
    for i, champion_id in enumerate(algo_champion_order):
        # ... [rest of the outlier code is correct and remains the same] ...
        vals = sub_df.loc[sub_df['Champion'] == champion_id, 'Distance (cm)'].values
        if len(vals) < 5: continue
        q1, q3 = np.percentile(vals, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        for v in vals:
            if v < lower_bound or v > upper_bound:
                ax.scatter(i, v, marker='D', s=80, color='black', edgecolor='white', zorder=10)

    # 4. Add the baseline reference line
    ax.axhline(217.0, color=PALETTE['Baseline'], linestyle='--', linewidth=2)

    # 5. Set titles and labels
    ax.set_title(algo, fontsize=18)
    ax.set_xlabel('Champion ID (Ordered by Mean)', fontsize=14)
    ax.set_ylabel('Verified Distance (cm)' if algo == 'BO' else '', fontsize=16)
    ax.grid(alpha=0.5, linestyle='--')

# --- Add a global title and a unified legend ---
fig.suptitle('Hardware Robustness of Champion Gaits', fontsize=22, weight='bold', y=0.95)
legend_handles = [
    Line2D([0], [0], linestyle='--', color=PALETTE['Baseline'], lw=2, label='Manual Baseline (217 cm)'),
    Line2D([0], [0], marker='D', color='w', markerfacecolor='black', markersize=8, linestyle='None', label='Outlier (1.5×IQR)')
]
fig.legend(handles=legend_handles, ncol=2, frameon=True, loc='lower center', bbox_to_anchor=(0.5, 0))
plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.show()


# ---------- Plot 4: Strategy Fingerprint (Parallel Coordinates) ----------
# Normalize parameters
plot_para = all_trials_raw.dropna(subset=param_cols + ['Distance (cm)']).copy()
for p, lim in param_ranges.items():
    plot_para[p] = (plot_para[p] - lim[0]) / (lim[1] - lim[0])

# Top 10% threshold on performance
top10 = plot_para['Distance (cm)'].quantile(0.90)
top_df = plot_para[plot_para['Distance (cm)'] >= top10]

# Background: ALL top-10% trials (both algos) faint grey
plt.figure(figsize=(14, 8))
for _, row in top_df.iterrows():
    plt.plot(param_cols, row[param_cols].values, color='lightgrey', alpha=0.35, linewidth=1, zorder=1)

# Medians by algorithm (among top-10%)
medians = top_df.groupby('Algorithm')[param_cols].median()
# Baseline normalized
baseline_norm = [(baseline_row[p] - param_ranges[p][0]) / (param_ranges[p][1] - param_ranges[p][0]) for p in param_cols]

# Thick median lines
plt.plot(param_cols, medians.loc['BO'].values, color=PALETTE['BO'], linewidth=4, marker='o', label='BO Median (Top 10%)', zorder=5)
plt.plot(param_cols, medians.loc['CMA-ES'].values, color=PALETTE['CMA-ES'], linewidth=4, marker='o', label='CMA-ES Median (Top 10%)', zorder=6)
# Baseline dashed
plt.plot(param_cols, baseline_norm, color=PALETTE['Baseline'], linewidth=3, linestyle='--', marker='s', label='Manual Baseline', zorder=7)

plt.title('Parameter Strategies of High-Performing Gaits', fontsize=20, weight='bold')
plt.ylabel('Normalized Parameter Value (0 = Min, 1 = Max)', fontsize=16)
plt.ylim(-0.05, 1.05)
plt.grid(True, alpha=0.4)
# Legend position per your spec
plt.legend(title='Parameter Strategy', loc='upper left', bbox_to_anchor=(0.6, 1.0), frameon=True)
plt.tight_layout()
plt.show()

# ---------- Plot 5: Failure Rate (per-run mean ± SD, with clear annotations) ----------
def per_run_percentages(runs):
    return [count_failures(df)/len(df)*100 for df in runs]

bo_percents = per_run_percentages(bo_runs)
cma_percents = per_run_percentages(cma_runs)

fail_df = pd.DataFrame({
    'Algorithm': ['BO']*3 + ['CMA-ES']*3,
    'Failure Rate (%)': bo_percents + cma_percents
})

plt.figure(figsize=(8, 7))
ax = sns.barplot(
    data=fail_df, x='Algorithm', y='Failure Rate (%)',
    estimator=np.mean, ci=None, palette=PALETTE, errorbar=None
)
# Add SD error bars manually for clarity
means = fail_df.groupby('Algorithm')['Failure Rate (%)'].mean()
sds = fail_df.groupby('Algorithm')['Failure Rate (%)'].std(ddof=1)
xpos = [p.get_x() + p.get_width()/2 for p in ax.patches]
ax.errorbar(x=xpos, y=means.values, yerr=sds.values, fmt='none', ecolor='black', elinewidth=2, capsize=6, zorder=5)

# Big, clear annotations above caps
totals = {'BO': (bo_total, bo_total_trials), 'CMA-ES': (cma_total, cma_total_trials)}
for p in ax.patches:
    algo = p.get_x() + p.get_width()/2
    algoname = ax.get_xticklabels()[int(round(algo))].get_text() if False else None  # unused safe-guard
for idx, (alg, mean_val) in enumerate(means.items()):
    total_fail, total_n = totals[alg]
    ax.text(
        idx, mean_val + sds[alg] + 0.8,
        f"{mean_val:.1f}% ({total_fail}/{total_n})",
        ha='center', va='bottom', fontsize=14, weight='bold'
    )

plt.title('Failure Rate', fontsize=20, weight='bold')
plt.ylabel('Failure Rate (%)', fontsize=16)
plt.xlabel('')
max_y = max((means + sds).max() + 3, 10)
plt.ylim(0, max_y)
plt.tight_layout()
plt.show()


# ---------- Plot 6 (Optional): Failure Heatmap ----------
print("\n--- Generating Optional 2D Failure Heatmap ---")
df_heat = all_trials_raw.copy()
df_heat['is_failure'] = (pd.to_numeric(df_heat.get('Score', np.nan), errors='coerce') == 1000).astype(int)

# Use coarser 7x7 bins within exact ranges
n_bins = 7
amp_bins = np.linspace(param_ranges['amp_push'][0], param_ranges['amp_push'][1], n_bins + 1)
swing_bins = np.linspace(param_ranges['swing_time'][0], param_ranges['swing_time'][1], n_bins + 1)

df_heat['amp_bin'] = pd.cut(df_heat['amp_push'], bins=amp_bins, include_lowest=True)
df_heat['swing_bin'] = pd.cut(df_heat['swing_time'], bins=swing_bins, include_lowest=True)

grp = df_heat.groupby(['amp_bin', 'swing_bin'])
counts = grp['is_failure'].count().unstack()
rates = (grp['is_failure'].mean()*100).unstack()
# mask for insufficient data
insufficient = (counts < 3)

# Midpoint tick labels
def midpoints(bins):
    mids = (bins[:-1] + bins[1:]) / 2
    return [f"{m:.2f}" for m in mids]

plt.figure(figsize=(11, 9))
ax = sns.heatmap(
    rates.T, cmap='Reds', cbar_kws={'label': 'Failure Rate (%)'},
    vmin=0, vmax=max(1.0, np.nanmax(rates.values)),
    linewidths=0.5, linecolor='white'
)


ax.grid(False)


# Grey out n<3
for (i, j), val in np.ndenumerate(insufficient.T.values):
    if val:
        ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=True, color='lightgrey', zorder=2, alpha=0.8))

# Annotate only non-zero cells (rounded), leave zeros/NaNs blank
for (i, j), val in np.ndenumerate(rates.T.values):
    if not np.isnan(val) and val > 0:
        ax.text(j+0.5, i+0.5, f"{val:.1f}%", ha='center', va='center', fontsize=10, color='black')

ax.set_xlabel('Swing Time (s)', fontsize=16)
ax.set_ylabel('Push Amplitude (rad)', fontsize=16)
ax.set_xticklabels(midpoints(swing_bins), rotation=45, ha='right')
ax.set_yticklabels(midpoints(amp_bins), rotation=0)
plt.title('Failure Rate by amp_push vs. swing_time', fontsize=20, weight='bold')
plt.tight_layout()
plt.show()
print("\nAnalysis complete.")
