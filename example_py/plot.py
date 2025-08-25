# ==============================================================================
# PLOTTING SCRIPT
#   - Loads the same inputs
#   - Builds all figures, shows interactively by default (saving optional)
#   - No hypothesis testing here
# ==============================================================================

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.ticker import FuncFormatter, FixedLocator, AutoMinorLocator, AutoMinorLocator, AutoMinorLocator

# ---------- toggle saving ----------
SAVE = False  # set True to save PNGs as in your original script

sns.set_theme(style="whitegrid", context="talk")
PALETTE = {'BO': '#4c72b0', 'CMA-ES': '#55a868', 'Baseline': '#c44e52'}
ALGO_COLOR_MAP = {'Bayesian': PALETTE['BO'], 'CMA-ES': PALETTE['CMA-ES']}

# -------------------------------------
# Load data
# -------------------------------------
def _load_inputs():
    df_validation_raw = pd.read_csv("best_fitness_Sheet1.csv")
    bo_run1 = pd.read_csv("optimization_results_BO_1.csv")
    bo_run2 = pd.read_csv("optimization_results_BO_2.csv")
    bo_run3 = pd.read_csv("optimization_results_BO_3.csv")
    cmaes_run1 = pd.read_csv("optimization_log_cmaes_1.csv")
    cmaes_run2 = pd.read_csv("optimization_log_cmaes_2.csv")
    cmaes_run3 = pd.read_csv("optimization_log_cmaes_3.csv")
    df_params = pd.read_csv("champion_parameters_ranked.csv")
    return df_validation_raw, bo_run1, bo_run2, bo_run3, cmaes_run1, cmaes_run2, cmaes_run3, df_params

(df_validation_raw,
 bo_run1, bo_run2, bo_run3,
 cmaes_run1, cmaes_run2, cmaes_run3,
 df_params) = _load_inputs()

print("\n" + "=" * 80)
print("GENERATING THE FINAL PLOTS")
print("=" * 80)

# -------------------------------------
# Build validation dataframe
# -------------------------------------
def to_validation(df_validation_raw):
    bo_vals = df_validation_raw['BO Re-run'].dropna().values.astype(float)
    cma_vals = df_validation_raw['CMAES Re-run'].dropna().values.astype(float)
    rows = []
    for i in range(3):
        for j in range(5):
            rows.append({"Algorithm": "BO", "Champion": f"C{i+1}", "Distance (cm)": bo_vals[i*5 + j]})
            rows.append({"Algorithm": "CMA-ES", "Champion": f"C{i+1}", "Distance (cm)": cma_vals[i*5 + j]})
    return pd.DataFrame(rows)

df_validation = to_validation(df_validation_raw)

# -------------------------------------
# Trial utilities
# -------------------------------------
def to_distance(df):
    s = pd.to_numeric(df.get('Score'), errors='coerce') if 'Score' in df.columns else pd.Series(np.nan, index=df.index)
    d = pd.to_numeric(df.get('Distance (cm)'), errors='coerce') if 'Distance (cm)' in df.columns else pd.Series(np.nan, index=df.index)
    dist = d.copy()
    dist = dist.where(dist.notna(), -s)  # Score is negative distance; 1000=fail
    dist = dist.fillna(0.0)
    if 'Score' in df.columns:
        dist[s == 1000.0] = 0.0
    return dist

all_trials_raw = pd.concat([
    bo_run1.assign(Algorithm='BO'), bo_run2.assign(Algorithm='BO'), bo_run3.assign(Algorithm='BO'),
    cmaes_run1.assign(Algorithm='CMA-ES'), cmaes_run2.assign(Algorithm='CMA-ES'), cmaes_run3.assign(Algorithm='CMA-ES'),
], ignore_index=True)
all_trials_raw['Distance (cm)'] = to_distance(all_trials_raw)

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

# =========================
# Plot 1.1 & 1.2 — Single-panel efficiency plots
# =========================
def efficiency_single_plot(algo_name, color, panel_title, outfile, fixed_xlim=(1, 50), fixed_ylim=(100, 400)):
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
        x_last = r['Trial'].iloc[-1]
        y_last = r['Best So Far'].iloc[-1]
        ax.text(
            x_last - 0.8, y_last, f'R{rid}',
            fontsize=10, ha='right', va='center',
            color=color, alpha=1.0, zorder=5,
            path_effects=[pe.withStroke(linewidth=2.2, foreground="white")]
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

    # FIXED axes for comparison
    ax.set_xlim(fixed_xlim)
    ax.set_ylim(fixed_ylim)

    # Annotate truncation for CMA-ES
    if algo_name == 'CMA-ES':
        ax.axvline(48, color='red', ls=':', lw=2, label='Truncated at 48')
        ax.text(48.2, fixed_ylim[1]*0.9, "Truncated\nat 48", color='red', fontsize=11,
                va='top', ha='left')

    # Cosmetics
    ax.grid(True, which='both', linestyle='--', alpha=0.35)
    ax.set_xlabel('Trial Number', fontsize=14)
    ax.set_ylabel('Best-so-far Distance (cm)', fontsize=14)
    for spine in ['top','right','left','bottom']:
        ax.spines[spine].set_alpha(0.3)

    # Custom legend
    legend_handles = [
        Line2D([0],[0], color=color, lw=3, label=f'{algo_name} median'),
        Patch(facecolor=color, alpha=0.18, label='Min–max range across runs'),
        Line2D([0],[0], color=color, lw=1.4, alpha=0.6, label='Individual runs'),
        Line2D([0],[0], color=PALETTE['Baseline'], lw=2, ls='--', label='Manual Baseline'),
    ]
    if algo_name == 'CMA-ES':
        legend_handles.append(Line2D([0],[0], color='red', ls=':', lw=2, label='Truncated at 48'))

    ax.legend(
        handles=legend_handles,
        title=None,
        loc='center',
        bbox_to_anchor=(0.49, 0.20),
        frameon=True,
        fontsize=9,        # <<< smaller legend text
        title_fontsize=10, # <<< smaller legend title
        handlelength=1.2,  # <<< shorten line marker length
        handletextpad=0.6, # <<< tighten text padding
        borderpad=0.5      # <<< tighten box padding
    )


    plt.tight_layout(rect=(0, 0, 1, 0.96))
    if SAVE and outfile:
        fig.savefig(outfile, dpi=300, bbox_inches='tight')
    print(f"Ready: {panel_title}")
    plt.show()


# ---- Plot 1.1 (BO) ----
efficiency_single_plot(
    'BO', PALETTE['BO'],
    'Plot 1.1 — Optimization Efficiency (BO)',
    'plot_1_1_efficiency_BO.png'
)

# ---- Plot 1.2 (CMA-ES) ----
efficiency_single_plot(
    'CMA-ES', PALETTE['CMA-ES'],
    'Plot 1.2 — Optimization Efficiency (CMA-ES)',
    'plot_1_2_efficiency_CMAES.png'
)

#===========================================================================================
# Plot 2.1 & 2.2 — Parameter Space Exploration (clean + ruler ticks)
#   - Place after: `_load_inputs()` and building `all_trials_raw`
#===========================================================================================

# ---- style knobs (override earlier in the file if you like) ----
AXIS_LABEL_FONTSIZE = globals().get('AXIS_LABEL_FONTSIZE', 14)
TICK_FONTSIZE       = globals().get('TICK_FONTSIZE', 11)
TITLE_FONTSIZE      = globals().get('TITLE_FONTSIZE', 22)
LEGEND_FONTSIZE     = globals().get('LEGEND_FONTSIZE', 11)
GRID_HEIGHT         = globals().get('GRID_HEIGHT', 3.5)   # per-cell size in inches
GRID_ASPECT         = globals().get('GRID_ASPECT', 1.0)   # 1.0 = square cells
MINOR_SUBDIVS       = globals().get('MINOR_SUBDIVS', 4)   # minor ticks between majors

# ---- required globals for Plot 2 ----
param_cols = ['tau_boost', 'z_offset', 'amp_push', 'push_ratio', 'swing_time']
param_ranges = {
    'tau_boost': [5, 15],
    'z_offset': [0.1, 0.4],
    'amp_push': [0.4, 0.8],
    'push_ratio': [0.5, 0.9],
    'swing_time': [0.5, 2.0],
}
_unit_to_bare = {
    'tau_boost (Nm)': 'tau_boost',
    'z_offset (rad)': 'z_offset',
    'amp_push (rad)': 'amp_push',
    'push_ratio': 'push_ratio',
    'swing_time (s)': 'swing_time',
}
df_params_bare = df_params.rename(columns=_unit_to_bare).copy()
_base_mask = df_params_bare['Algorithm'].astype(str).str.contains('baseline', case=False, na=False)
assert _base_mask.any(), "Could not find a baseline row in champion_parameters_ranked.csv"
baseline_row = df_params_bare.loc[_base_mask].iloc[0]

def parameter_space_exploration_single(algo_name, color, title_text):
    """
    Context–Story–Climax corner plot with ruler-like minor ticks.
    - Context: low/mid performers (grey)
    - Story: top 25% (algo color)
    - Climax: champions (gold *) + baseline (red X)
    - Major ticks: min / mid / max only; labels only on bottom row & left col
    - Minor ticks: evenly subdivide each major interval for a 'ruler' feel
    """
    from matplotlib.ticker import FixedLocator, FuncFormatter, AutoMinorLocator

    GOLD = "#d4af37"

    # helpers for ticks/formatting
    def three_ticks(var):
        lo, hi = param_ranges[var]
        mid = (lo + hi) / 2.0
        return [lo, mid, hi]

    def nice_formatter(var):
        if var == 'tau_boost':
            return FuncFormatter(lambda v, pos: f"{int(round(v))}")
        elif var in ('amp_push', 'z_offset', 'swing_time'):
            return FuncFormatter(lambda v, pos: f"{v:.2f}")
        elif var == 'push_ratio':
            return FuncFormatter(lambda v, pos: f"{v:.1f}")
        else:
            return FuncFormatter(lambda v, pos: f"{v:g}")

    # trials for this algorithm
    sub = all_trials_raw[all_trials_raw['Algorithm'] == algo_name].dropna(
        subset=param_cols + ['Distance (cm)']
    ).copy()
    if sub.empty:
        print(f"[WARN] No trials for {algo_name}. Skipping {title_text}.")
        return

    # split by performance
    cutoff = sub['Distance (cm)'].quantile(0.75)
    low_mid = sub[sub['Distance (cm)'] <  cutoff]
    high    = sub[sub['Distance (cm)'] >= cutoff]

    # champions for THIS algorithm
    champs_algo = df_params_bare[df_params_bare['Algorithm'].isin(
        ['Bayesian' if algo_name == 'BO' else 'CMA-ES']
    )]

    # build corner grid
    g = sns.PairGrid(
        data=sub,
        vars=param_cols,
        corner=True,
        diag_sharey=False,
        height=GRID_HEIGHT,
        aspect=GRID_ASPECT
    )

    # fill lower triangle manually (hide diagonal)
    for i, yvar in enumerate(g.y_vars):
        for j, xvar in enumerate(g.x_vars):
            ax = g.axes[i, j]
            if ax is None:
                continue

            if yvar == xvar:
                ax.set_visible(False)
                continue

            # bounds
            ax.set_xlim(param_ranges[xvar])
            ax.set_ylim(param_ranges[yvar])

            # major ticks: min/mid/max; clean formatting
            xt = three_ticks(xvar)
            yt = three_ticks(yvar)
            ax.xaxis.set_major_locator(FixedLocator(xt))
            ax.yaxis.set_major_locator(FixedLocator(yt))
            ax.xaxis.set_major_formatter(nice_formatter(xvar))
            ax.yaxis.set_major_formatter(nice_formatter(yvar))

            # minor ticks: ruler-like subdivisions (no labels)
            ax.xaxis.set_minor_locator(AutoMinorLocator(MINOR_SUBDIVS))
            ax.yaxis.set_minor_locator(AutoMinorLocator(MINOR_SUBDIVS))
            ax.tick_params(axis='both', which='minor', length=4, width=0.8, color='black')

            # major tick styling + label visibility only on outer edges
            is_bottom_row = (i == len(g.y_vars) - 1)
            is_left_col   = (j == 0)
            ax.tick_params(axis='x', which='major',
                           labelsize=TICK_FONTSIZE, labelbottom=is_bottom_row,
                           length=7, width=1.2, colors='black')
            ax.tick_params(axis='y', which='major',
                           labelsize=TICK_FONTSIZE, labelleft=is_left_col,
                           length=7, width=1.2, colors='black')

            # axis labels only on outer edges
            ax.set_xlabel(xvar if is_bottom_row else "", fontsize=AXIS_LABEL_FONTSIZE)
            ax.set_ylabel(yvar if is_left_col   else "", fontsize=AXIS_LABEL_FONTSIZE)

            # dotted search bounds
            ax.axvline(param_ranges[xvar][0], color='black', linestyle=':', linewidth=1.2)
            ax.axvline(param_ranges[xvar][1], color='black', linestyle=':', linewidth=1.2)
            ax.axhline(param_ranges[yvar][0], color='black', linestyle=':', linewidth=1.2)
            ax.axhline(param_ranges[yvar][1], color='black', linestyle=':', linewidth=1.2)

            # layers
            if not low_mid.empty:
                ax.scatter(low_mid[xvar], low_mid[yvar],
                           s=12, alpha=0.25, c="grey", linewidths=0, zorder=1)
            if not high.empty:
                ax.scatter(high[xvar], high[yvar],
                           s=26, alpha=0.9, c=color, linewidths=0, zorder=2)
            for _, row in champs_algo.iterrows():
                ax.scatter(row[xvar], row[yvar],
                           marker='*', s=240, c=GOLD, edgecolor='white', linewidth=1, zorder=10)
            ax.scatter(baseline_row[xvar], baseline_row[yvar],
                       marker='X', s=240, c=PALETTE['Baseline'],
                       edgecolor='white', linewidth=1, zorder=11)

    # title & layout
    g.fig.suptitle(title_text, y=0.9, fontsize=TITLE_FONTSIZE, fontweight='bold')
    g.fig.subplots_adjust(top=0.92, bottom=0.20, left=0.08, right=0.80)

    # legend (center-left, outside)
    legend_handles = [
        Line2D([0], [0], marker='o', color='w', label='Explored trials',
               markerfacecolor='grey', markeredgecolor='none', markersize=7),
        Line2D([0], [0], marker='o', color='w', label='Top performers (top 25%)',
               markerfacecolor=color, markeredgecolor='none', markersize=7),
        Line2D([0], [0], marker='*', color='w', label='Champion parameters',
               markerfacecolor="#d4af37", markeredgecolor='white', markersize=14),
        Line2D([0], [0], marker='X', color='w', label='Manual baseline',
               markerfacecolor=PALETTE['Baseline'], markeredgecolor='white', markersize=12)
    ]
    g.fig.legend(
        handles=legend_handles,
        loc='center left', bbox_to_anchor=(0.52, 0.6),
        frameon=True, ncol=1, fontsize=LEGEND_FONTSIZE, title=None
    )

    print(f"Ready: {title_text}")
    plt.show()
    plt.close(g.fig)

# ---- Plot 2.1: BO ----
parameter_space_exploration_single(
    algo_name='BO',
    color=PALETTE['BO'],
    title_text='Plot 2.1 — Parameter Space Exploration (BO)'
)

# ---- Plot 2.2: CMA-ES ----
parameter_space_exploration_single(
    algo_name='CMA-ES',
    color=PALETTE['CMA-ES'],
    title_text='Plot 2.2 — Parameter Space Exploration (CMA-ES)'
)




# ==============================================================================
# Plot 3 : Hardware Robustness of Champion Gaits (shared y, start at 250, no baseline, no saving)
# ==============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 8), sharey=True)

for ax, algo in zip(axes, ['BO', 'CMA-ES']):
    sub_df = df_validation[df_validation['Algorithm'] == algo]

    # Algorithm-specific ordering
    algo_champion_order = (
        sub_df.groupby('Champion')['Distance (cm)']
        .mean().sort_values(ascending=False).index
    )

    # Boxplot
    sns.boxplot(
        data=sub_df, x='Champion', y='Distance (cm)',
        order=algo_champion_order,
        color=PALETTE[algo],
        width=0.8, linewidth=1.5, fliersize=0, ax=ax
    )

    # Raw points
    sns.stripplot(
        data=sub_df, x='Champion', y='Distance (cm)',
        order=algo_champion_order,
        ax=ax, jitter=0.1, alpha=0.9, size=8,
        color=PALETTE[algo], edgecolor='black', linewidth=0.6
    )

    # Mark outliers manually (1.5×IQR)
    for i, champion_id in enumerate(algo_champion_order):
        vals = sub_df.loc[sub_df['Champion'] == champion_id, 'Distance (cm)'].values
        if len(vals) < 5:
            continue
        q1, q3 = np.percentile(vals, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        for v in vals:
            if v < lower_bound or v > upper_bound:
                ax.scatter(i, v, marker='D', s=80, color='black',
                           edgecolor='white', zorder=10)

    # Removed baseline line
    # ax.axhline(217.0, ...)

    ax.set_title(algo, fontsize=18)
    ax.set_xlabel('Champion ID (Ordered by Mean)', fontsize=14)
    ax.set_ylabel('Verified Distance (cm)' if algo == 'BO' else '', fontsize=16)
    ax.grid(alpha=0.5, linestyle='--')

# ---- Force shared y-limits: start at 250, same max on both panels ----
ymin = 275
max_val = df_validation.loc[df_validation['Algorithm'].isin(['BO', 'CMA-ES']), 'Distance (cm)'].max()
ypad = max(5.0, 0.05 * max(1.0, (max_val - ymin)))
ymax = max_val + ypad
axes[0].set_ylim(ymin, ymax)  # sharey=True syncs to both

fig.suptitle('Hardware Robustness of Champion Gaits', fontsize=22, weight='bold', y=0.95)

# Legend: only keep the outlier marker (baseline removed)
legend_handles = [
    Line2D([0], [0], marker='D', color='w',
           markerfacecolor='black', markersize=8,
           linestyle='None', label='Outlier (1.5×IQR)')
]
fig.legend(handles=legend_handles, ncol=1, frameon=True,
           loc='lower center', bbox_to_anchor=(0.5, 0))
plt.tight_layout(rect=[0, 0.05, 1, 0.95])

# ==============================================================================
# Plot 4: Strategy Fingerprint (Parallel Coordinates)
# ==============================================================================
param_labels = {
    'tau_boost':  'tau_boost (Nm)',
    'z_offset':   'z_offset (rad)',
    'amp_push':   'amp_push (rad)',
    'push_ratio': 'push_ratio (–)',
    'swing_time': 'swing_time (s)',
}
plot_para = all_trials_raw.dropna(subset=param_cols + ['Distance (cm)']).copy()
for p, lim in param_ranges.items():
    plot_para[p] = (plot_para[p] - lim[0]) / (lim[1] - lim[0])

# ==============================================================================
# Plot 4: Strategy Fingerprint (Parallel Coordinates)
# ==============================================================================
param_labels = {
    'tau_boost':  'tau_boost (Nm)',
    'z_offset':   'z_offset (rad)',
    'amp_push':   'amp_push (rad)',
    'push_ratio': 'push_ratio (–)',
    'swing_time': 'swing_time (s)',
}
plot_para = all_trials_raw.dropna(subset=param_cols + ['Distance (cm)']).copy()
for p, lim in param_ranges.items():
    plot_para[p] = (plot_para[p] - lim[0]) / (lim[1] - lim[0])
top10 = plot_para['Distance (cm)'].quantile(0.90)
top_df = plot_para[plot_para['Distance (cm)'] >= top10]
plt.figure(figsize=(14, 8))
#for _, row in top_df.iterrows():
#    plt.plot(param_cols, row[param_cols].values, color='lightgrey', alpha=0.35, linewidth=1, zorder=1)
medians = top_df.groupby('Algorithm')[param_cols].median()
#baseline_row = df_params_bare_idx.loc['Baseline (Manual)']
baseline_norm = [(baseline_row[p] - param_ranges[p][0]) / (param_ranges[p][1] - param_ranges[p][0]) for p in param_cols]
plt.plot(param_cols, medians.loc['BO'].values, color=PALETTE['BO'], linewidth=4, marker='o', label='BO Median (Top 10%)', zorder=5)
plt.plot(param_cols, medians.loc['CMA-ES'].values, color=PALETTE['CMA-ES'], linewidth=4, marker='o', label='CMA-ES Median (Top 10%)', zorder=6)
plt.plot(param_cols, baseline_norm, color=PALETTE['Baseline'], linewidth=3, linestyle='--', marker='s', label='Manual Baseline', zorder=7)
plt.title('Parameter Strategies of High-Performing Gaits', fontsize=20, weight='bold')
plt.ylabel('Normalized Parameter Value (0 = Min, 1 = Max)', fontsize=16)
plt.ylim(-0.05, 1.05)
plt.grid(True, alpha=0.4)
plt.legend(title='Parameter Strategy', loc='upper left', bbox_to_anchor=(0.6, 1.0), frameon=True)
plt.tight_layout()
if SAVE:
    plt.savefig("plot_4_parallel_strategy.png", dpi=300, bbox_inches="tight")
print("Ready: Plot 4 — Strategy Fingerprint (Parallel Coordinates)")
plt.show()
plt.close()

# ---------- Plot 5: Failure Rate (per-run mean ± SD, with clear annotations) ----------
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
means = fail_df.groupby('Algorithm')['Failure Rate (%)'].mean()
sds = fail_df.groupby('Algorithm')['Failure Rate (%)'].std(ddof=1)
xpos = [p.get_x() + p.get_width()/2 for p in ax.patches]
ax.errorbar(x=xpos, y=means.values, yerr=sds.values, fmt='none', ecolor='black', elinewidth=2, capsize=6, zorder=5)

# totals for annotation
totals = {
    'BO': (sum(count_failures(df) for df in bo_runs), sum(len(df) for df in bo_runs)),
    'CMA-ES': (sum(count_failures(df) for df in cma_runs), sum(len(df) for df in cma_runs))
}
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
if SAVE:
    plt.savefig("plot_5_failure_rate.png", dpi=300, bbox_inches="tight")
print("Ready: Plot 5 — Failure Rate")
plt.show()
plt.close()

# ---------- Plot 6 (Optional): Failure Heatmap ----------
print("\n--- Generating Optional 2D Failure Heatmap ---")
df_heat = all_trials_raw.copy()
df_heat['is_failure'] = (pd.to_numeric(df_heat.get('Score', np.nan), errors='coerce') == 1000).astype(int)

n_bins = 7
amp_bins = np.linspace(param_ranges['amp_push'][0], param_ranges['amp_push'][1], n_bins + 1)
swing_bins = np.linspace(param_ranges['swing_time'][0], param_ranges['swing_time'][1], n_bins + 1)

df_heat['amp_bin'] = pd.cut(df_heat['amp_push'], bins=amp_bins, include_lowest=True)
df_heat['swing_bin'] = pd.cut(df_heat['swing_time'], bins=swing_bins, include_lowest=True)

grp = df_heat.groupby(['amp_bin', 'swing_bin'])
counts = grp['is_failure'].count().unstack()
rates = (grp['is_failure'].mean()*100).unstack()
insufficient = (counts < 3)

def midpoints(bins):
    mids = (bins[:-1] + bins[1:]) / 2
    return [f"{m:.2f}" for m in mids]

plt.figure(figsize=(11, 9))
ax = sns.heatmap(
    rates.T, cmap='Reds', cbar_kws={'label': 'Failure Rate (%)'},
    vmin=0, vmax=max(1.0, np.nanmax(rates.values)),
    linewidths=0.5, linecolor='white'
)

# grey out n<3
for (i, j), val in np.ndenumerate(insufficient.T.values):
    if val:
        ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=True, color='lightgrey', zorder=2, alpha=0.8))

# annotate non-zero
for (i, j), val in np.ndenumerate(rates.T.values):
    if not np.isnan(val) and val > 0:
        ax.text(j+0.5, i+0.5, f"{val:.1f}%", ha='center', va='center', fontsize=10, color='black')

ax.set_xlabel('Swing Time (s)', fontsize=16)
ax.set_ylabel('Push Amplitude (rad)', fontsize=16)
ax.set_xticklabels(midpoints(swing_bins), rotation=45, ha='right')
ax.set_yticklabels(midpoints(amp_bins), rotation=0)
plt.title('Failure Rate by amp_push vs. swing_time', fontsize=20, weight='bold')
plt.tight_layout()
if SAVE:
    plt.savefig("plot_6_failure_heatmap.png", dpi=300, bbox_inches="tight")
print("Ready: Plot 6 — Failure Heatmap (amp_push vs swing_time)")
plt.show()
plt.close()

print("\nAnalysis complete.")
