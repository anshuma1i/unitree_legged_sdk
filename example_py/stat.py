# ==============================================================================
# FINAL STATS BATTERY: Champions (3v3) + Efficiency (BO_1 vs CMAES_1)
# Files required in the same folder:
#   - best_fitness_Sheet1.csv
#   - optimization_results_BO_1.csv
#   - optimization_log_cmaes_1.csv
# ==============================================================================

import numpy as np
import pandas as pd
from scipy import stats
from itertools import combinations

RNG = np.random.default_rng(123)

# ----------------------------
# Helpers
# ----------------------------
def bootstrap_mean_ci(x, B=10000, alpha=0.05, rng=RNG):
    """Percentile bootstrap CI for the mean of 1D array x."""
    x = np.asarray(x, float)
    n = len(x)
    boots = rng.choice(x, size=(B, n), replace=True).mean(axis=1)
    lo, hi = np.percentile(boots, [100*alpha/2, 100*(1-alpha/2)])
    return lo, hi

def bootstrap_diff_ci(x, y, B=10000, alpha=0.05, rng=RNG):
    """Percentile bootstrap CI for mean(x) - mean(y)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    nx, ny = len(x), len(y)
    bx = rng.choice(x, size=(B, nx), replace=True).mean(axis=1)
    by = rng.choice(y, size=(B, ny), replace=True).mean(axis=1)
    diffs = bx - by
    lo, hi = np.percentile(diffs, [100*alpha/2, 100*(1-alpha/2)])
    return lo, hi

def exact_permutation_pvalues(x, y):
    """Exact permutation test for difference in means (x - y).
       Returns one-sided p (>= observed) and two-sided p (|diff| >= |obs|)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    all_vals = np.concatenate([x, y])
    k = len(x)
    obs = x.mean() - y.mean()

    ge, ge_abs, total = 0, 0, 0
    for idx in combinations(range(len(all_vals)), k):
        total += 1
        perm_x = all_vals[list(idx)]
        perm_y = np.delete(all_vals, idx)
        d = perm_x.mean() - perm_y.mean()
        if d >= obs - 1e-12:
            ge += 1
        if abs(d) >= abs(obs) - 1e-12:
            ge_abs += 1
    return ge/total, ge_abs/total, obs

def load_distance_series(df):
    """Return per-trial distance series, handling 'Score' (1000=failure) or 'Distance (cm)'. """
    df = df.copy()
    if 'Distance (cm)' in df.columns:
        dist = pd.to_numeric(df['Distance (cm)'], errors='coerce')
        if 'Score' in df.columns:
            s = pd.to_numeric(df['Score'], errors='coerce')
            dist = dist.where(s != 1000, 0.0)  # failures -> 0
        dist = dist.fillna(0.0)
    else:
        s = pd.to_numeric(df['Score'], errors='coerce').fillna(1000)
        dist = -s
        dist = dist.where(s != 1000, 0.0)    # failures -> 0
    return dist.to_numpy()

def best_so_far(x):
    x = np.asarray(x, float)
    return np.maximum.accumulate(x)

def mannwhitney_one_sided(x, y):
    """Mann–Whitney U test (one-sided, x > y). Chooses exact/asymptotic based on SciPy support."""
    # SciPy >=1.9 supports method='exact' for small samples without ties.
    try:
        res = stats.mannwhitneyu(x, y, alternative='greater', method='auto')
    except TypeError:
        # Older SciPy: no 'method' arg
        res = stats.mannwhitneyu(x, y, alternative='greater')
    return res.statistic, res.pvalue

# ==============================================================================
# PART A: FINAL PERFORMANCE (3 champions vs 3 champions)
# ==============================================================================
print("="*80)
print("PERFORMANCE TESTS (Champion means: 3 BO vs 3 CMA-ES)")
print("="*80)

# Load champion validation runs (15 per algorithm = 3 champs x 5 re-runs)
dfv = pd.read_csv("best_fitness_Sheet1.csv")
bo_vals  = dfv['BO Re-run'].dropna().astype(float).to_numpy()
cma_vals = dfv['CMAES Re-run'].dropna().astype(float).to_numpy()

assert len(bo_vals)  == 15, f"Expected 15 BO validation values, got {len(bo_vals)}"
assert len(cma_vals) == 15, f"Expected 15 CMA-ES validation values, got {len(cma_vals)}"

# Reshape into (3,5) and compute champion means
bo_means  = bo_vals.reshape(3, 5).mean(axis=1)
cma_means = cma_vals.reshape(3, 5).mean(axis=1)

print("\nChampion means (cm):")
print(f"  BO   : {np.round(bo_means,  1)}  -> mean={bo_means.mean():.1f}")
print(f"  CMA-ES: {np.round(cma_means,1)}  -> mean={cma_means.mean():.1f}")

# 1) Mann–Whitney U (one-sided, BO > CMA-ES)
U, p_mwu = mannwhitney_one_sided(bo_means, cma_means)
print("\n[1] Mann–Whitney U (one-sided, BO > CMA-ES)")
print(f"    U = {U:.1f}, p = {p_mwu:.4f}")

# 2) Exact permutation test (one- and two-sided) on means
p_perm_one, p_perm_two, obs_diff = exact_permutation_pvalues(bo_means, cma_means)
print("\n[2] Exact permutation test on mean difference (BO − CMA-ES)")
print(f"    Observed mean diff = {obs_diff:.2f} cm")
print(f"    p (one-sided, ≥obs) = {p_perm_one:.4f}")
print(f"    p (two-sided, |diff|≥|obs|) = {p_perm_two:.4f}")

# 3) Mean difference (already above) + 4) 95% CIs (bootstrap)
bo_ci_lo,  bo_ci_hi  = bootstrap_mean_ci(bo_means,  B=10000, alpha=0.05, rng=RNG)
cma_ci_lo, cma_ci_hi = bootstrap_mean_ci(cma_means, B=10000, alpha=0.05, rng=RNG)
diff_ci_lo, diff_ci_hi = bootstrap_diff_ci(bo_means, cma_means, B=10000, alpha=0.05, rng=RNG)

print("\n[3] Mean difference (BO − CMA-ES)")
print(f"    {obs_diff:.2f} cm  (bootstrap 95% CI for diff: [{diff_ci_lo:.2f}, {diff_ci_hi:.2f}] cm)")

print("\n[4] 95% bootstrap CIs for champion mean (per algorithm)")
print(f"    BO mean   95% CI: [{bo_ci_lo:.2f}, {bo_ci_hi:.2f}] cm")
print(f"    CMA-ES mean 95% CI: [{cma_ci_lo:.2f}, {cma_ci_hi:.2f}] cm")

# ==============================================================================
# PART B: LEARNING EFFICIENCY (single representative runs)
# ==============================================================================
print("\n" + "="*80)
print("EFFICIENCY TEST (Single runs: BO_1 vs CMA-ES_1)")
print("="*80)

# Load the per-trial histories
bo1 = pd.read_csv("optimization_results_BO_1.csv")
cma1 = pd.read_csv("optimization_log_cmaes_1.csv")

bo_dist  = load_distance_series(bo1)
cma_dist = load_distance_series(cma1)

# Build best-so-far curves
bo_bsf  = best_so_far(bo_dist)
cma_bsf = best_so_far(cma_dist)

# Align to common length (robust to files with <50 or >50 rows)
n = min(len(bo_bsf), len(cma_bsf))
bo_bsf  = bo_bsf[:n]
cma_bsf = cma_bsf[:n]

# 5) Mann–Whitney U on per-trial best-so-far values (one-sided, BO > CMA-ES)
U_eff, p_eff = mannwhitney_one_sided(bo_bsf, cma_bsf)
auc_prob = U_eff / (len(bo_bsf) * len(cma_bsf))  # Prob(BO per-trial BSF > CMA-ES per-trial BSF)

print("\n[5] Mann–Whitney U on per-trial best-so-far (one-sided, BO > CMA-ES)")
print(f"    Trials compared: {n} × {n} pairs")
print(f"    U = {U_eff:.1f}, p = {p_eff:.6f}")
print(f"    Interpretation (U/(n*n)): Prob(BO > CMA-ES at a random trial) ≈ {auc_prob*100:.1f}%")

print("\nDone.")
