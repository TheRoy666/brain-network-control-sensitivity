"""
compare_real_reward_vs_original.py — the HCP alcohol-use analysis with the anatomical reward
parcellation against the original proxy definition, subject by subject.

    python3 compare_real_reward_vs_original.py <real_reward_batch.log> <original_batch.log>

Each log is the console output of run_hcp_aud_batch_real_reward.py and run_hcp_aud_batch.py, one line
per subject ("[  1/238] <id> Sev=0 (Social_Drinker) E_crav→rest=0.2469  teleport=3.55x").
HCP identifiers and severity groups are restricted data, so the logs stay with the data.
"""
import re
import sys
import numpy as np
from scipy.stats import spearmanr, pearsonr, mannwhitneyu

REAL_REWARD_TEXT = open(sys.argv[1]).read()
ORIGINAL_TEXT = open(sys.argv[2]).read()



def parse(text):
    pattern = re.compile(
        r"(\d+)\s+Sev=(\d)\s+\(([\w_]+)\s*\)\s+E_crav→rest=([\d.]+)\s+teleport=([\d.]+)x"
    )
    out = {}
    for m in pattern.finditer(text):
        sid, sev, label, e, tp = m.groups()
        out[sid] = {"sev": int(sev), "label": label, "E": float(e), "tp": float(tp)}
    return out


real = parse(REAL_REWARD_TEXT)
orig = parse(ORIGINAL_TEXT)

print(f"Parsed: real_reward={len(real)}, original={len(orig)}")

common = sorted(set(real) & set(orig))
print(f"Common subjects: {len(common)}")

E_real = np.array([real[s]["E"] for s in common])
E_orig = np.array([orig[s]["E"] for s in common])
tp_real = np.array([real[s]["tp"] for s in common])
tp_orig = np.array([orig[s]["tp"] for s in common])
sev = np.array([real[s]["sev"] for s in common])

print("\n=== Per-subject correlation: real-reward vs original-proxy ===")
rho_E, p_E = spearmanr(E_real, E_orig)
r_E, pr_E = pearsonr(E_real, E_orig)
print(f"E_craving_to_rest: Spearman rho={rho_E:.4f} (p={p_E:.2e}), Pearson r={r_E:.4f}")

rho_tp, p_tp = spearmanr(tp_real, tp_orig)
print(f"teleport_ratio:    Spearman rho={rho_tp:.4f} (p={p_tp:.2e})")

print("\n=== Per-subject absolute change ===")
diff = E_real - E_orig
print(f"E_craving_to_rest change: mean={diff.mean():.4f}, std={diff.std():.4f}, "
      f"median={np.median(diff):.4f}")

print("\n=== Group comparison, BOTH versions ===")
for label_sev, label_name in [(0, "Social_Drinker"), (1, "Abuser"), (2, "Dependent")]:
    mask = sev == label_sev
    print(f"{label_name} (N={mask.sum()}): "
          f"real_median={np.median(E_real[mask]):.4f}  "
          f"orig_median={np.median(E_orig[mask]):.4f}")

print("\n=== Mann-Whitney U: Social_Drinker vs Abuser, BOTH versions ===")
sd_real = E_real[sev == 0]; ab_real = E_real[sev == 1]
sd_orig = E_orig[sev == 0]; ab_orig = E_orig[sev == 1]
u_real, p_u_real = mannwhitneyu(sd_real, ab_real, alternative='two-sided')
u_orig, p_u_orig = mannwhitneyu(sd_orig, ab_orig, alternative='two-sided')
print(f"Real-reward:  U={u_real:.0f}, p={p_u_real:.4f}")
print(f"Original:     U={u_orig:.0f}, p={p_u_orig:.4f}")

print("\n=== Mann-Whitney U: Social_Drinker vs Dependent, BOTH versions ===")
dp_real = E_real[sev == 2]; dp_orig = E_orig[sev == 2]
u_real2, p_u_real2 = mannwhitneyu(sd_real, dp_real, alternative='two-sided')
u_orig2, p_u_orig2 = mannwhitneyu(sd_orig, dp_orig, alternative='two-sided')
print(f"Real-reward:  U={u_real2:.0f}, p={p_u_real2:.4f}")
print(f"Original:     U={u_orig2:.0f}, p={p_u_orig2:.4f}")
