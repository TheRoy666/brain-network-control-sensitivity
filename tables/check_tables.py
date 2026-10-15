#!/usr/bin/env python3
"""
check_tables.py — the numbers in Table 1 and Supplementary Tables S1 to S3,
computed from results/.

Supplementary Tables S1 to S3 are recomputed from the deposited result
files. Table 1 describes the cohorts; its values come from the source
datasets (HCP restricted data among them) and are therefore entered here as
constants, and only in aggregate.

With --check, every computed cell is compared with tables/published_tables.json,
the values as printed in the paper, and the script exits non-zero on any
difference.

USAGE (from the repository root; or `make tables`)
  python3 tables/check_tables.py --results results --out tables/out [--check]
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

SUP = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
COHORTS = ("HCP", "ADNI-400", "ADNI-450")
DASH = "—"


def sci(x):
    """Two significant figures; scientific notation below 0.01."""
    if x >= 0.01:
        return f"{x:.2f}"
    m, e = f"{x:.1e}".split("e")
    return f"{m}×10{str(int(e)).translate(SUP)}"


def med_iqr(v):
    q = np.quantile(v, [0.5, 0.25, 0.75])
    return f"{q[0]:.2f} ({q[1]:.2f}–{q[2]:.2f})"


# --------------------------------------------------------------- Table 1
TABLE1 = {
    "header": ["", "HCP", "ADNI", "Epilepsy (UNAM)"],
    "rows": [
        ["Participants", "238", "49", "62"],
        ["Groups", "80 MZ and 39 DZ twin pairs", "25 CN, 24 MCI", "34 healthy, 17 left TLE, 11 right TLE"],
        ["Female / male", "134 / 104", "26 / 23", "45 / 17"],
        ["Age, years (mean ± SD)", "29.6 ± 3.3", "78.4 ± 5.4", "31.5 ± 11.7"],
        ["Scanners", "Siemens 3T Skyra, single site", "Siemens 24, GE 16, Philips 9; multi-site", "single site"],
        ["Resting-state timepoints", "4800 (4 × 1200)", "197 or 200", "200 (59) or 130 (3)"],
        ["Repetition time", "0.72 s", "about 3 s", "[CONFIRM]"],
        ["Parcellation (regions)", "Glasser 360 + Tian S3 (410)", "Schaefer 400 (400); + Tian S3 (450)", "Desikan–Killiany (68)"],
        ["Timepoints per region", "11.71", "0.49 (400); 0.44 (450)", "2.94 or 1.91"],
        ["Structural connectome", "individual, SIFT2", "individual, SIFT2", "ENIGMA template"],
    ],
}


# ------------------------------------------------- Supplementary Table S1
def table_s1(R):
    h = pd.read_csv(f"{R}/01_operator_divergence/operator_divergence_vs_length_HCP.csv")
    f = pd.read_csv(f"{R}/01_operator_divergence/operator_divergence_frozen_states_HCP.csv")
    m = h.merge(f, on=["subject_id", "T_effective"], suffixes=("", "_fixed"))
    full = h[h.T_effective == h.T_effective.max()].set_index("subject_id").teleport_ratio
    N = int(h.N.iloc[0])
    rows = []
    for T in sorted(m.T_effective.unique(), reverse=True):
        g = m[m.T_effective == T]
        is_full = T == h.T_effective.max()
        lower = (g.teleport_ratio_fixed < g.teleport_ratio).mean()
        rose = (g.teleport_ratio.values > full.loc[g.subject_id].values).mean()
        rows.append([f"{T} (full)" if is_full else str(T), f"{T / N:.2f}",
                     med_iqr(g.teleport_ratio), med_iqr(g.teleport_ratio_fixed),
                     DASH if is_full else f"{lower:.0%}", DASH if is_full else f"{rose:.0%}"])
    return {"header": ["Timepoints", "Timepoints per region", "Ratio, states recomputed",
                       "Ratio, states held fixed", "Held-fixed lower", "Rose from full length"],
            "rows": rows}


# ------------------------------------------------- Supplementary Table S2
def table_s2(R):
    stats = {}
    for c in COHORTS:
        d = pd.read_csv(f"{R}/02_ec_uncertainty/graphnet_sensitivity_{c}.csv")
        stats[c] = {"ratio": f"{d.N.iloc[0] / d['T'].iloc[0]:.2f}"}
        for col, key in (("mean_tau_gu", "driver"), ("mean_tau", "receiver")):
            dd = d.dropna(subset=[col])
            pub = dd[(dd.lambda_ridge == 1) & (dd.lambda_graph == 1)][col].median()
            per_setting = dd.groupby(["lambda_ridge", "lambda_graph"])[col].median()
            pv = dd.pivot_table(index="subject_id", columns=["lambda_ridge", "lambda_graph"], values=col)
            spread = (pv.max(axis=1) - pv.min(axis=1)).median()
            stats[c][key] = [f"{pub:.3f}", f"{per_setting.min():.3f} – {per_setting.max():.3f}", sci(spread),
                             f"{dd[col].max():.3f}", f"{(dd[col] > 0.7).sum()} / {len(dd)}"]
    labels = ["   median τ, published setting", "   range of median τ across settings",
              "   within-subject spread across settings", "   best single cell", "   cells with τ > 0.7"]
    rows = [["Parameters per observation"] + [stats[c]["ratio"] for c in COHORTS]]
    for key, title in (("driver", "Driver ranking"), ("receiver", "Receiver ranking")):
        rows.append([title, "", "", ""])
        rows += [[lab] + [stats[c][key][i] for c in COHORTS] for i, lab in enumerate(labels)]
    return {"header": [""] + list(COHORTS), "rows": rows}


# ------------------------------------------------- Supplementary Table S3
def table_s3(R):
    med = {c: pd.read_csv(f"{R}/02_ec_uncertainty/block_length_{c}.csv")
               .groupby("block_length")[["mean_tau_gu", "mean_tau", "jaccard_top5_gu"]].median()
           for c in COHORTS}
    cell = lambda c, b, k: f"{med[c].loc[b, k]:.3f}" if b in med[c].index and pd.notna(med[c].loc[b, k]) else DASH
    rows = []
    for b in sorted(med["HCP"].index):
        rows.append([str(b), cell("HCP", b, "mean_tau_gu"), cell("HCP", b, "mean_tau"), cell("HCP", b, "jaccard_top5_gu"),
                     cell("ADNI-400", b, "mean_tau_gu"), cell("ADNI-400", b, "mean_tau"),
                     cell("ADNI-450", b, "mean_tau_gu"), cell("ADNI-450", b, "mean_tau")])
    return {"header": ["Block length", "HCP τ, driver", "HCP τ, receiver", "HCP top-5 Jaccard, driver",
                       "ADNI-400 τ, driver", "ADNI-400 τ, receiver", "ADNI-450 τ, driver", "ADNI-450 τ, receiver"],
            "rows": rows}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="tables/out")
    ap.add_argument("--check", action="store_true", help="compare with tables/published_tables.json")
    a = ap.parse_args()
    tables = {"Table_1": TABLE1, "Table_S1": table_s1(a.results),
              "Table_S2": table_s2(a.results), "Table_S3": table_s3(a.results)}
    os.makedirs(a.out, exist_ok=True)
    json.dump(tables, open(os.path.join(a.out, "tables.json"), "w"), ensure_ascii=False, indent=1)
    for name, t in tables.items():
        pd.DataFrame(t["rows"], columns=t["header"]).to_csv(os.path.join(a.out, f"{name}.csv"), index=False)
        print(f"\n{name}")
        print(pd.DataFrame(t["rows"], columns=t["header"]).to_string(index=False))
    if a.check:
        ref = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "published_tables.json")))
        bad = [(n, i, j, ref[n]["rows"][i][j], c) for n in ref for i, r in enumerate(tables[n]["rows"])
               for j, c in enumerate(r) if ref[n]["rows"][i][j] != c]
        bad += [(n, "header", None, ref[n]["header"], tables[n]["header"]) for n in ref if ref[n]["header"] != tables[n]["header"]]
        for b in bad:
            print("MISMATCH", b)
        print(f"\ncheck: {'every cell matches the published tables' if not bad else f'{len(bad)} mismatches'}")
        sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
