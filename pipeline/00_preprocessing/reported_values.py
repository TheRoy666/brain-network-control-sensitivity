#!/usr/bin/env python3
"""
reported_values.py — cohort numbers in the text (Methods 4.2 and 4.4,
Supplementary S6 "Cohort details") recomputed from the deposited files in
results/ and compared with the value printed in the paper.

    python3 pipeline/00_preprocessing/reported_values.py [--results results]

The epilepsy ages need the public OpenNeuro participants file joined through
the local pseudonym key, so they are listed rather than checked. HCP demographics
and twin structure are restricted data (docs/DATA_ACCESS.md) and are checked
by tables/check_tables.py against Table 1.
"""
import argparse
import os
import sys

import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from reported import Report  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    a = ap.parse_args()
    R = lambda p: os.path.join(a.results, p)
    rep = Report("00 Cohorts")

    rep.section("4.2 ADNI-3")
    q = pd.read_csv(R("cohort/adni_qc.csv"))
    res = pd.read_csv(R("05_clinical/adni_results_400.csv"))
    m = q.merge(res[["subject_id", "manufacturer"]], left_on="subject", right_on="subject_id")
    rep.check("participants", len(res), 49, digits=0)
    for man, n in (("Siemens", 24), ("GE", 16), ("Philips", 9)):
        rep.check(f"{man} scanners", int((res.manufacturer == man).sum()), n, digits=0)
    rep.check("timepoints are 197 or 200", set(m.func_ntimepoints.astype(int)) == {197, 200}, True)
    rep.check("the 200-timepoint recordings are exactly the GE ones",
              set(m[m.func_ntimepoints == 200].manufacturer) == {"GE"} and (m.manufacturer == "GE").sum() == (m.func_ntimepoints == 200).sum(), True)
    ts = m.groupby("manufacturer").func_tsnr.median()
    for man, v in (("GE", 54.5), ("Siemens", 281.4), ("Philips", 351.4)):
        rep.check(f"median tSNR, {man}", ts[man], v)
    rep.check("GE vs other manufacturers, Mann-Whitney p", stats.mannwhitneyu(m[m.manufacturer == "GE"].func_tsnr,
                                                                             m[m.manufacturer != "GE"].func_tsnr).pvalue, 1.9e-8, kind="sci")
    rep.check("median mean framewise displacement (mm)", m.func_fd_mean.median(), 0.153)

    rep.section("4.2 UNAM temporal lobe epilepsy")
    u = pd.read_csv(R("05_clinical/unam_results.csv"))
    rep.check("analyzed", len(u), 62, digits=0)
    for g, n in (("Healthy", 34), ("Left-TLE", 17), ("Right-TLE", 11)):
        rep.check(f"{g}", int((u.TLEside == g).sum()), n, digits=0)
    rep.check("women", int((u.gender == "F").sum()), 45, digits=0)
    rep.check("men", int((u.gender == "M").sum()), 17, digits=0)
    rep.check("recordings of 130 timepoints", int((u["T"] == 130).sum()), 3, digits=0)
    rep.server_only([
        ("UNAM ages (mean 31.5, median 28, range 17-57; lateralized groups 30.4 vs 29.2, Mann-Whitney p 0.48)",
         "OpenNeuro ds004469 participants.tsv joined through the local pseudonym key (not deposited)"),
        ("UNAM release: 74 participants, 4 bilateral and 4 unknown focus excluded by design", "OpenNeuro ds004469 participants.tsv"),
        ("HCP: 238 individuals, 119 pairs (80 MZ, 39 DZ), 134 women and 104 men, aged 22-37",
         "Table 1; restricted HCP data, checked by tables/check_tables.py"),
        ("ADNI: about 27 sites; one subject with 15 diffusion directions excluded", "site and diffusion metadata are not deposited"),
        ("Connectome QC: 48 of 49 ADNI subjects pass; median edge density 20.5% (450) and 18.9% (400); one at 9.5% density",
         "pipeline/00_preprocessing/qc_connectomes.py (connectomes are not deposited)"),
        ("UNAM QC: one excluded for motion (0.623 mm), two for tSNR (21.68, 19.50), one without a resting run; 16 flagged",
         "pipeline/00_preprocessing/run_unam_mriqc.sh outputs"),
    ])
    sys.exit(rep.finish())


if __name__ == "__main__":
    main()
