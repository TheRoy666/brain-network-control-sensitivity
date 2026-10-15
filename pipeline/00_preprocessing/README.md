# 00 · Preprocessing and connectomes

Produces, per subject, a regional BOLD timeseries and a structural connectome. Outputs are not deposited (see `docs/DATA_ACCESS.md`); the QC summaries are, in `results/cohort/`.

| cohort | order | scripts |
|---|---|---|
| ADNI-3 | 1. FastSurfer, fMRIPrep and QSIPrep in containers | `run_adni_pipeline.sh` |
| | 2. confound regression and parcellation (Schaefer 400; + Tian S3 = 450) | `extract_timeseries.py` |
| | 3. MRtrix3 tractography, SIFT2 weights, connectomes in both parcellations | `build_connectomes.sh` |
| | 4. QC of all three modalities, and of the connectomes | `qc_postproc.py`, `qc_connectomes.py` |
| Epilepsy (UNAM) | 1. FastSurfer and fMRIPrep; MRIQC | `run_unam_pipeline.sh`, `run_unam_mriqc.sh` |
| | 2. surface-based Desikan–Killiany parcellation, paired with the ENIGMA template connectome | `extract_timeseries_surface.py` |
| HCP | structural connectomes (Glasser 360 + Tian S3 = 410) from the released diffusion data; functional data are used as released | `SCMatrixExtractor.py` |

Paths and subject lists come from environment variables listed in `docs/DATA_ACCESS.md`.

`reported_values.py` recomputes every number the paper reports about the cohorts from `results/` and prints it beside the printed value (`make values` runs all stages).

