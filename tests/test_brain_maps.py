"""pipeline/02_ec_uncertainty/brain_maps.py: driver score, sign convention and export."""
import importlib.util
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd

from nctsens import physics

_P = os.path.join(os.path.dirname(__file__), "..", "pipeline", "02_ec_uncertainty", "brain_maps.py")
_spec = importlib.util.spec_from_file_location("brain_maps", _P)
bm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bm)


def _planted(kind, n=40, seed=0):
    rng = np.random.default_rng(seed)
    A = np.abs(rng.normal(0, 0.02, (n, n)))
    if kind == "senders":
        A[:, :5] += 0.6
    else:
        A[:5, :] += 0.6
    return A


def test_driver_score_is_the_package_driver_definition():
    A = _planted("senders")
    expected = physics.driver_average_controllability(physics.normalise_matrix(A, target_rho=0.9))
    np.testing.assert_allclose(bm.driver_score(A), expected, rtol=1e-10)


def test_driver_score_finds_senders_not_receivers():
    assert set(np.argsort(-bm.driver_score(_planted("senders")))[:5]) == set(range(5))
    assert not set(np.argsort(-bm.driver_score(_planted("receivers")))[:5]) & set(range(5))


def test_out_minus_in_sign():
    assert np.all(bm.out_minus_in(_planted("senders"))[:5] > 0)
    assert np.all(bm.out_minus_in(_planted("receivers"))[:5] < 0)


def test_identical_resamples_are_perfectly_stable():
    r = bm.score_ensemble(np.repeat(_planted("senders")[None], 4, axis=0))
    assert r["jaccard"] == 1.0 and np.isclose(r["tau"], 1.0)
    assert np.isclose(r["freq"].sum(), 5) and set(np.flatnonzero(r["freq"] == 1)) == set(range(5))


def test_row_blocks_cover_each_connectome():
    for cohort, n in (("HCP", 410), ("ADNI-400", 400), ("ADNI-450", 450)):
        t = bm.regions(cohort, n)
        assert list(t.row) == list(range(n))
    assert bm.regions("HCP", 410).set_index("row").loc[50, "block"] == "HCP-MMP right"


def test_export_drops_per_subject_rows(tmp_path):
    rng = np.random.default_rng(1)
    for c, n in (("HCP", 410), ("ADNI-400", 400), ("ADNI-450", 450)):
        np.savez(tmp_path / f"driver_map_{c}.npz", freq=rng.random(n), mean_score=rng.random(n),
                 shared=rng.integers(0, 6, 10), jaccard=0.5, tau=0.8, n_boot=200, n_regions=n, rule="test")
        np.savez(tmp_path / f"sender_receiver_map_{c}.npz", mean=rng.normal(size=n), per_subject=rng.normal(size=(3, n)),
                 sign_agreement=rng.random(n), n_subjects=3, n_regions=n)
    out = tmp_path / "out"
    bm.cmd_export(SimpleNamespace(maps=str(tmp_path), out=str(out)))
    s = pd.read_csv(out / "sender_receiver_map_HCP.csv")
    assert list(s.columns) == ["row", "block", "parcel", "out_minus_in", "sign_agreement"] and len(s) == 410
    assert len(pd.read_csv(out / "brain_maps_summary.csv")) == 6
