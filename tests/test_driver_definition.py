"""The two average-controllability definitions on planted directed operators.

Convention: x(t+1) = A x(t), so column j of A holds node j's outgoing
influence. Planted senders must top the driver ranking; planted receivers
must top the receiver ranking; on a symmetric matrix the two coincide.
"""
import warnings

import numpy as np
import pytest

from nctsens import physics

N, K = 40, 5


def _planted(kind, seed=0):
    rng = np.random.default_rng(seed)
    A = np.abs(rng.normal(scale=0.05, size=(N, N)))
    if kind == "senders":
        A[:, :K] += 1.0          # nodes 0..K-1 send to everyone
    else:
        A[:K, :] += 1.0          # nodes 0..K-1 receive from everyone
    return physics.normalise_matrix(A, target_rho=0.9)


def _top(v):
    return set(np.argsort(v)[::-1][:K])


def test_driver_finds_planted_senders():
    ac = physics.average_controllability(_planted("senders"), direction="driver")
    assert _top(ac) == set(range(K))


def test_receiver_finds_planted_receivers():
    ac = physics.average_controllability(_planted("receivers"), direction="receiver")
    assert _top(ac) == set(range(K))


def test_definitions_differ_on_directed_operator():
    A = _planted("senders")
    d = physics.average_controllability(A, direction="driver")
    r = physics.average_controllability(A, direction="receiver")
    assert _top(d) != _top(r)


def test_definitions_coincide_on_symmetric_operator():
    rng = np.random.default_rng(1)
    S = np.abs(rng.normal(size=(N, N)))
    S = physics.normalise_matrix(S + S.T, target_rho=0.9)
    np.testing.assert_allclose(physics.average_controllability(S, direction="driver"),
                               physics.average_controllability(S, direction="receiver"))


def test_default_is_receiver_and_warns_on_directed_input():
    A = _planted("senders")
    with pytest.warns(UserWarning, match="receiver"):
        ac = physics.average_controllability(A)
    np.testing.assert_array_equal(ac, physics.average_controllability(A, direction="receiver"))


def test_no_warning_on_symmetric_input():
    S = physics.normalise_matrix(np.ones((N, N)) + np.eye(N), target_rho=0.9)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        physics.average_controllability(S)


def test_driver_matches_explicit_series():
    A = _planted("senders")
    series = sum(np.sum(np.linalg.matrix_power(A, t) ** 2, axis=0) for t in range(400))
    np.testing.assert_allclose(physics.driver_average_controllability(A), series, rtol=1e-6)


def test_bad_direction_raises():
    with pytest.raises(ValueError):
        physics.average_controllability(np.eye(3) * 0.5, direction="sideways")
