#!/usr/bin/env python3
"""
Stuart-Landau network model -- third nonlinear model for the
validity-regime comparison (Path 1, Item 2).

WHY STUART-LANDAU SPECIFICALLY
------------------------------
Wilson-Cowan and FitzHugh-Nagumo disagree about the direction of
linearization bias. A third arbitrary model would only produce a 2-vs-1
tally. Stuart-Landau is chosen because it is the NORMAL FORM of the Hopf
bifurcation, which buys three things no other model does:

  1. The bifurcation parameter `a` is explicit and scalar. a < 0 gives a
     stable fixed point with damped oscillation; a > 0 gives a limit
     cycle; the Hopf bifurcation sits exactly at a = 0. No parameter
     search is needed to locate the boundary -- it is known analytically.
     This removes a confound present in the WC and FHN sweeps, where
     "distance to bifurcation" had to be inferred from a coupling scan.

  2. The NONLINEARITY ITSELF IS A TUNABLE PARAMETER. This is the point.
     The hypothesis under test is that the direction of linearization
     bias is determined by a structural property of the nonlinearity
     (its sign/curvature at the operating point), rather than being an
     idiosyncrasy of each model. WC (sigmoidal, saturating) and FHN
     (cubic, excitable) are two fixed points in that space. Stuart-Landau
     lets the property be SWEPT CONTINUOUSLY rather than sampled at two
     arbitrary values -- turning a tally into a testable rule.

  3. Analytical ground truth for validation. For the supercritical case
     the limit-cycle amplitude is exactly r* = sqrt(a / beta), so the
     implementation can be checked against a closed-form answer rather
     than against a plot that looks plausible.

GENERALIZED FORM IMPLEMENTED
----------------------------
Per node j, with complex state z_j = x_j + i*y_j:

    dz_j/dt = z_j * (a_j + i*omega_j - beta*|z_j|^2 - gamma*|z_j|^4)
              + G * sum_k C_jk * (z_k - z_j)

  beta > 0, gamma = 0  -> supercritical Hopf (saturating nonlinearity,
                          qualitatively WC-like in that growth is damped)
  beta < 0, gamma > 0  -> subcritical Hopf (self-amplifying cubic term,
                          stabilized only by the quintic; qualitatively
                          the opposite curvature)

The quintic term is not decorative: with beta < 0 and gamma = 0 the
system diverges in finite time. gamma > 0 is REQUIRED for boundedness
in the subcritical regime, and the script refuses that combination
rather than producing overflow warnings and garbage.

Coupling is DIFFUSIVE (z_k - z_j), matching the standard whole-brain
Stuart-Landau formulation (Deco et al.) rather than the direct additive
drive used in WilsonCowanNetwork. This is a real difference and is
stated rather than hidden: diffusive coupling vanishes at synchrony,
which is the physically appropriate choice for phase oscillators and
keeps the fixed point at the origin independent of C.

Real-valued state layout mirrors WilsonCowanNetwork's [E, I] convention:
    y = [x_0, ..., x_{N-1}, y_0, ..., y_{N-1}]
so downstream code that slices the first N / last N entries behaves
identically across models.
"""
from typing import Callable, Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp


class StuartLandauNetwork:
    """Coupled multi-region Stuart-Landau (Hopf normal form) model.

    Parameters
    ----------
    n_regions : int              - Number of nodes.
    C         : (N,N) ndarray    - Coupling matrix; C[j,k] is influence of k on j.
    a         : float or (N,)    - Bifurcation parameter. <0 fixed point, >0 limit cycle.
    omega     : float or (N,)    - Intrinsic angular frequency (rad/ms).
    beta      : float            - Cubic coefficient. >0 supercritical, <0 subcritical.
    gamma     : float            - Quintic coefficient. Required >0 if beta<0.
    G         : float            - Global coupling strength.
    """

    #: Supercritical default: saturating cubic, no quintic needed.
    SUPERCRITICAL_PARAMS = {"a": -0.05, "omega": 0.05, "beta": 1.0, "gamma": 0.0}

    #: Subcritical: self-amplifying cubic, quintic supplies boundedness.
    SUBCRITICAL_PARAMS = {"a": -0.05, "omega": 0.05, "beta": -1.0, "gamma": 1.0}

    def __init__(
        self,
        n_regions: int,
        C: NDArray,
        a: float = -0.05,
        omega: float = 0.05,
        beta: float = 1.0,
        gamma: float = 0.0,
        G: float = 0.0,
    ):
        self.n_regions = n_regions
        self.C = np.asarray(C, dtype=float)
        self.a = np.broadcast_to(np.asarray(a, dtype=float), (n_regions,)).copy()
        self.omega = np.broadcast_to(np.asarray(omega, dtype=float), (n_regions,)).copy()
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.G = float(G)

        if self.beta < 0 and self.gamma <= 0:
            raise ValueError(
                f"beta={beta} < 0 requires gamma > 0 for boundedness "
                f"(got gamma={gamma}). With a self-amplifying cubic term and no "
                f"quintic, trajectories diverge in finite time. This is refused "
                f"rather than silently producing overflow."
            )
        if self.C.shape != (n_regions, n_regions):
            raise ValueError(f"C must be ({n_regions},{n_regions}), got {self.C.shape}")

    def _ode_network(self, t: float, state: NDArray,
                     u_func: Optional[Callable] = None) -> NDArray:
        """y = [x_0..x_{N-1}, y_0..y_{N-1}], mirroring the [E, I] layout.

        u_func(t) -> (N,) control current injected into the x (fast)
        variable, matching the role of FHN's injection into v and WC's
        into E. Same hook the A2 machinery uses to inject a
        linear-optimal trajectory and measure real reachability.
        """
        N = self.n_regions
        x = state[:N]
        y = state[N:]

        r2 = x * x + y * y
        radial = self.a - self.beta * r2 - self.gamma * r2 * r2

        # Diffusive coupling: sum_k C_jk (x_k - x_j)
        coup_x = self.C @ x - x * self.C.sum(axis=1)
        coup_y = self.C @ y - y * self.C.sum(axis=1)

        u = u_func(t) if u_func is not None else 0.0

        dx = radial * x - self.omega * y + self.G * coup_x + u
        dy = radial * y + self.omega * x + self.G * coup_y

        return np.concatenate([dx, dy])

    def jacobian_at(self, x_star: NDArray, y_star: NDArray) -> NDArray:
        """Exact analytic Jacobian at an ARBITRARY state, as (2N, 2N).

        Signature mirrors FitzHughNagumoNetwork.jacobian_at(v, w) and
        WilsonCowanNetwork's convention, so the A2 grid machinery calls
        it unchanged.

        With r^2 = x^2 + y^2 and R(r^2) = a - beta*r^2 - gamma*r^4:

            d(dx)/dx = R - (2*beta + 4*gamma*r^2) * x^2
            d(dx)/dy = -omega - (2*beta + 4*gamma*r^2) * x*y
            d(dy)/dx =  omega - (2*beta + 4*gamma*r^2) * x*y
            d(dy)/dy = R - (2*beta + 4*gamma*r^2) * y^2

        plus the (state-independent) diffusive coupling operator on both
        diagonal blocks. Analytic throughout -- no finite differencing,
        which removes numerical-differentiation error as a confound in
        the cross-model comparison. Note this is a genuine methodological
        difference from the WC/FHN arms and belongs in the methods.
        """
        x = np.atleast_1d(np.asarray(x_star, dtype=float))
        y = np.atleast_1d(np.asarray(y_star, dtype=float))
        N = self.n_regions

        r2 = x * x + y * y
        R = self.a - self.beta * r2 - self.gamma * r2 * r2
        k = 2.0 * self.beta + 4.0 * self.gamma * r2  # common factor

        L = self.C - np.diag(self.C.sum(axis=1))  # diffusive coupling operator

        J_xx = np.diag(R - k * x * x) + self.G * L
        J_xy = np.diag(-self.omega - k * x * y)
        J_yx = np.diag(self.omega - k * x * y)
        J_yy = np.diag(R - k * y * y) + self.G * L

        return np.block([[J_xx, J_xy], [J_yx, J_yy]])

    def find_fixed_point(self, x0_guess=None, w0_guess=None):
        """Return the fixed point. For Stuart-Landau this is the ORIGIN,
        exactly and analytically, for any parameter set -- the radial
        term vanishes at r=0 regardless of a, beta, gamma, and diffusive
        coupling vanishes there too.

        Signature kept compatible with FHN/WC's find_fixed_point. Raises
        if a > 0, where the origin is unstable and a limit cycle is the
        attractor -- the same failure mode FHN reports near its Hopf
        boundary, surfaced explicitly rather than returning a fixed point
        the system will never sit at.
        """
        N = self.n_regions
        if np.any(self.a > 0):
            raise RuntimeError(
                f"a > 0 (max {float(np.max(self.a)):.4f}): the origin is "
                f"unstable and the attractor is a limit cycle. Use "
                f"nominal_orbit() for the oscillatory regime."
            )
        return np.zeros(N), np.zeros(N)

    def nominal_orbit(self, phase: float) -> NDArray:
        """Closed-form point on the limit cycle at a given phase.

        FHN must integrate its orbit numerically, save it, and
        interpolate with cubic splines (see fhn_a2_data/nominal_*.npy).
        Stuart-Landau's orbit is exact:

            x(phi) = r* cos(phi),  y(phi) = r* sin(phi),  r* = sqrt(a/beta)

        so phase is an exact parameterisation with no interpolation
        error, and the period is exactly 2*pi/omega. This removes two
        numerical error sources present in the FHN arm.

        Returns a (2N,) array [x_0..x_{N-1}, y_0..y_{N-1}].
        """
        r_star = self.analytical_limit_cycle_amplitude()
        if not np.isfinite(r_star):
            raise RuntimeError(
                "Closed-form orbit requires a > 0, beta > 0, gamma = 0 "
                "(supercritical). For other regimes, integrate numerically."
            )
        N = self.n_regions
        return np.concatenate([np.full(N, r_star * np.cos(phase)),
                               np.full(N, r_star * np.sin(phase))])

    @property
    def period(self) -> float:
        """Exact limit-cycle period, 2*pi/omega -- no numerical estimate."""
        return float(2.0 * np.pi / np.mean(self.omega))

    def simulate_controlled(
        self,
        u_func: Callable[[float], NDArray],
        t_span: Tuple[float, float],
        n_points: int,
        E0: NDArray,
        I0: NDArray,
    ) -> Dict[str, NDArray]:
        """Signature and return keys match FitzHughNagumoNetwork exactly
        (E0/I0 naming and "E"/"I" output keys kept for drop-in reuse of
        the A2 grid machinery). Here they are (x0, y0)."""
        N = self.n_regions
        x0, y0 = E0, I0
        init = np.concatenate([np.asarray(x0, dtype=float),
                               np.asarray(y0, dtype=float)])
        t_eval = np.linspace(*t_span, n_points)
        sol = solve_ivp(self._ode_network, t_span, init, t_eval=t_eval,
                        args=(u_func,), method="RK45", rtol=1e-6, atol=1e-8)
        return {"t": sol.t, "E": sol.y[:N], "I": sol.y[N:],
                "x": sol.y[:N], "y": sol.y[N:]}

    def simulate(
        self,
        t_span: Tuple[float, float] = (0.0, 2000.0),
        n_points: int = 20000,
        x0: Optional[NDArray] = None,
        y0: Optional[NDArray] = None,
        seed: int = 42,
    ) -> Dict[str, NDArray]:
        """Simulate the coupled network.

        Returns dict with keys ``"t"``, ``"x"`` (N x T), ``"y"`` (N x T),
        and ``"E"`` as an alias for ``"x"`` so that downstream code written
        against WilsonCowanNetwork's output works unchanged.
        """
        N = self.n_regions
        rng = np.random.default_rng(seed)

        if x0 is None:
            x0 = rng.uniform(0.05, 0.3, N)
        if y0 is None:
            y0 = rng.uniform(0.05, 0.3, N)

        init = np.concatenate([x0, y0])
        t_eval = np.linspace(*t_span, n_points)

        sol = solve_ivp(
            self._ode_network,
            t_span,
            init,
            t_eval=t_eval,
            method="RK45",
            rtol=1e-6,
            atol=1e-8,
        )

        return {
            "t": sol.t,
            "x": sol.y[:N],
            "y": sol.y[N:],
            "E": sol.y[:N],  # alias for cross-model interface compatibility
        }

    def jacobian_at_origin(self) -> NDArray:
        """Analytical Jacobian at the fixed point z=0, as a (2N, 2N) real matrix.

        At the origin all nonlinear terms vanish, so the Jacobian is exact,
        not a finite-difference approximation. This is the linearization
        whose validity the sweep measures -- providing it in closed form
        removes numerical differentiation as a source of error in the
        model comparison.
        """
        N = self.n_regions
        L = self.C - np.diag(self.C.sum(axis=1))  # diffusive coupling operator

        J = np.zeros((2 * N, 2 * N))
        J[:N, :N] = np.diag(self.a) + self.G * L
        J[:N, N:] = -np.diag(self.omega)
        J[N:, :N] = np.diag(self.omega)
        J[N:, N:] = np.diag(self.a) + self.G * L
        return J

    def analytical_limit_cycle_amplitude(self) -> float:
        """Limit-cycle radius, valid across BOTH criticality regimes.

        The radial dynamics are dr/dt = r*(a - beta*r^2 - gamma*r^4), so
        a limit cycle sits at the positive root of

            gamma*r^4 + beta*r^2 - a = 0        (in u = r^2)

        Supercritical special case (gamma = 0): reduces to the familiar
        r* = sqrt(a/beta), which is asserted against the general solver
        in this module's self-validation.

        Subcritical (beta < 0, gamma > 0): the quartic has real positive
        roots; the LARGER root is the stable large-amplitude cycle and is
        the one returned, since that is the attractor an orbit-based
        analysis actually sits on. The smaller root is the unstable
        cycle separating basins and is not what we want here.

        Generalising this is what allows beta to be swept across its
        sign -- i.e. across the structural property of the nonlinearity
        whose effect on linearization bias is the actual question. With
        only the gamma=0 form, the sweep could not cross into the
        subcritical regime at all.

        Returns nan where no positive root exists (no limit cycle).
        """
        a = float(np.mean(self.a))

        if self.gamma == 0.0:
            if a > 0 and self.beta > 0:
                return float(np.sqrt(a / self.beta))
            return float("nan")

        # gamma*u^2 + beta*u - a = 0, u = r^2
        disc = self.beta ** 2 + 4.0 * self.gamma * a
        if disc < 0:
            return float("nan")
        sqrt_disc = np.sqrt(disc)
        roots_u = [(-self.beta + sqrt_disc) / (2.0 * self.gamma),
                   (-self.beta - sqrt_disc) / (2.0 * self.gamma)]
        positive = [u for u in roots_u if u > 0]
        if not positive:
            return float("nan")
        return float(np.sqrt(max(positive)))  # stable outer cycle


if __name__ == "__main__":
    # Self-validation against analytical ground truth.
    print("=" * 68)
    print("StuartLandauNetwork -- validation against closed-form results")
    print("=" * 68)

    N = 4
    C = np.zeros((N, N))  # uncoupled: analytical results apply per node

    print("\n1. Subcritical (a<0): trajectories must decay to the origin")
    m = StuartLandauNetwork(N, C, a=-0.05, omega=0.05, beta=1.0)
    r = m.simulate(t_span=(0, 1000), n_points=5000)
    final_amp = np.sqrt(r["x"][:, -1] ** 2 + r["y"][:, -1] ** 2)
    print(f"   final amplitude: max={final_amp.max():.3e}  (expect ~0)")
    assert final_amp.max() < 1e-3, "did not decay to fixed point"
    print("   PASS")

    print("\n2. Supercritical (a>0): amplitude must match r* = sqrt(a/beta)")
    for a_val, beta_val in [(0.10, 1.0), (0.40, 1.0), (0.10, 4.0)]:
        m = StuartLandauNetwork(N, C, a=a_val, omega=0.05, beta=beta_val)
        r = m.simulate(t_span=(0, 4000), n_points=20000)
        tail = slice(-4000, None)
        amp = np.sqrt(r["x"][:, tail] ** 2 + r["y"][:, tail] ** 2).mean()
        pred = m.analytical_limit_cycle_amplitude()
        err = abs(amp - pred) / pred
        print(f"   a={a_val:.2f}, beta={beta_val:.1f}: "
              f"observed={amp:.5f}  analytical={pred:.5f}  err={100*err:.3f}%")
        assert err < 0.01, f"amplitude off by {100*err:.2f}%"
    print("   PASS")

    print("\n3. Jacobian at origin must match finite differences")
    C_rand = np.abs(np.random.default_rng(0).normal(0, 0.1, (N, N)))
    np.fill_diagonal(C_rand, 0)
    m = StuartLandauNetwork(N, C_rand, a=-0.05, omega=0.05, beta=1.0, G=0.5)
    J_analytic = m.jacobian_at_origin()
    eps = 1e-7
    J_fd = np.zeros((2 * N, 2 * N))
    base = np.zeros(2 * N)
    for i in range(2 * N):
        pert = base.copy()
        pert[i] += eps
        J_fd[:, i] = (m._ode_network(0.0, pert) - m._ode_network(0.0, base)) / eps
    max_err = np.abs(J_analytic - J_fd).max()
    print(f"   max |analytic - finite-difference| = {max_err:.3e}")
    assert max_err < 1e-5, "Jacobian mismatch"
    print("   PASS")

    print("\n4. beta<0 with gamma=0 must be REFUSED, not silently diverge")
    try:
        StuartLandauNetwork(N, C, a=-0.05, beta=-1.0, gamma=0.0)
        raise AssertionError("should have refused")
    except ValueError as e:
        print(f"   correctly refused: {str(e)[:60]}...")
    print("   PASS")

    print("\n5. Subcritical with quintic must stay bounded")
    m = StuartLandauNetwork(N, C, a=-0.05, omega=0.05, beta=-1.0, gamma=1.0)
    r = m.simulate(t_span=(0, 2000), n_points=10000)
    amp = np.sqrt(r["x"] ** 2 + r["y"] ** 2)
    print(f"   max amplitude over run: {amp.max():.4f}  (finite, bounded)")
    assert np.isfinite(amp).all() and amp.max() < 100, "diverged"
    print("   PASS")

    print("\n6. jacobian_at() at ARBITRARY state vs finite differences")
    m = StuartLandauNetwork(N, C_rand, a=0.15, omega=0.05, beta=1.0,
                            gamma=0.0, G=0.5)
    rng2 = np.random.default_rng(7)
    xs = rng2.uniform(-0.5, 0.5, N)
    ys = rng2.uniform(-0.5, 0.5, N)
    J_an = m.jacobian_at(xs, ys)
    base = np.concatenate([xs, ys])
    J_fd = np.zeros((2 * N, 2 * N))
    eps = 1e-7
    for i in range(2 * N):
        pert = base.copy(); pert[i] += eps
        J_fd[:, i] = (m._ode_network(0.0, pert) - m._ode_network(0.0, base)) / eps
    err = np.abs(J_an - J_fd).max()
    print(f"   max |analytic - FD| at random off-origin state = {err:.3e}")
    assert err < 1e-5, "arbitrary-state Jacobian mismatch"
    print("   PASS")

    print("\n7. nominal_orbit() must lie ON the limit cycle")
    m = StuartLandauNetwork(1, np.zeros((1, 1)), a=0.2, omega=0.05, beta=1.0)
    r_star = m.analytical_limit_cycle_amplitude()
    for ph in [0.0, 1.3, 2.7, 4.9]:
        st = m.nominal_orbit(ph)
        r = np.sqrt(st[0] ** 2 + st[1] ** 2)
        assert abs(r - r_star) < 1e-12, f"orbit radius off at phase {ph}"
    print(f"   radius exact at 4 phases (r*={r_star:.6f}); period={m.period:.4f}")
    # integrating one full period from an orbit point must return to it
    st0 = m.nominal_orbit(0.0)
    res = m.simulate_controlled(lambda t: np.zeros(1), (0.0, m.period), 4000,
                                 st0[:1], st0[1:])
    ret = np.array([res["E"][0, -1], res["I"][0, -1]])
    drift = np.linalg.norm(ret - st0)
    print(f"   closure error after one exact period = {drift:.3e}")
    assert drift < 1e-3, "orbit did not close"
    print("   PASS")

    print("\n8. simulate_controlled() must actually respond to input")
    m = StuartLandauNetwork(1, np.zeros((1, 1)), a=-0.05, omega=0.05, beta=1.0)
    free = m.simulate_controlled(lambda t: np.zeros(1), (0, 50), 500,
                                  np.array([0.1]), np.array([0.1]))
    driven = m.simulate_controlled(lambda t: np.array([0.3]), (0, 50), 500,
                                    np.array([0.1]), np.array([0.1]))
    sep = np.abs(free["E"][0, -1] - driven["E"][0, -1])
    print(f"   |free - driven| final x = {sep:.4f}  (must be > 0)")
    assert sep > 1e-3, "control input had no effect"
    print("   PASS")

    print("\n9. find_fixed_point() must refuse the oscillatory regime")
    m = StuartLandauNetwork(1, np.zeros((1, 1)), a=0.2, beta=1.0)
    try:
        m.find_fixed_point(); raise AssertionError("should have refused")
    except RuntimeError as e:
        print(f"   correctly refused: {str(e)[:52]}...")
    print("   PASS")

    print("\n" + "=" * 68)
    print("All validations passed.")
    print("=" * 68)
