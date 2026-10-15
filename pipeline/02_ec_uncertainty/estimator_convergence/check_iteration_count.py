"""
Checks how many FISTA iterations actually ran before the (buggy) break,
per subject -- tells us whether "ridge-only" is an accurate description
of what was computed for everyone, or whether it's heterogeneous
(some subjects near-pure-ridge, others near-true-converged-GraphNet).

Instrumented copy of the CURRENT (unpatched) function -- does not
modify connectivity.py, just traces it externally.
"""
import numpy as np
from nctsens.connectivity import graph_laplacian, ridge_effective_connectivity

def graphnet_traced(X, SC, lambda_ridge=1.0, lambda_graph=1.0, max_iter=500, tol=1e-6, lag=1):
    N, T = X.shape
    X_t, X_tp1 = X[:, :-lag], X[:, lag:]
    L = graph_laplacian(SC, normalised=True)
    XXT, YXT = X_t @ X_t.T, X_tp1 @ X_t.T
    Reg = lambda_ridge * np.eye(N) + lambda_graph * L
    A = ridge_effective_connectivity(X, alpha=lambda_ridge, lag=lag)
    A_prev = A.copy()
    t_k = 1.0
    step = 1.0 / (2*np.max(np.abs(np.linalg.eigvalsh(XXT))) + 2*np.max(np.abs(np.linalg.eigvalsh(Reg))) + 1e-8)
    for k in range(max_iter):
        Y = A + ((t_k-1)/(t_k+1)) * (A - A_prev)
        A_new = Y - step * (2*(Y@XXT-YXT) + 2*(Reg@Y.T).T)
        if np.linalg.norm(A_new-A,'fro')/(np.linalg.norm(A,'fro')+1e-12) < tol:
            return A, k  # k = iteration index where it broke (0 = immediate)
        A_prev = A.copy(); A = A_new
        t_k = (1+np.sqrt(1+4*t_k**2))/2
    return A, max_iter

# Fill in real subject data the same way as before, then run this
# across several real subjects per cohort:
#
# A_out, k_broke = graphnet_traced(X, SC, lambda_ridge=1.0, lambda_graph=1.0)
# print(f"subject {sub}: broke at iteration {k_broke}")
