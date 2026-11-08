#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
warp_shell_minmass.py
=====================

Minimum-mass search for a constant-velocity "warp shell" (Fuchs et al. 2024 class)
that satisfies the null, weak, strong and dominant energy conditions for ALL
observers.

Metric (comoving with the shell, static):

    ds^2 = -A(r)^2 dt^2 + P(r)^4 delta_ij (dx^i + B^i dt)(dx^j + B^j dt),
    B = (-beta * S(r), 0, 0)

  * P = 1 + Phi,  laplacian(Phi) = -2 pi s(r),  s >= 0   -> 3R = 16 pi s / P^5 >= 0
  * A = (1 - Psi/2)/(1 + Psi/2),  laplacian(Psi) = -4 pi u(r),  u >= 0
        -> laplacian(A) >= 0 (attractive gravity)
  * int s dV = int u dV = M  -> exterior is exactly isotropic Schwarzschild (vacuum)
  * S = 1 inside (flat passenger region carried at dx/dt = beta), S = 0 outside,
    monotone transition sigma = -S' >= 0 confined inside the shell.

s, u and sigma are non-negative B-spline combinations.  The Einstein tensor is
computed exactly by SymPy (lambdified), the stress tensor is projected onto the
Eulerian orthonormal tetrad, and the energy conditions are evaluated for every
observer:

  NEC, WEC, SEC : exact Lagrangian-dual lower bounds of the trust-region problems
                  min_{|n|=1} T(k_n,k_n)  and  min_{|v|<=1} T(u_v,u_v)
                  (any dual-feasible multiplier gives a CERTIFIED lower bound)
  DEC           : min over null k,l of T(k,l) = min_n [rho + q.n - |q + S n|]
                  (dense sphere sampling + local refinement)

Units: G = c = 1, lengths in metres (1 m of mass = 1.3466e27 kg).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
from scipy.interpolate import BSpline
from scipy.integrate import cumulative_simpson

KG_PER_M = 2.99792458e8 ** 2 / 6.67430e-11
M_JUP = 1.89813e27
FUCHS_M_KG = 4.49e27
FUCHS_M_GEOM = FUCHS_M_KG / KG_PER_M     # ~3.334 m
FIELD_NAMES = ("A", "P", "B1", "B2", "B3")
SECOND = ("xx", "xy", "xz", "yy", "yz", "zz")
JET_NAMES = [n for F in FIELD_NAMES for n in
             ([F] + [f"{F}_{c}" for c in "xyz"] + [f"{F}_{p}" for p in SECOND])]
G_COMPONENTS = [(a, b) for a in range(4) for b in range(a, 4)]


# =============================================================================
# 1. Exact Einstein tensor (static, general lapse/conformal factor/shift)
# =============================================================================
def build_einstein_static():
    import sympy as sp
    t, x, y, z = sp.symbols("t x y z", real=True)
    X = (t, x, y, z)
    A, P, B1, B2, B3 = [sp.Function(n, real=True)(*X) for n in FIELD_NAMES]
    B = (B1, B2, B3)
    P4 = P ** 4
    g = sp.zeros(4, 4)
    g[0, 0] = -A ** 2 + P4 * (B1 ** 2 + B2 ** 2 + B3 ** 2)
    for i in range(3):
        g[0, i + 1] = g[i + 1, 0] = P4 * B[i]
        g[i + 1, i + 1] = P4
    gi = sp.zeros(4, 4)
    gi[0, 0] = -1 / A ** 2
    for i in range(3):
        gi[0, i + 1] = gi[i + 1, 0] = B[i] / A ** 2
        for j in range(3):
            gi[i + 1, j + 1] = (1 / P4 if i == j else 0) - B[i] * B[j] / A ** 2
    if sp.simplify(g * gi - sp.eye(4)) != sp.zeros(4, 4):
        raise RuntimeError("inverse metric check failed")
    dg = [[[sp.diff(g[a, b], X[c]) for c in range(4)] for b in range(4)] for a in range(4)]
    Gam = [[[sp.expand(sum(gi[a, d] * (dg[d][b][c] + dg[d][c][b] - dg[b][c][d])
                           for d in range(4)) / 2) for c in range(4)] for b in range(4)] for a in range(4)]
    Ric = sp.zeros(4, 4)
    for b in range(4):
        for d in range(b, 4):
            e = 0
            for a in range(4):
                e += sp.diff(Gam[a][b][d], X[a]) - sp.diff(Gam[a][b][a], X[d])
                for k in range(4):
                    e += Gam[a][a][k] * Gam[k][b][d] - Gam[a][d][k] * Gam[k][b][a]
            Ric[b, d] = Ric[d, b] = e
    Rs = sum(gi[a, b] * Ric[a, b] for a in range(4) for b in range(4))
    Ein = Ric - g * Rs / 2
    jet = {nm: sp.Symbol(nm, real=True) for nm in JET_NAMES}
    fields = dict(zip(FIELD_NAMES, (A, P, B1, B2, B3)))
    rep = {F: jet[nm] for nm, F in fields.items()}
    outs = [Ein[a, b] for a, b in G_COMPONENTS]
    for e in outs:
        for d in e.atoms(sp.Derivative):
            cnt = dict(d.variable_count)
            if cnt.get(t, 0):
                rep[d] = sp.Integer(0)          # static in the comoving frame
            else:
                lab = "x" * cnt.get(x, 0) + "y" * cnt.get(y, 0) + "z" * cnt.get(z, 0)
                rep[d] = jet[f"{d.expr.func.__name__}_{lab}"]
    outs = [e.xreplace(rep) for e in outs]
    for e in outs:
        if e.free_symbols - set(jet.values()):
            raise RuntimeError("unresolved symbols")
    return [jet[n] for n in JET_NAMES], outs


def kernel_path() -> Path:
    import sympy as sp
    key = hashlib.sha256((inspect.getsource(build_einstein_static) + sp.__version__).encode()).hexdigest()[:16]
    cache = Path(__file__).resolve().parent / ".warp_kernel_cache"
    cache.mkdir(exist_ok=True)
    path = cache / f"einstein_static_{key}.py"
    if not path.exists():
        import sympy as sp
        t0 = time.perf_counter()
        syms, outs = build_einstein_static()
        fn = sp.lambdify(syms, outs, modules="numpy", cse=True)
        src = inspect.getsource(fn).replace("def _lambdifygenerated(", "def einstein_kernel(", 1)
        tmp = path.with_suffix(".tmp")
        tmp.write_text("import numpy\nfrom numpy import *\n\n" + src)
        os.replace(tmp, path)
        print(f"[sympy] static Einstein kernel built in {time.perf_counter() - t0:.1f} s")
    return path


def load_kernel():
    spec = importlib.util.spec_from_file_location("einstein_static", str(kernel_path()))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.einstein_kernel


# =============================================================================
# 2. Radial profiles from non-negative sources
# =============================================================================
def bspline_basis(a: float, b: float, n: int, k: int = 3):
    """n interior B-spline basis functions on [a,b] that vanish (with k-1 derivatives) at a, b."""
    n_tot = n + 2
    inner = np.linspace(a, b, n_tot - k + 1)
    t = np.concatenate([[a] * k, inner, [b] * k])
    basis = []
    for j in range(1, n_tot - 1):          # drop the two end functions -> vanish at a and b
        c = np.zeros(n_tot)
        c[j] = 1.0
        basis.append(BSpline(t, c, k, extrapolate=False))
    return basis


def eval_combo(basis, coef, r, nu=0):
    out = np.zeros_like(r)
    for bf, c in zip(basis, coef):
        if c != 0.0:
            f = bf if nu == 0 else bf.derivative(nu)
            val = f(r)
            out += c * np.nan_to_num(val, nan=0.0)
    return out


@dataclass
class ShellDesign:
    M: float                 # ADM mass [m]
    beta: float              # interior shift (passenger speed, units of c)
    r1: float                # isotropic inner radius of the shell support
    r2: float                # isotropic outer radius
    a: np.ndarray            # s-source coefficients  (>= 0)
    b: np.ndarray            # u-source coefficients  (>= 0)
    c: np.ndarray            # shift-transition coefficients (>= 0)
    shift_lo: float = 0.15   # shift transition support as fractions of [r1, r2]
    shift_hi: float = 0.85


class Profiles:
    """Radial functions A, P, S and their first two derivatives on a fine grid."""

    def __init__(self, d: ShellDesign, n_fine: int = 24001, normalize: bool = True):
        self.d = d
        r_hi = d.r2 * 1.6
        self.r = np.linspace(1e-6, r_hi, n_fine)
        r = self.r
        bs = bspline_basis(d.r1, d.r2, len(d.a))
        bu = bspline_basis(d.r1, d.r2, len(d.b))
        s = eval_combo(bs, d.a, r)
        u = eval_combo(bu, d.b, r)
        ms = 4 * np.pi * cumulative_simpson(s * r ** 2, x=r, initial=0.0)
        mu = 4 * np.pi * cumulative_simpson(u * r ** 2, x=r, initial=0.0)
        if normalize:
            s *= d.M / ms[-1]
            ms *= d.M / ms[-1]
            u *= d.M / mu[-1]
            mu *= d.M / mu[-1]
        self.s, self.u = s, u
        self.Ms, self.Mu = float(ms[-1]), float(mu[-1])
        # Phi: lap = -2 pi s ;  Phi' = -ms/(2 r^2) ; Phi'' = -2 pi s - 2 Phi'/r
        d1 = -ms / (2 * r ** 2)
        d2 = -2 * np.pi * s - 2 * d1 / r
        # Phi(r) = Phi(r_end) - int_r^{r_end} Phi' dr'   (integrated in u = -r so x is increasing)
        tail = self.Ms / (2 * r[-1])
        Phi = tail - cumulative_simpson(d1[::-1], x=-r[::-1], initial=0.0)[::-1]
        self.P, self.P1, self.P2 = 1 + Phi, d1, d2
        e1 = -mu / r ** 2
        e2 = -4 * np.pi * u - 2 * e1 / r
        Psi = self.Mu / r[-1] - cumulative_simpson(e1[::-1], x=-r[::-1], initial=0.0)[::-1]
        h = 1 + Psi / 2
        self.A = (1 - Psi / 2) / h
        self.A1 = -e1 / h ** 2
        self.A2 = -e2 / h ** 2 + e1 ** 2 / h ** 3
        # shift
        lo = d.r1 + d.shift_lo * (d.r2 - d.r1)
        hi = d.r1 + d.shift_hi * (d.r2 - d.r1)
        bc = bspline_basis(lo, hi, len(d.c))
        sig = eval_combo(bc, d.c, r)
        dsig = eval_combo(bc, d.c, r, nu=1)
        norm = cumulative_simpson(sig, x=r, initial=0.0)[-1]
        sig, dsig = sig / norm, dsig / norm
        C = cumulative_simpson(sig, x=r, initial=0.0)
        self.S, self.S1, self.S2 = 1 - C, -sig, -dsig

    def areal(self, idx):
        return self.P[idx] ** 2 * self.r[idx]


def radial_jets(F, F1, F2, x, y, z, r):
    """Cartesian value/gradient/Hessian of a radial function."""
    xs = (x, y, z)
    out = [F] + [F1 * c / r for c in xs]
    for p in SECOND:
        i, j = "xyz".index(p[0]), "xyz".index(p[1])
        val = F2 * xs[i] * xs[j] / r ** 2 + F1 * ((1.0 if i == j else 0.0) / r - xs[i] * xs[j] / r ** 3)
        out.append(val)
    return out


# =============================================================================
# 3. Stress tensor in the Eulerian orthonormal frame
# =============================================================================
def frame_stress(kernel, pr: Profiles, idx, theta):
    """Return rho (N,), q (N,3), S (N,3,3) at points (r[idx], theta)."""
    R = pr.r[idx][:, None] * np.ones_like(theta)[None, :]
    TH = np.ones_like(pr.r[idx])[:, None] * theta[None, :]
    x, y, z = R * np.cos(TH), R * np.sin(TH), np.zeros_like(R)
    r = R
    beta = pr.d.beta
    def pick(arr):
        return arr[idx][:, None] * np.ones_like(theta)[None, :]
    jA = radial_jets(pick(pr.A), pick(pr.A1), pick(pr.A2), x, y, z, r)
    jP = radial_jets(pick(pr.P), pick(pr.P1), pick(pr.P2), x, y, z, r)
    jS = radial_jets(pick(pr.S), pick(pr.S1), pick(pr.S2), x, y, z, r)
    jB1 = [-beta * v for v in jS]
    zero = [np.zeros_like(R)] * 10
    jets = jA + jP + jB1 + zero + zero
    G = kernel(*jets)
    shape = R.shape
    Gm = np.zeros((4, 4) + shape)
    for (a, b), val in zip(G_COMPONENTS, G):
        Gm[a, b] = Gm[b, a] = np.broadcast_to(val, shape)
    T = Gm / (8 * np.pi)
    Av, Pv, B1v = jA[0], jP[0], jB1[0]
    n = np.zeros((4,) + shape)
    n[0] = 1 / Av
    n[1] = -B1v / Av
    E = np.zeros((4, 4) + shape)          # E[a, mu] tetrad components
    E[0] = n
    for i in range(3):
        E[i + 1, i + 1] = 1 / Pv ** 2
    Tf = np.einsum("am...,bn...,mn...->ab...", E, E, T)
    rho = Tf[0, 0].reshape(-1)
    q = np.stack([Tf[0, i].reshape(-1) for i in (1, 2, 3)], axis=-1)
    S = np.stack([np.stack([Tf[i, j].reshape(-1) for j in (1, 2, 3)], -1) for i in (1, 2, 3)], -2)
    return rho, q, S, R.reshape(-1), TH.reshape(-1)


# =============================================================================
# 4. Energy conditions for all observers
# =============================================================================
def _dual_bound(rho, c, s, lam_cap, iters=200, return_lam=False):
    """sup_{lam <= min(s0, lam_cap)} [rho + lam - sum c_i^2/(s_i - lam)]  (vectorised).

    Any feasible lam gives a certified LOWER bound on
        min_{|n| = 1} (rho + 2 c.n + n^T S n)          (lam_cap = +inf, sphere)
        min_{|n| <= 1} (...)                           (lam_cap = 0,    ball)
    """
    s0 = s[:, 0]
    ub = np.minimum(s0, lam_cap)
    cn = np.sqrt((c ** 2).sum(-1))
    lo = np.minimum(ub, s0 - cn) - 1e-300
    lo = np.minimum(lo, ub - 1e-12 * (np.abs(ub) + 1))
    hi = ub.copy()

    def dprime(lam):
        return 1 - (c ** 2 / (s - lam[:, None]) ** 2).sum(-1)

    # if derivative still positive at hi (hard case / interior), optimum is at hi
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        pos = dprime(mid) > 0
        lo = np.where(pos, mid, lo)
        hi = np.where(pos, hi, mid)
    lam = lo                                   # strictly feasible
    val = rho + lam - (c ** 2 / (s - lam[:, None])).sum(-1)
    if return_lam:
        return val, lam
    return val


def energy_conditions(rho, q, S, n_dirs=2000):
    """Return dict of minimum values over all observers for NEC, WEC, SEC, DEC."""
    w, V = np.linalg.eigh(S)
    c = np.einsum("nji,nj->ni", V, q)
    nec = _dual_bound(rho, c, w, np.inf)
    wec = _dual_bound(rho, c, w, 0.0)
    trace = -rho + np.trace(S, axis1=1, axis2=2)
    rho_s = rho + 0.5 * trace
    S_s = S - 0.5 * trace[:, None, None] * np.eye(3)[None]
    ws, Vs = np.linalg.eigh(S_s)
    cs = np.einsum("nji,nj->ni", Vs, q)
    sec = _dual_bound(rho_s, cs, ws, 0.0)
    # DEC: min_n [rho + q.n - |q + S n|]
    k = np.arange(n_dirs) + 0.5
    ph = np.arccos(1 - 2 * k / n_dirs)
    th = np.pi * (1 + 5 ** 0.5) * k
    dirs = np.stack([np.cos(th) * np.sin(ph), np.sin(th) * np.sin(ph), np.cos(ph)], -1)
    best = np.full(rho.shape, np.inf)
    best_n = np.zeros((len(rho), 3))
    chunk = 200
    for i0 in range(0, n_dirs, chunk):
        dn = dirs[i0:i0 + chunk]
        Sn = np.einsum("pij,dj->pdi", S, dn)
        val = rho[:, None] + q @ dn.T - np.linalg.norm(q[:, None, :] + Sn, axis=-1)
        j = val.argmin(1)
        v = val[np.arange(len(rho)), j]
        better = v < best
        best = np.where(better, v, best)
        best_n[better] = dn[j[better]]
    step = 0.05
    rng = np.random.default_rng(0)
    for _ in range(40):
        cand = best_n + step * rng.normal(size=best_n.shape)
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        v = rho + (q * cand).sum(-1) - np.linalg.norm(q + np.einsum("pij,pj->pi", S, cand), axis=-1)
        better = v < best
        best = np.where(better, v, best)
        best_n[better] = cand[better]
        step *= 0.9
    return {"NEC": nec, "WEC": wec, "SEC": sec, "DEC": best}


# =============================================================================
# 5. Evaluation of a design
# =============================================================================
@dataclass
class EvalSpec:
    n_r: int = 360
    n_theta: int = 25
    n_dirs: int = 600


def evaluate(kernel, d: ShellDesign, spec: EvalSpec, n_fine=24001, full=False):
    pr = Profiles(d, n_fine)
    lo, hi = d.r1 - 0.02 * (d.r2 - d.r1), d.r2 + 0.02 * (d.r2 - d.r1)
    cand = np.where((pr.r >= lo) & (pr.r <= hi))[0]
    idx = cand[np.linspace(0, len(cand) - 1, spec.n_r).astype(int)]
    theta = np.linspace(0, np.pi, spec.n_theta)
    parts = [frame_stress(kernel, pr, ch, theta) for ch in np.array_split(idx, max(1, len(idx) // 250))]
    rho, q, S, R, TH = (np.concatenate([p_[i] for p_ in parts]) for i in range(5))
    if not (np.all(np.isfinite(rho)) and np.all(np.isfinite(q)) and np.all(np.isfinite(S))
            and pr.A.min() > 0):
        bad = {k: -np.inf for k in ("NEC", "WEC", "SEC", "DEC")}
        return {"worst_rel": bad, "rho_max": float("nan"), "min_rel": -np.inf, "A_min": float(pr.A.min()),
                "P_in": float(pr.P[0]), "areal_inner": float("nan"), "areal_outer": float("nan")}
    ec = energy_conditions(rho, q, S, spec.n_dirs)
    scale = max(rho.max(), 1e-300)
    worst = {k: float(v.min() / scale) for k, v in ec.items()}
    out = {"worst_rel": worst, "rho_max": float(scale), "min_rel": min(worst.values()),
           "A_min": float(pr.A.min()), "P_in": float(pr.P[0]),
           "areal_inner": float(pr.P[0] ** 2 * d.r1),
           "areal_outer": float((1 + d.M / (2 * d.r2)) ** 2 * d.r2)}
    if full:
        out.update({"R": R, "TH": TH, "rho": rho, "q": q, "S": S, "ec": ec, "profiles": pr})
    return out


def violation(kernel, d, spec, tol=1e-9):
    pr = Profiles(d)
    lo, hi = d.r1 - 0.02 * (d.r2 - d.r1), d.r2 + 0.02 * (d.r2 - d.r1)
    cand = np.where((pr.r >= lo) & (pr.r <= hi))[0]
    idx = cand[np.linspace(0, len(cand) - 1, spec.n_r).astype(int)]
    theta = np.linspace(0, np.pi, spec.n_theta)
    rho, q, S, *_ = frame_stress(kernel, pr, idx, theta)
    ec = energy_conditions(rho, q, S, spec.n_dirs)
    scale = max(rho.max(), 1e-300)
    v = 0.0
    for val in ec.values():
        neg = np.minimum(val / scale + tol, 0.0)
        v += float((neg ** 2).sum())
    return v, min(float(val.min() / scale) for val in ec.values())


def iso_radii(M, R1_areal, R2_areal, P_in_guess=1.0):
    """Isotropic radii whose areal radii are R2 (outer, exact) and ~R1 (inner, using P_in)."""
    from scipy.optimize import brentq
    r2 = brentq(lambda r: (1 + M / (2 * r)) ** 2 * r - R2_areal, M / 2, R2_areal)  # areal radius is monotone for r > M/2
    r1 = R1_areal / P_in_guess ** 2
    return r1, r2


# =============================================================================
# 6. Constructive minimum-mass shell
# =============================================================================
def design_for(M, a, b, c, beta, lo, hi, R1=10.0, R2=20.0, iters=5):
    """ShellDesign whose areal radii are R1 (passenger boundary) and R2 (outer edge)."""
    P_in = 1.0
    d = None
    for _ in range(iters):
        r1, r2 = iso_radii(M, R1, R2, P_in)
        d = ShellDesign(M=M, beta=beta, r1=r1, r2=r2, a=np.asarray(a, float), b=np.asarray(b, float),
                        c=np.asarray(c, float), shift_lo=lo, shift_hi=hi)
        P_in = Profiles(d, 6001).P[0]
    return d


def momentum_envelope(kernel, d: ShellDesign, n_r=1600, n_theta=37):
    """max over polar angle of |q| (Eulerian momentum density) on radii inside the shell."""
    pr = Profiles(d)
    cand = np.where((pr.r > d.r1) & (pr.r < d.r2))[0]
    idx = cand[np.linspace(0, len(cand) - 1, n_r).astype(int)]
    th = np.linspace(0, np.pi, n_theta)
    rho, q, S, R, TH = frame_stress(kernel, pr, idx, th)
    qm = np.linalg.norm(q, axis=1).reshape(len(idx), n_theta).max(1)
    return pr, idx, qm


def estimated_mass(kernel, d, kappa=2.0):
    pr, idx, qm = momentum_envelope(kernel, d)
    rr = pr.r[idx]
    return float(np.trapezoid(4 * np.pi * rr ** 2 * kappa * qm * pr.P[idx] ** 5, rr))


def envelope_source(r_samples, target, r1, r2, n_basis=48):
    """Least-mass non-negative B-spline s(r) on [r1,r2] with s(r_k) >= target_k (LP)."""
    from scipy.optimize import linprog
    basis = bspline_basis(r1, r2, n_basis)
    Bm = np.stack([np.nan_to_num(bf(r_samples), nan=0.0) for bf in basis], 1)
    rq = np.linspace(r1, r2, 4001)
    w = np.array([np.trapezoid(4 * np.pi * rq ** 2 * np.nan_to_num(bf(rq), nan=0.0), rq) for bf in basis])
    res = linprog(w, A_ub=-Bm, b_ub=-target, bounds=[(0, None)] * n_basis, method="highs")
    if not res.success:
        raise RuntimeError(res.message)
    return res.x


def construct_min_shell(kernel, beta, c, lo, hi, kappa0=2.0, R1=10.0, R2=20.0,
                        n_basis=48, max_rounds=40, spec=None, verbose=True, boost_step=0.12):
    """Iteratively build the lightest density covering the shift's momentum, then
    raise it locally wherever any energy condition fails."""
    spec = spec or EvalSpec(n_r=2000, n_theta=31, n_dirs=700)
    M = 1.0
    a = np.ones(n_basis)
    boost = None
    history = []
    for rnd in range(max_rounds):
        d = design_for(M, a, a, c, beta, lo, hi, R1, R2)
        pr, idx, qm = momentum_envelope(kernel, d)
        rr = pr.r[idx]
        if boost is None:
            boost = np.full_like(rr, kappa0)
        target = boost * qm * pr.P[idx] ** 5
        # a small floor across the shift support keeps rho strictly positive there
        target = np.maximum(target, 1e-3 * target.max())
        a = envelope_source(rr, target, d.r1, d.r2, n_basis)
        M_new = float(np.trapezoid(4 * np.pi * pr.r ** 2 * eval_combo(bspline_basis(d.r1, d.r2, n_basis), a, pr.r), pr.r))
        a = a / M_new                       # shape only; Profiles renormalises to M
        M = M_new
        d = design_for(M, a, a, c, beta, lo, hi, R1, R2)
        out = evaluate(kernel, d, spec, full=True)
        worst = out["worst_rel"]
        history.append({"round": rnd, "M": M, **worst})
        if verbose:
            print(f"  round {rnd}: M = {M:.4f} m  worst rel EC " +
                  " ".join(f"{k}={v:+.1e}" for k, v in worst.items()))
        if min(worst.values()) >= -1e-9:
            return d, out, history
        # raise the target where violations occur (per radius)
        bad = np.zeros(len(rr))
        Rv = out["R"]
        viol = np.min(np.stack([out["ec"][k] for k in out["ec"]]), axis=0) / out["rho_max"]
        for k, rv in enumerate(rr):
            sel = np.abs(Rv - rv) < 0.5 * (rr[1] - rr[0]) + (Rv.max() - Rv.min()) / spec.n_r
            if np.any(sel):
                bad[k] = min(0.0, viol[sel].min())
        bump = np.convolve((bad < -1e-9).astype(float), np.ones(5) / 5, mode="same") > 0
        boost = boost * (1 + boost_step * bump)
    return d, out, history


# =============================================================================
# 7. Sequential linear programming on the exact Einstein tensor
# =============================================================================
def fib_sphere(n):
    k = np.arange(n) + 0.5
    ph = np.arccos(1 - 2 * k / n)
    th = np.pi * (1 + 5 ** 0.5) * k
    return np.stack([np.cos(th) * np.sin(ph), np.sin(th) * np.sin(ph), np.cos(ph)], -1)


def _frame_tensor(kernel, pr, idx, theta):
    rho, q, S, R, TH = frame_stress(kernel, pr, idx, theta)
    T = np.zeros((len(rho), 4, 4))
    T[:, 0, 0] = rho
    T[:, 0, 1:] = q
    T[:, 1:, 0] = q
    T[:, 1:, 1:] = S
    return T, R, TH


def _contract(T, k, l):
    """T_ab k^a l^b for per-point tensors T (N,4,4) and per-point vectors k,l (N,4)."""
    return np.einsum("nab,na,nb->n", T, k, l)


def worst_observers(T, n_dirs=1500):
    """Per-point worst observers for NEC, WEC, SEC, DEC of frame tensors T (N,4,4).

    Returns dict name -> (value (N,), k (N,4), l (N,4)); value = T(k,l) (or the SEC form)."""
    rho, q, S = T[:, 0, 0], T[:, 0, 1:], T[:, 1:, 1:]
    out = {}
    w, V = np.linalg.eigh(S)
    c = np.einsum("nji,nj->ni", V, q)

    def ball_vec(rho_, c_, w_, V_, cap):
        val, lam = _dual_bound(rho_, c_, w_, cap, return_lam=True)
        coef = -c_ / (w_ - lam[:, None])
        v = np.einsum("nij,nj->ni", V_, coef)
        nv = np.linalg.norm(v, axis=1, keepdims=True)
        v = np.where(nv > 1, v / np.maximum(nv, 1e-300), v)
        return val, v

    val, n = ball_vec(rho, c, w, V, np.inf)
    n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-300)
    k = np.concatenate([np.ones((len(rho), 1)), n], 1)
    out["NEC"] = (val, k, k)
    val, v = ball_vec(rho, c, w, V, 0.0)
    v = v * np.minimum(1.0, 0.999 / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-300))
    u = np.concatenate([np.ones((len(rho), 1)), v], 1) / np.sqrt(1 - (v ** 2).sum(1, keepdims=True))
    out["WEC"] = (val, u, u)
    tr = -rho + np.trace(S, axis1=1, axis2=2)
    rs, Ss = rho + 0.5 * tr, S - 0.5 * tr[:, None, None] * np.eye(3)
    ws, Vs = np.linalg.eigh(Ss)
    cs = np.einsum("nji,nj->ni", Vs, q)
    val, v = ball_vec(rs, cs, ws, Vs, 0.0)
    v = v * np.minimum(1.0, 0.999 / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-300))
    u = np.concatenate([np.ones((len(rho), 1)), v], 1) / np.sqrt(1 - (v ** 2).sum(1, keepdims=True))
    out["SEC"] = (val, u, u)
    dirs = fib_sphere(n_dirs)
    best = np.full(len(rho), np.inf)
    bn = np.zeros((len(rho), 3))
    for i0 in range(0, n_dirs, 250):
        dn = dirs[i0:i0 + 250]
        val = rho[:, None] + q @ dn.T - np.linalg.norm(q[:, None, :] + np.einsum("pij,dj->pdi", S, dn), axis=-1)
        j = val.argmin(1)
        vv = val[np.arange(len(rho)), j]
        bt = vv < best
        best = np.where(bt, vv, best)
        bn[bt] = dn[j[bt]]
    m = -(q + np.einsum("pij,pj->pi", S, bn))
    m /= np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-300)
    one = np.ones((len(rho), 1))
    out["DEC"] = (best, np.concatenate([one, bn], 1), np.concatenate([one, m], 1))
    return out


ETA = np.diag([-1.0, 1.0, 1.0, 1.0])


class SLPShell:
    """Minimise ADM mass over non-negative sources (s, u) at fixed shift, subject to
    NEC/WEC/SEC/DEC for ALL observers, by sequential linear programming with an inner
    cutting-plane loop on the exact worst observers of the linearised tensor."""

    def __init__(self, kernel, beta, c, lo, hi, n_basis=48, R1=10.0, R2=20.0,
                 n_r=220, n_theta=19):
        self.K, self.beta, self.c, self.lo, self.hi = kernel, beta, np.asarray(c, float), lo, hi
        self.nb, self.R1, self.R2 = n_basis, R1, R2
        self.n_r, self.n_theta = n_r, n_theta
        self.theta = np.linspace(0, np.pi, n_theta)
        self._Pin = 1.0

    def _design(self, M):
        r1, r2 = iso_radii(M, self.R1, self.R2, self._Pin)
        return ShellDesign(M=M, beta=self.beta, r1=r1, r2=r2, a=np.ones(self.nb), b=np.ones(self.nb),
                           c=self.c, shift_lo=self.lo, shift_hi=self.hi)

    def _points(self, d, pr):
        lo, hi = d.r1 - 0.01 * (d.r2 - d.r1), d.r2 + 0.01 * (d.r2 - d.r1)
        cand = np.where((pr.r >= lo) & (pr.r <= hi))[0]
        return cand[np.linspace(0, len(cand) - 1, self.n_r).astype(int)]

    def tensors(self, a, b, d, idx):
        pr = Profiles(ShellDesign(**{**d.__dict__, "a": a, "b": b}), normalize=False, n_fine=16001)
        T, R, TH = _frame_tensor(self.K, pr, idx, self.theta)
        return T, pr

    def mass_weights(self, d):
        rq = np.linspace(d.r1, d.r2, 8001)
        return np.array([np.trapezoid(4 * np.pi * rq ** 2 * np.nan_to_num(bf(rq), nan=0.0), rq)
                         for bf in bspline_basis(d.r1, d.r2, self.nb)])

    @staticmethod
    def _row(T0, J, pi, k, l, sec):
        Tp, Jp = T0[pi], J[:, pi]
        base = np.einsum("nab,na,nb->n", Tp, k, l)
        grad = np.einsum("vnab,na,nb->nv", Jp, k, l)
        if sec:
            kl = np.einsum("na,ab,nb->n", k, ETA, l)
            tr0 = -Tp[:, 0, 0] + np.trace(Tp[:, 1:, 1:], axis1=1, axis2=2)
            trJ = -Jp[:, :, 0, 0] + np.trace(Jp[:, :, 1:, 1:], axis1=2, axis2=3)
            base = base - 0.5 * tr0 * kl
            grad = grad - 0.5 * trJ.T * kl[:, None]
        return base, grad

    def solve(self, a0, b0, outer=8, trust=0.3, margin=2e-3, verbose=True, spec=None):
        from scipy.optimize import linprog
        spec = spec or EvalSpec(n_r=420, n_theta=31, n_dirs=800)
        a, b = np.asarray(a0, float).copy(), np.asarray(b0, float).copy()
        log, best = [], None
        for it in range(outer):
            d = self._design(1.0)
            for _ in range(3):                      # radii consistent with the current mass
                pr = Profiles(ShellDesign(**{**d.__dict__, "a": a, "b": b}), normalize=False, n_fine=8001)
                self._Pin = pr.P[0]
                d = self._design(max(pr.Ms, 1e-4))
            w = self.mass_weights(d)
            idx = self._points(d, pr)
            T0, _ = self.tensors(a, b, d, idx)
            N = T0.shape[0]
            x0 = np.concatenate([a, b])
            nv = 2 * self.nb
            J = np.zeros((nv, N, 4, 4))
            h = 1e-4 * x0.max()
            for v in range(nv):
                xp = x0.copy()
                xp[v] += h
                Tp, _ = self.tensors(xp[:self.nb], xp[self.nb:], d, idx)
                J[v] = (Tp - T0) / h
            scale = T0[:, 0, 0].max()
            rows_A, rows_b = [], []

            e0 = np.tile([1.0, 0, 0, 0], (N, 1))
            rho_base, rho_grad = self._row(T0, J, allp_ := np.arange(N), e0, e0, False)

            def add(pi, k, l, sec=False):
                base, grad = self._row(T0, J, pi, k, l, sec)
                # strict interior relative to the local density:
                #   T(k,l) - margin * k0 l0 * rho >= 0   (homogeneous, so vacuum stays feasible)
                f = margin * k[:, 0] * l[:, 0]
                base = base - f * rho_base[pi]
                grad = grad - f[:, None] * rho_grad[pi]
                rows_A.append(-grad / scale)
                rows_b.append(base / scale)

            # seed constraints: Eulerian energy and NEC along +-x, +-y, +-z at every point
            allp = np.arange(N)
            add(allp, np.tile([1.0, 0, 0, 0], (N, 1)), np.tile([1.0, 0, 0, 0], (N, 1)))
            for e in np.vstack([np.eye(3), -np.eye(3)]):
                k = np.tile(np.concatenate([[1.0], e]), (N, 1))
                add(allp, k, k)
            lb = np.maximum(-x0, -trust * np.maximum(x0, 0.05 * x0.max()))
            ub = trust * np.maximum(x0, 0.05 * x0.max())
            A_eq = np.concatenate([w, -w])[None, :]
            b_eq = np.array([-(w @ a - w @ b)])
            cvec = np.concatenate([w, np.zeros(self.nb)])
            res = None
            for inner in range(25):
                res = linprog(cvec, A_ub=np.vstack(rows_A), b_ub=np.concatenate(rows_b),
                              A_eq=A_eq, b_eq=b_eq, bounds=list(zip(lb, ub)), method="highs")
                if not res.success:
                    break
                Tl = T0 + np.einsum("v,vnab->nab", res.x, J)
                wo = worst_observers(Tl)
                n_cut = 0
                for name, (val, k, l) in wo.items():
                    bad = np.where(val < 0.5 * margin * Tl[:, 0, 0] * k[:, 0] * l[:, 0] - 1e-12 * scale)[0]
                    if len(bad):
                        kk = k[bad] if name != "NEC" else k[bad]
                        add(bad, kk, l[bad], sec=(name == "SEC"))
                        n_cut += len(bad)
                if n_cut == 0:
                    break
            if res is None or not res.success:
                if verbose:
                    print(f"  SLP {it}: LP failed ({res.message if res else ''}); trust -> {trust * 0.5:.3f}")
                trust *= 0.5
                continue
            x_new = np.maximum(x0 + res.x, 0.0)
            a_new, b_new = x_new[:self.nb], x_new[self.nb:]
            M_new = float(w @ a_new)
            d_chk = ShellDesign(**{**d.__dict__, "a": a_new, "b": b_new, "M": M_new})
            out = evaluate(self.K, d_chk, spec, full=False)
            worst = out["worst_rel"]
            log.append({"iter": it, "M": M_new, "inner_rounds": inner + 1, **worst})
            ok = min(worst.values()) >= -1e-9
            if verbose:
                print(f"  SLP {it}: M = {M_new:.4f} m  inner={inner + 1}  exact worst rel EC " +
                      " ".join(f"{k}={v:+.1e}" for k, v in worst.items()) + ("  FEASIBLE" if ok else ""))
            if ok and (best is None or M_new < best["M"]):
                best = {"M": M_new, "a": a_new.copy(), "b": b_new.copy(), "design": d_chk, "worst": worst}
            if ok:
                a, b = a_new, b_new
            elif np.isfinite(min(worst.values())) and min(worst.values()) > -0.05:
                # mildly infeasible: accept the step (the cuts will repair it) but be cautious
                trust *= 0.5
                margin *= 2.0
                a, b = a_new, b_new
            else:
                # degenerate / strongly infeasible step: reject it
                trust *= 0.5
                margin *= 2.0
        return best, log


# =============================================================================
# 8. Leading-order lower bound (linear program over ALL shift profiles)
# =============================================================================
def mass_lower_bound(beta, r1, r2, n=800, n_theta=19, n_e=16):
    """min over shift profiles S (S=1 for r<=r1, S=0 for r>=r2) of
        int 4 pi r^2 max_theta |j(r,theta)| dr,
    with the weak-field momentum density of B = -beta S(r) e_x
        j = (1/16pi) (lap B - grad div B).
    The Eulerian dominant energy condition (rho >= |j|) with a spherical density makes this
    a lower bound on the ADM mass, to leading order in beta and M/R.  Every node, including
    both edges, is constrained (S is continued flat beyond the shell)."""
    import scipy.sparse as sparse
    from scipy.optimize import linprog
    rk = np.linspace(r1, r2, n + 1)
    h = rk[1] - rk[0]
    m = n - 1
    thetas = np.linspace(0, np.pi, n_theta)
    ang = np.linspace(0, 2 * np.pi, n_e, endpoint=False)
    es = np.stack([np.cos(ang), np.sin(ang)], 1)
    rows, cols, vals, rhs = [], [], [], []
    row = 0
    for k in range(n + 1):
        rr = rk[k]

        def coef(kind):
            cd, const = {}, 0.0
            stencil = [(1, 1 / (2 * h)), (-1, -1 / (2 * h))] if kind == 1 else \
                      [(1, 1 / h ** 2), (0, -2 / h ** 2), (-1, 1 / h ** 2)]
            for off, wv in stencil:
                kk = k + off
                if kk <= 0:
                    const += wv
                elif kk < n:
                    cd[kk - 1] = cd.get(kk - 1, 0.0) + wv
            return cd, const

        d1, c1 = coef(1)
        d2, c2 = coef(2)
        for t_ in thetas:
            cs, sn = np.cos(t_), np.sin(t_)
            A1 = np.array([-(1 + cs ** 2) / rr, -cs * sn / rr]) * beta / (16 * np.pi)
            A2 = np.array([-sn ** 2, cs * sn]) * beta / (16 * np.pi)
            for e in es:
                g1, g2 = e @ A1, e @ A2
                cd = {}
                for kk, wv in d1.items():
                    cd[kk] = cd.get(kk, 0.0) + g1 * wv
                for kk, wv in d2.items():
                    cd[kk] = cd.get(kk, 0.0) + g2 * wv
                for kk, wv in cd.items():
                    rows.append(row); cols.append(kk); vals.append(wv)
                rows.append(row); cols.append(m + k); vals.append(-1.0)
                rhs.append(-(g1 * c1 + g2 * c2))
                row += 1
    A = sparse.csr_matrix((vals, (rows, cols)), shape=(row, m + n + 1))
    wq = 4 * np.pi * rk ** 2 * h
    wq[0] *= 0.5
    wq[-1] *= 0.5
    res = linprog(np.concatenate([np.zeros(m), wq]), A_ub=A, b_ub=np.array(rhs),
                  bounds=[(None, None)] * m + [(0, None)] * (n + 1), method="highs")
    if not res.success:
        raise RuntimeError(res.message)
    S = np.concatenate([[1.0], res.x[:m], [0.0]])
    return float(res.fun), rk, S


def shift_from_profile(rk, S, n_c=60, ramp=0.6):
    """Smooth, monotone B-spline shift transition approximating S(r) with cosine ramps at the ends."""
    from scipy.optimize import nnls
    sig = -np.gradient(S, rk)
    lo_r, hi_r = rk[0], rk[-1]

    def rp(u):
        return np.where(u <= 0, 0.0, np.where(u >= ramp, 1.0, 0.5 - 0.5 * np.cos(np.pi * np.clip(u / ramp, 0, 1))))

    target = np.maximum(sig, 0) * rp(rk - lo_r) * rp(hi_r - rk)
    basis = bspline_basis(lo_r, hi_r, n_c)
    Bm = np.stack([np.nan_to_num(b(rk), nan=0.0) for b in basis], 1)
    c, _ = nnls(Bm, target)
    return c, lo_r, hi_r


def stress_test(kernel, d, n_r=4000, n_theta=91, n_dirs=3000, n_fine=48001):
    """Worst relative energy-condition value on two interleaved angular grids."""
    pr = Profiles(d, n_fine)
    lo_, hi_ = d.r1 - 0.02 * (d.r2 - d.r1), d.r2 + 0.02 * (d.r2 - d.r1)
    cand = np.where((pr.r >= lo_) & (pr.r <= hi_))[0]
    idx = cand[np.linspace(0, len(cand) - 1, n_r).astype(int)]
    th1 = np.linspace(0, np.pi, n_theta)
    th2 = th1[:-1] + 0.5 * (th1[1] - th1[0])
    worst = {k: np.inf for k in ("NEC", "WEC", "SEC", "DEC")}
    rmax, tight = 0.0, np.inf
    for th in (th1, th2):
        for chunk in np.array_split(idx, max(1, len(idx) // 60)):
            rho, q, S, R, TH = frame_stress(kernel, pr, chunk, th)
            ec = energy_conditions(rho, q, S, n_dirs)
            rmax = max(rmax, rho.max())
            inside = rho > 1e-6 * rmax
            for k, v in ec.items():
                worst[k] = min(worst[k], float(v.min()))
                if inside.any():
                    tight = min(tight, float((v[inside] / rho[inside]).min()))
    return {k: v / rmax for k, v in worst.items()}, tight


def plot_result(d, kernel, bound_M, beta_points, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pr = Profiles(d)
    idx = np.where((pr.r > d.r1 - 0.3) & (pr.r < d.r2 + 0.3))[0][::6]
    th = np.linspace(0, np.pi, 37)
    rho, q, S, R, TH = frame_stress(kernel, pr, idx, th)
    nr = len(idx)
    qm = np.linalg.norm(q, axis=1).reshape(nr, len(th)).max(1)
    rhor = rho.reshape(nr, len(th)).mean(1)
    areal = pr.P[idx] ** 2 * pr.r[idx]
    INK, INK2, GRID = "#0b0b0b", "#52514e", "#dddcd8"
    BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.8), facecolor="#fcfcfb")
    ax = axs[0]
    ax.plot(areal, rhor * 1e6, color=BLUE, lw=2, label="energy density ρ_E")
    ax.plot(areal, 2 * qm * 1e6, color=ORANGE, lw=2, ls="--", label="2 × max|momentum density|")
    ax.set_xlabel("areal radius [m]", color=INK2)
    ax.set_ylabel("[10⁻⁶ m⁻²]  (G = c = 1)", color=INK2)
    ax.set_title("Density is shaped to the momentum it must carry", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    ax = axs[1]
    ax.plot(areal, pr.S[idx], color=BLUE, lw=2, label="shift profile S(r)")
    ax.plot(areal, (pr.A[idx] - pr.A[idx].min()) / (pr.A[idx].max() - pr.A[idx].min()), color=AQUA, lw=2,
            label="lapse A(r), rescaled 0–1")
    ax.set_xlabel("areal radius [m]", color=INK2)
    ax.set_title("Shift and lapse across the shell", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    ax = axs[2]
    bs = np.array([b for b, _ in beta_points])
    ms = np.array([m for _, m in beta_points]) * KG_PER_M / 1e27
    ax.plot(bs, ms, color=BLUE, lw=2, marker="o", ms=8, label="this work (verified)")
    ax.scatter([0.02], [FUCHS_M_KG / 1e27], color=ORANGE, s=70, zorder=3, label="Fuchs et al. 2024")
    bb = np.linspace(0, bs.max(), 50)
    ax.plot(bb, bound_M / 0.02 * bb * KG_PER_M / 1e27, color=INK2, lw=1.5, ls=":",
            label="leading-order lower bound")
    ax.set_xlabel("passenger speed β = v/c", color=INK2)
    ax.set_ylabel("shell ADM mass [10²⁷ kg]", color=INK2)
    ax.set_title("Mass needed vs speed (R = 10–20 m)", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    for a in axs:
        a.grid(True, color=GRID, lw=0.5)
        a.tick_params(colors=INK2)
        for sp_ in a.spines.values():
            sp_.set_color(GRID)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Minimum-mass physical warp shell (reproduces the reported result)")
    ap.add_argument("--beta", type=float, default=0.02)
    ap.add_argument("--extra-betas", type=float, nargs="*", default=[0.06])
    ap.add_argument("--buffer", type=float, default=0.05, help="shift-free buffer at each shell edge (fraction)")
    ap.add_argument("--outdir", default="warp_shell_output")
    a = ap.parse_args()
    out = Path(a.outdir)
    out.mkdir(exist_ok=True)
    K = load_kernel()
    t0 = time.perf_counter()
    # geometry for the bound and the shift: isotropic radii of a reference shell
    dref = design_for(0.7, np.ones(64), np.ones(64), np.ones(9), a.beta, a.buffer, 1 - a.buffer)
    lo_r = dref.r1 + a.buffer * (dref.r2 - dref.r1)
    hi_r = dref.r2 - a.buffer * (dref.r2 - dref.r1)
    Mb_full, _, _ = mass_lower_bound(a.beta, dref.r1, dref.r2)
    Mb, rk, Sopt = mass_lower_bound(a.beta, lo_r, hi_r)
    print(f"[bound] leading-order lower bound: M >= {Mb_full:.4f} m ({Mb_full * KG_PER_M:.3e} kg) any shift; "
          f"{Mb:.4f} m with the {a.buffer:.0%} buffer")
    c, lo_r, hi_r = shift_from_profile(rk, Sopt)
    lo = (lo_r - dref.r1) / (dref.r2 - dref.r1)
    hi = (hi_r - dref.r1) / (dref.r2 - dref.r1)
    results = {"beta": a.beta, "bound_any_shift_m": Mb_full, "bound_buffered_m": Mb,
               "fuchs_mass_m": FUCHS_M_GEOM, "designs": []}
    beta_points = []
    best = None
    for beta in [a.beta] + list(a.extra_betas):
        d, o, h = construct_min_shell(K, beta, c, lo, hi, n_basis=96, boost_step=0.04, max_rounds=200, verbose=False)
        worst, tight = stress_test(K, d)
        ok = min(worst.values()) >= -1e-9 and tight > 0
        pr = Profiles(d)
        rec = {"beta": beta, "M_m": d.M, "M_kg": d.M * KG_PER_M, "M_jupiter": d.M * KG_PER_M / M_JUP,
               "fuchs_over_this": FUCHS_M_GEOM / d.M, "stress_test_worst_rel": worst,
               "tightest_margin_over_local_rho": tight, "PASS": bool(ok),
               "areal_inner_m": float(pr.P[0] ** 2 * d.r1), "areal_outer_m": float((1 + d.M / (2 * d.r2)) ** 2 * d.r2),
               "lapse_min": float(pr.A.min()), "compactness_2M_over_R2": 2 * d.M / 20.0,
               "design": {"r1_iso": d.r1, "r2_iso": d.r2, "shift_lo": d.shift_lo, "shift_hi": d.shift_hi,
                          "s_coeffs": d.a.tolist(), "u_coeffs": d.b.tolist(), "shift_coeffs": d.c.tolist()}}
        results["designs"].append(rec)
        print(f"[shell] beta={beta}: M = {d.M:.4f} m = {d.M * KG_PER_M:.3e} kg ({rec['M_jupiter']:.3f} M_J); "
              f"published/this = {rec['fuchs_over_this']:.2f}; stress test "
              f"{'PASS' if ok else 'FAIL'} (worst {min(worst.values()):+.1e}, tightest margin {tight:.2e} rho)")
        if ok:
            beta_points.append((beta, d.M))
        if beta == a.beta:
            best = d
    if best is not None and beta_points:
        plot_result(best, K, Mb_full, beta_points, out / "warp_shell_result.png")
        print(f"[plot] {out / 'warp_shell_result.png'}")
    results["runtime_s"] = time.perf_counter() - t0
    (out / "warp_shell_results.json").write_text(json.dumps(results, indent=2))
    print(f"[out] {out / 'warp_shell_results.json'}  ({results['runtime_s']:.0f} s)")


if __name__ == "__main__":
    main()
