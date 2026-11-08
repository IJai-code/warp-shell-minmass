# Formal Verification Architecture (Conceptual)

This document outlines how a proof assistant such as Lean 4 with Mathlib could formally verify the results in this repository. It is a design, not an implementation: no Lean code has been written or checked. The snippets below are pseudocode in Lean's style.

## 1. What Lean can realistically check

Lean should not be asked to do the numerical work. Mathlib has no ready-made library for Lorentzian curvature, and re-running the optimisation inside a proof assistant would be impractical. The workable pattern is **certificate checking**:

1. The Python pipeline searches for a design and, alongside it, emits **certificates**: exact rational numbers that witness each claim.
2. Lean checks the certificates using exact rational or interval arithmetic.
3. The search can be wrong, slow or heuristic. Only the checker has to be trusted.

This is how the result moves from "passed a stress test" to "proved, given clearly stated assumptions".

## 2. Trusted base: definitions, assumptions, theorems

The original request suggested an axiom named `Fuchs_Metric_Parameters`. Parameters are better stated as **definitions**: they are data, and nothing about them needs to be assumed. Axioms should be kept to statements that cannot yet be proved inside Lean, and each one should be listed and reviewed.

```
-- DEFINITIONS (data, no trust needed)
def FuchsBenchmark : ShellParams :=
  { R_inner := 10, R_outer := 20, beta := 1/50, M_published := 3.334 }   -- metres, G = c = 1

def Design : ShellDesign :=                 -- read verbatim from warp_shell_results.json
  { s_coeffs := [...], u_coeffs := [...], shift_coeffs := [...],
    r1_iso := ..., r2_iso := ..., shift_lo := ..., shift_hi := ... }    -- exact rationals

def ADMMass (d : ShellDesign) : ℚ := ...    -- integral of a piecewise polynomial: exact
```

```
-- ASSUMPTIONS (trusted, to be eliminated over time)
axiom einstein_tensor_formula :
  ∀ (jets : FieldJets), EinsteinTensor (metricOf jets) = G_closed_form jets
  -- The symbolic Einstein tensor of the 3+1 ansatz.
  -- Removable in principle: it is a finite rational-function identity that Lean
  -- could verify by clearing denominators and using `ring`, though the expression is large.

axiom tetrad_projection :
  ∀ p, FrameStress (Design, p) = E(p)ᵀ · (G_closed_form (jets Design p) / (8π)) · E(p)
```

Everything after this point is a **theorem**, proved from these definitions and assumptions.

## 3. Energy conditions for all observers

### 3a. At a single point: replacing "for all observers" with one matrix check

At a point with frame energy density ρ, momentum q and stress S, the NEC says ρ + 2q·n + nᵀSn ≥ 0 for every unit vector n. Quantifying over infinitely many observers is awkward to prove. The S-lemma (lossless for a single quadratic constraint) replaces it with an existence statement:

```
theorem nec_of_certificate (ρ : ℚ) (q : ℚ³) (S : ℚ³ˣ³) (λ : ℚ)
    (h : PosSemidef !![ρ - λ, qᵀ; q, S + λ • I₃]) :
    ∀ n : ℝ³, ‖n‖ = 1 → ρ + 2 * (q ⬝ n) + nᵀ S n ≥ 0
```

Python supplies λ (the multiplier its dual solver already computes) and an exact LDLᵀ factorisation of the 4×4 matrix with rational entries. Lean checks that the factorisation multiplies back exactly and that every pivot is non-negative. WEC and SEC follow the same pattern with an extra sign condition λ ≥ 0 (the unit-ball version). SEC uses the trace-reversed tensor.

### 3b. DEC: the current gap

The DEC check in the code is sampled, not certified. Options for a certificate:

- If the stress tensor is Hawking–Ellis type I at the point (true for this shell), exhibit the eigenframe with exact rational bounds and check ρ ≥ |pᵢ| on the eigenvalues.
- Alternatively, use the bilinear form min over null k, l of T(k, l) ≥ 0, covered by a sum-of-squares certificate.

Until this exists, `Theorem DEC` rests on sampling and should be labelled as such.

### 3c. Between the points: where both edge bugs lived

Both withdrawn designs passed at every sampled point and failed between them. A grid-based check cannot prevent that; a continuum argument can.

- Every radial profile in the design is a B-spline: a **piecewise polynomial** with rational coefficients.
- On each polynomial piece, A, P, S and their derivatives have rigorous enclosures, from Bernstein-coefficient bounds or interval Taylor models.
- The certificate condition from 3a is therefore checkable on a whole **box** [r_a, r_b] × [θ_a, θ_b] at once. Use interval arithmetic for the matrix entries, and a single λ per box with a positive-semidefinite margin large enough to absorb the enclosure width.
- Lean checks that the boxes **cover** the shell, the interior and the exterior, with no gaps.

```
theorem Energy_Conditions_Satisfied_For_All_Observers :
    ∀ x ∈ Spacetime, ∀ k : NullVector x, T(x)(k, k) ≥ 0          -- NEC
  ∧ ∀ x, ∀ u : TimelikeVector x, T(x)(u, u) ≥ 0                  -- WEC
  ∧ ∀ x, ∀ u : TimelikeVector x, (T - ½ tr T · g)(x)(u, u) ≥ 0   -- SEC
  ∧ DEC_statement                                                -- pending 3b
  := by
    cover_by_boxes Design.boxes      -- boxes and λ's are emitted by Python
    · exact interior_flat_vacuum     -- r < r1: constant fields, T = 0 exactly
    · exact exterior_schwarzschild   -- r > r2: vacuum, proved once symbolically
    · intro b hb; exact box_certificate b (certs b)
```

Point evaluation is replaced by box evaluation, so "passes on the grid, fails between grid points" becomes impossible. A failure appears as a box whose certificate does not check, and the build stops.

## 4. The lower bound

The original request suggested `Mass_Lower_Bound_Is_Strict`. Two corrections to that name:

- The bound is **not known to be strict** (attained). The best verified design is about 3.8× above it.
- The bound is proved **only under explicit hypotheses**, which belong in the theorem statement:
  1. the linearised (weak-field) momentum constraint, which neglects O(M/R) and O(β²);
  2. a spherically symmetric energy density;
  3. the Eulerian dominant energy condition, ρ ≥ |j|;
  4. the discretisation used by the linear program, until a continuum version is proved.

A linear-program lower bound has a natural certificate: a **dual feasible vector** y. Weak duality says bᵀy ≤ (any primal value), so Lean only has to check that y is feasible, which is a sparse rational matrix–vector inequality, and evaluate bᵀy exactly.

```
theorem Mass_Lower_Bound_Linearised
    (hlin : LinearisedMomentumConstraint) (hsph : SphericalDensity)
    (hdec : EulerianDEC) (hdisc : Discretisation n_r := 800) :
    ∀ shell ∈ WarpShellClass FuchsBenchmark, ADMMass shell ≥ bᵀy := by   -- bᵀy ≈ 0.2017 m for the shipped geometry
  apply lp_weak_duality
  exact dual_certificate_checks y   -- y emitted by the LP solver, checked in ℚ
```

Upgrading this to a statement about the full nonlinear problem means bounding the neglected O(M/R) and O(β²) terms explicitly. That is a well-defined research task.

The withdrawn 0.1255 m bound would have been caught here: with the edge nodes missing from the constraint set, the formal statement of feasibility would not match the definition of the class, and the proof would fail.

## 5. Preventing regressions

| Failure mode seen in this project | How the formal pipeline stops it |
|---|---|
| Violation between grid points (both withdrawn designs) | Box-cover certificates (3c) replace point sampling |
| A modelling error in the bound (edge nodes left out) | The theorem statement fixes the class; a mismatched LP cannot prove it |
| A future optimiser run that silently regresses | CI gate: every candidate must ship certificates that `lake build` accepts |
| Someone weakening a theorem to make a design pass | Theorem statements live in a separate, protected file; changes need review |
| Floating-point roundoff hiding a small violation | All checks run in exact rational or interval arithmetic |

Proposed CI flow: optimisation run → export design and certificates as exact rationals → Lean check → merge only if every theorem builds with no `sorry` and no new axioms.

## 6. Suggested build order

1. LP dual certificate for the lower bound. Smallest step, pure linear algebra over ℚ.
2. Pointwise NEC/WEC/SEC certificates at the existing grid points.
3. Box-cover certificates for NEC/WEC/SEC: this closes the edge-bug gap.
4. A DEC certificate.
5. Eliminate `einstein_tensor_formula` by verifying the symbolic identity in Lean.
6. Nonlinear corrections to the lower bound.
