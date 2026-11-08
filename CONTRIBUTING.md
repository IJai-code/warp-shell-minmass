# Community Review & Contribution Protocol

This result has not been peer reviewed. The most valuable contribution is an **independent attempt to break it**. We especially invite the authors of the original warp-shell paper and of the Warp Factory toolkit (Fuchs, Helmerich, Bobrick, Sellers, Melcher, Martire) to cross-check it with their own tools.

## 1. Reproduce

```bash
git clone https://github.com/IJai-code/warp-shell-minmass.git
cd warp-shell-minmass
python3 -m venv .venv && source .venv/bin/activate
pip install numpy scipy sympy matplotlib
python3 warp_shell_minmass.py
```

The run takes about 7 minutes on 2 cores. It should print a lower bound of about 0.2017 m, then a β = 0.02 shell of about 0.7586 m (1.02×10²⁷ kg) and a β = 0.06 shell of about 2.970 m (4.00×10²⁷ kg), both reported as `PASS`.

Please record your Python, NumPy, SciPy and SymPy versions and your platform. Small differences in the last digits are expected across platforms; a `FAIL`, or a mass that differs by more than about 1%, is a finding worth reporting.

## 2. Cross-check in Warp Factory or another toolkit

The strongest test is an evaluation that shares no code with this repository.

1. **Rebuild the metric** from `warp_shell_output/warp_shell_results.json`. Each design lists the spline coefficients of the density source `s`, the lapse source `u` and the shift transition, together with the isotropic radii and the shift support. The construction is defined in sections 2 and 6 of `warp_shell_minmass.py` (`Profiles`, `design_for`).
2. **Sample g_μν on a Cartesian grid** in the format your toolkit expects. This repository does not yet include an exporter; a pull request adding one would be very welcome.
3. **Evaluate the energy conditions** with your own observer sampling. Warp Factory's published analysis used 100 orientations × 10 speeds per point and reports violations as negative values.
4. **Compare:**
   - ADM mass, from the far-field metric (expected: P → 1 + M/2r with M as listed);
   - areal radii of 10 m (inner) and 20 m (outer);
   - the sign of every energy condition everywhere, especially in the shift transitions and within 1 m of the shell edges, where earlier designs failed;
   - the transport effect: the published light-travel-time test, which was not repeated here.

## 3. Reporting

Open an issue titled `[review] <short description>` and include:

- what you ran (commit hash, toolkit and version, grid resolution, observer sampling);
- the quantity you compared and both values;
- for any energy-condition violation: the location (r, θ), the condition, and its value relative to the local energy density.

Negative findings are as welcome as confirmations. Any confirmed violation is listed in the README under *Claims withdrawn*, credited to the reporter.

To reach the original authors, please use the corresponding-author contact on their arXiv paper ([2405.02709](https://arxiv.org/abs/2405.02709)) or their toolkit's public issue tracker.

## 4. Code contributions

Welcome areas, roughly in order of value:

1. A Cartesian-grid metric exporter for Warp Factory and other toolkits.
2. A certified DEC check (currently sampled).
3. Certificate export for the formal verification plan in [docs/FORMAL_VERIFICATION.md](docs/FORMAL_VERIFICATION.md), starting with the LP dual certificate for the lower bound.
4. Non-spherical density profiles, which the bound does not cover.
5. Bounding the O(M/R) and O(β²) corrections to the lower bound.

Rules for pull requests:

- Any new design must pass the full `stress_test` (4,000 radii, interleaved angular grids, 3,000 DEC directions) and record its result in the JSON output.
- Do not relax tolerances or shrink stress-test resolutions to make a design pass.
- Changes to the physics (metric ansatz, energy-condition definitions, the bound's constraint set) need a test against a known solution, as in the README's validation table.

## 5. License

All code and generated data in this repository, including every file in `warp_shell_output/`, are released under the **MIT License** (see [LICENSE](LICENSE)). You may use, copy, modify, merge, publish, distribute, sublicense and sell copies, for academic or commercial purposes, provided the copyright notice and permission notice are kept. By submitting a contribution you agree to license it under the same terms.
