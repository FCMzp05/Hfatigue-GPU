#!/usr/bin/env python3
"""Error-controlled cycle blocks on the Yang et al. (2026) constitutive model.

The unified-model manuscript verifies its coupled full-step/two-half-step cycle
block on the Cui et al. (2024) compact-tension model, where the hydrogen
degradation is the instantaneous pressure-vessel function
``f_H = 0.12 + 0.88 exp(-7 c^2)``.  A controller that is only ever tested inside
the constitutive model it was designed with cannot be claimed to be a property
of the algorithm.  This driver therefore re-runs the same controller, with the
same preregistered tolerances and no retuning, on an independent published
model: the Langmuir coverage law ``f_H = 1 - chi * theta`` of Yang et al.
(2026), its mean-load fatigue accumulation (their Eq. 16) and its SENT and
compact-tension geometries.

The driver reuses the verified Yang operators rather than reimplementing them:
``make_solvers`` from the reproduction solver supplies the identical mechanics,
phase-field bound-constrained solve and stress-assisted diffusion.  Only the
time integration around them is replaced.

Three run modes are provided:

``reference``  one cycle per solve, the ground truth for block error;
``fixed``      a fixed cycle block, the negative control;
``controlled`` the error-controlled block of the manuscript's Algorithm 1.

The finite-rate hydrogen memory of the manuscript is carried over as well.  The
Yang law is treated as the equilibrium value of a relaxing internal variable,

    d(zeta)/dt = (zeta_eq(c) - zeta) / tau_H,

integrated exactly over the physical block time ``dt = dN / f``.  Setting
``tau_H = 0`` recovers the published instantaneous law exactly, so the
verification runs and the memory study share one code path.
"""

from __future__ import annotations

from pathlib import Path as _Path
import sys as _sys
_root = next(
    p for p in _Path(__file__).resolve().parents
    if (p / "code" / "reproductions" / "cui2024").is_dir()
)
_sys.path.insert(0, str(_root / 'code' / 'reproductions/cui2024'))
_sys.path.insert(0, str(_root / 'code' / 'reproductions/yang2026'))
_sys.path.insert(0, str(_root / 'code' / 'paper'))

import argparse
import csv
import json
import time
from pathlib import Path

import yang2026_fig4_sent_p1_cudss as yang

REPO = yang.root()
DEFAULT_OUT = REPO / "outputs" / "unified_model_paper" / "yang_transfer"


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("reference", "fixed", "controlled"),
                   default="controlled")
    p.add_argument("--mesh", type=Path, default=yang.default_mesh())
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--total-cycles", type=int, default=20000)
    p.add_argument("--delta-n", type=int, default=100,
                   help="fixed block size, or the initial trial block size")
    p.add_argument("--min-delta-n", type=int, default=1)
    p.add_argument("--max-delta-n", type=int, default=4096)

    # Preregistered controller tolerances, identical to the manuscript.
    p.add_argument("--crack-rtol", type=float, default=5.0e-3)
    p.add_argument("--crack-atol", type=float, default=1.0e-6,
                   help="absolute crack-area floor [mm^2]")
    p.add_argument("--phase-rtol", type=float, default=1.0e-2)
    p.add_argument("--concentration-rtol", type=float, default=5.0e-3)
    p.add_argument("--memory-rtol", type=float, default=5.0e-3)
    p.add_argument("--growth-patience", type=int, default=4,
                   help="consecutive comfortable accepts before the block grows")
    p.add_argument("--growth-margin", type=float, default=0.5,
                   help="error ratio below which an accept counts as comfortable")
    p.add_argument("--probe-every", type=int, default=1,
                   help="evaluate the full/two-half estimate every k-th block; "
                        "the intervening blocks reuse the accepted size and "
                        "take a single full step, which amortises the cost of "
                        "the estimate over k blocks")

    # Constitutive model, defaulting to the published SENT case.
    p.add_argument("--split", default="spectral",
                   choices=("spectral", "volumetric", "felino_stress"))
    p.add_argument("--hydrogen", choices=("on", "off"), default="off")
    p.add_argument("--hydrogen-law", choices=("langmuir", "pressure_vessel"),
                   default="langmuir")
    p.add_argument("--hydrogen-memory-tau-s", type=float, default=0.0)
    p.add_argument("--environment-wppm", type=float, default=1.0)
    p.add_argument("--precharge-ratio", type=float, default=0.1)
    p.add_argument("--diffusivity-mm2-s", type=float, default=2.0e-4)
    p.add_argument("--frequency-hz", type=float, default=1.0)
    p.add_argument("--diffusion-max-step-s", type=float, default=1.0)
    p.add_argument("--diffusion-max-substeps", type=int, default=0,
                   help="cap the transport substeps per block; with the cap "
                        "the transport cost follows the number of blocks "
                        "rather than the number of cycles, and the resulting "
                        "transport error is what the concentration component "
                        "of the block estimator then has to police")
    p.add_argument("--chi", type=float, default=yang.HYDROGEN_CHI)
    p.add_argument("--delta-gb-j-mol", type=float, default=yang.DELTA_GB)
    p.add_argument("--gc", type=float, default=yang.GC)
    p.add_argument("--length-scale", type=float, default=yang.L0)
    p.add_argument("--alpha-t", type=float, default=yang.ALPHA_T)
    p.add_argument("--umax", type=float, default=yang.U_MAX)
    p.add_argument("--load-ratio", type=float, default=yang.LOAD_RATIO)
    p.add_argument("--fatigue-n", type=float, default=yang.FATIGUE_N)

    p.add_argument("--loading", choices=("displacement", "pin"),
                   default="displacement")
    p.add_argument("--pin-load-n", type=float, default=590.0)
    p.add_argument("--initial-crack-damage", action="store_true",
                   help="impose d = 1 on the Crack set, as the CT case does")
    p.add_argument("--precharge-hours", type=float, default=0.0,
                   help="physical charging time before cycling starts")
    p.add_argument("--stop-growth-mm", type=float, default=0.0,
                   help="stop once the tip has advanced this far [mm]")

    p.add_argument("--stagger-tol", type=float, default=1.0e-6)
    p.add_argument("--max-stagger", type=int, default=25)
    p.add_argument("--stagger-relaxation", type=float, default=1.0)
    p.add_argument("--cg-tol", type=float, default=1.0e-8)
    p.add_argument("--cg-maxiter", type=int, default=20000)
    p.add_argument("--dtype", choices=("float64", "float32"), default="float64")
    p.add_argument("--stop-crack-area", type=float, default=None,
                   help="stop once the smeared crack area exceeds this [mm^2]")
    p.add_argument("--wall-budget-s", type=float, default=0.0,
                   help="stop cleanly once this wall time is exceeded")
    return p


class Model:
    """The Yang operators plus the block advance they are wired into."""

    def __init__(self, args):
        self.args = args
        jnp, np, jax = yang.jnp, yang.np, yang.jax
        self.dtype = jnp.float64 if args.dtype == "float64" else jnp.float32
        nodes, tri, areas, groups = yang.load_mesh(args.mesh)
        self.nodes, self.tri, self.groups = nodes, tri, groups
        bmat, dofs, mass, grad = yang.preprocess(nodes, tri, areas)
        self.nnode, self.ncell = len(nodes), len(tri)
        ndof = 2 * self.nnode

        external_force_np = None
        if args.loading == "displacement":
            fixed = np.unique(np.concatenate(
                (2 * groups["BOTTOM"], 2 * groups["BOTTOM"] + 1,
                 2 * groups["TOP"] + 1)))
            boundary_np = np.zeros(ndof)
            boundary_np[2 * groups["TOP"] + 1] = args.umax
        else:
            # Compact tension: the lower pin hole is held and the upper hole
            # carries the load over the half of its boundary a pin bears on.
            fixed = np.unique(np.concatenate(
                (2 * groups["BOTTOM"], 2 * groups["BOTTOM"] + 1)))
            boundary_np = np.zeros(ndof)
            loaded = groups["TOP"]
            bearing = nodes[loaded, 1] >= nodes[loaded].mean(axis=0)[1]
            if not bearing.any():
                raise RuntimeError("no bearing nodes found on the loaded hole")
            weights = np.zeros(len(loaded))
            weights[bearing] = 1.0 / bearing.sum()
            external_force_np = np.zeros(ndof)
            external_force_np[2 * loaded + 1] = args.pin_load_n * weights
        free_u = np.ones(ndof)
        free_u[fixed] = 0
        self.boundary = jnp.asarray(boundary_np, dtype=self.dtype)

        self.c_environment = args.environment_wppm * yang.C_ENV
        (self.solve_u, self.qois, self.fatigue_factor, self.solve_d,
         self.solve_c, self.hydrogen_factor) = yang.make_solvers(
            nodes, tri, areas, bmat, dofs, mass, grad, free_u, args.split,
            self.dtype, args.cg_tol, args.cg_maxiter, args.diffusivity_mm2_s,
            args.gc, args.chi, self.c_environment, args.delta_gb_j_mol,
            external_force_np=external_force_np,
            hydrogen_law=args.hydrogen_law, fatigue_energy="mean_load",
            multiply_by_degradation=False)
        self.hydrogen = args.hydrogen == "on"
        # Consistent-mass row sums give the domain integral of a nodal field,
        # which is how the lattice-hydrogen inventory is accumulated.
        self.nodal_volume = np.bincount(
            tri.reshape(-1), weights=mass.sum(axis=2).reshape(-1),
            minlength=self.nnode)
        self.crack_plane = np.flatnonzero(
            np.abs(nodes[:, 1]) < 1.5 * args.length_scale)
        far_edge = nodes[:, 0].max()
        self.ligament_end = np.flatnonzero(
            (nodes[:, 0] > far_edge - 4.0 * args.length_scale)
            & (np.abs(nodes[:, 1]) < 6.0 * args.length_scale))

    def initial_state(self):
        jnp, np, jax = yang.jnp, yang.np, yang.jax
        d = self.dtype
        damage_np = np.zeros(self.nnode)
        if self.args.initial_crack_damage:
            damage_np[self.groups["Crack"]] = 1.0
        damage = jnp.asarray(damage_np, dtype=d)
        concentration = jnp.full(
            self.nnode,
            self.args.precharge_ratio * self.c_environment
            if self.hydrogen else 0.0, dtype=d)
        if self.hydrogen and self.args.precharge_hours > 0:
            # Charging for a finite time leaves a diffusion profile, not the
            # uniform field that a precharge ratio alone would give.
            seconds = self.args.precharge_hours * 3600.0
            concentration, _ = self.solve_c(
                concentration, jnp.zeros(self.ncell, dtype=d), damage, seconds,
                self.groups["Crack"],
                max(1, int(np.ceil(seconds / self.args.diffusion_max_step_s))))
            charged = np.asarray(jax.device_get(concentration))
            print(f"pre-charged {self.args.precharge_hours:g} h: "
                  f"c_max={charged.max():.4e} mol/mm^3", flush=True)
        return {
            "u": jnp.zeros(2 * self.nnode, dtype=d),
            "damage": damage,
            "history": jnp.zeros(self.ncell, dtype=d),
            "alpha": jnp.zeros(self.ncell, dtype=d),
            "c": concentration,
            "zeta": jnp.ones(self.ncell, dtype=d),
        }

    def advance(self, state, delta_n: int):
        """Advance ``delta_n`` cycles with one staggered coupled solve."""
        jnp, np, jax = yang.jnp, yang.np, yang.jax
        args = self.args
        dt = delta_n / args.frequency_hz
        substeps = max(1, int(np.ceil(dt / args.diffusion_max_step_s)))
        if args.diffusion_max_substeps > 0:
            substeps = min(substeps, args.diffusion_max_substeps)
        damage_start = state["damage"]
        alpha_start, c_start = state["alpha"], state["c"]
        history_start, zeta_start = state["history"], state["zeta"]

        u, ur, ui = self.solve_u(damage_start, self.boundary, state["u"])
        psi, psi_eff, _, sigma_h = self.qois(u, damage_start)
        alpha = alpha_start + delta_n * psi_eff
        if self.hydrogen:
            c, cr = self.solve_c(c_start, sigma_h, damage_start, dt,
                                 self.groups["Crack"], substeps)
            zeta = self.relax(zeta_start, self.hydrogen_factor(c), dt)
        else:
            c, cr = c_start, jnp.asarray(0.0, dtype=self.dtype)
            zeta = zeta_start
        factor = self.fatigue_factor(alpha) * zeta
        history = jnp.maximum(history_start, psi)
        damage = damage_start
        solves, residual = 1, float("inf")

        for stagger in range(1, args.max_stagger + 1):
            old_d, old_alpha, old_c = damage, alpha, c
            damage_candidate, dr, _ = self.solve_d(history, factor, damage_start)
            damage = jnp.maximum(
                damage_start,
                old_d + args.stagger_relaxation * (damage_candidate - old_d))
            u, ur, ui = self.solve_u(damage, self.boundary, u)
            psi, psi_eff, area, sigma_h = self.qois(u, damage)
            alpha_candidate = alpha_start + delta_n * psi_eff
            alpha = old_alpha + args.stagger_relaxation * (
                alpha_candidate - old_alpha)
            if self.hydrogen:
                c_candidate, cr = self.solve_c(c_start, sigma_h, damage, dt,
                                               self.groups["Crack"], substeps)
                c = old_c + args.stagger_relaxation * (c_candidate - old_c)
                zeta = self.relax(zeta_start, self.hydrogen_factor(c), dt)
            factor = self.fatigue_factor(alpha) * zeta
            history = jnp.maximum(history_start, psi)
            solves += 1
            jax.block_until_ready(damage)

            def drift(new, old):
                return jnp.linalg.norm(new - old) / jnp.maximum(
                    jnp.linalg.norm(new), 1e-30)

            residual = float(jax.device_get(jnp.maximum(
                jnp.maximum(drift(damage, old_d), drift(alpha, old_alpha)),
                drift(c, old_c) if self.hydrogen else 0.0)))
            if residual < args.stagger_tol:
                break

        new = {"u": u, "damage": damage, "history": history, "alpha": alpha,
               "c": c, "zeta": zeta}
        info = {"crack_area": float(jax.device_get(area)),
                "stagger_iterations": stagger, "stagger_residual": residual,
                "u_residual": float(jax.device_get(ur)),
                "d_residual": float(jax.device_get(dr)),
                "c_residual": float(jax.device_get(cr)),
                "solves": solves}
        return new, info

    def relax(self, zeta_old, zeta_eq, dt):
        """Exact update of the finite-rate degradation memory over ``dt``."""
        tau = self.args.hydrogen_memory_tau_s
        if tau <= 0:
            return zeta_eq
        decay = yang.np.exp(-dt / tau)
        return zeta_eq + (zeta_old - zeta_eq) * decay

    def crack_tip_x(self, damage) -> float:
        np, jax = yang.np, yang.jax
        field = np.asarray(jax.device_get(damage))
        broken = self.crack_plane[field[self.crack_plane] > 0.5]
        return float(self.nodes[broken, 0].max()) if broken.size else float("nan")

    def inventory(self, concentration) -> float:
        """Domain-integrated lattice hydrogen, in mol per unit thickness."""
        field = yang.np.asarray(yang.jax.device_get(concentration))
        return float(self.nodal_volume @ field)

    def traversed(self, damage) -> bool:
        """The ligament is gone, so equilibrium no longer has a solution."""
        if self.ligament_end.size == 0:
            return False
        field = yang.np.asarray(yang.jax.device_get(damage))
        return bool(field[self.ligament_end].max() > 0.95)


def block_error(model, full, half, info_full, info_half, args):
    """Normalised full-step versus two-half-step discrepancies."""
    jnp, jax = yang.jnp, yang.jax

    def relative(a, b, rtol):
        num = float(jax.device_get(jnp.linalg.norm(a - b)))
        den = rtol * max(float(jax.device_get(jnp.linalg.norm(b))), 1e-30)
        return num / den

    area_f, area_h = info_full["crack_area"], info_half["crack_area"]
    errors = {
        "crack": abs(area_f - area_h) / (args.crack_rtol * abs(area_h)
                                         + args.crack_atol),
        "phase": relative(full["damage"], half["damage"], args.phase_rtol),
    }
    if model.hydrogen:
        errors["concentration"] = relative(full["c"], half["c"],
                                           args.concentration_rtol)
        if args.hydrogen_memory_tau_s > 0:
            errors["memory"] = relative(full["zeta"], half["zeta"],
                                        args.memory_rtol)
    return errors


FIELDS = ("block", "cycle", "delta_n", "crack_area_mm2", "crack_tip_x_mm",
          "max_damage", "min_hydrogen_factor", "max_concentration_mol_mm3",
          "hydrogen_inventory_mol_per_mm",
          "block_error_ratio", "error_crack", "error_phase",
          "error_concentration", "error_memory", "rejections", "solves",
          "stagger_iterations", "stagger_residual", "u_linear_residual",
          "d_linear_residual", "c_linear_residual", "block_wall_s",
          "cumulative_wall_s")


def main(argv=None) -> None:
    args = make_parser().parse_args(argv)
    yang.load_runtime()
    yang.jax.config.update("jax_enable_x64", args.dtype == "float64")
    yang.__dict__.update(L0=args.length_scale, ALPHA_T=args.alpha_t,
                         U_MAX=args.umax, LOAD_RATIO=args.load_ratio,
                         FATIGUE_N=args.fatigue_n, GC=args.gc)
    jnp, np, jax = yang.jnp, yang.np, yang.jax

    model = Model(args)
    state = model.initial_state()
    first_tip = model.crack_tip_x(state["damage"])
    if not np.isfinite(first_tip):
        first_tip = float(model.nodes[:, 0].min())

    def failed(current) -> bool:
        """Under load control the ligament simply gives way after real growth."""
        if model.traversed(current["damage"]):
            return True
        if args.stop_growth_mm <= 0:
            return False
        tip = model.crack_tip_x(current["damage"])
        return bool(np.isfinite(tip) and tip - first_tip > args.stop_growth_mm)

    args.outdir.mkdir(parents=True, exist_ok=True)
    print(f"mode={args.mode} mesh={model.nnode} nodes/{model.ncell} triangles "
          f"hydrogen={model.hydrogen} tau_H={args.hydrogen_memory_tau_s:g}s "
          f"device={jax.devices()[0]}", flush=True)

    delta_n = 1 if args.mode == "reference" else args.delta_n
    cycle, block, total_solves, total_rejects = 0, 0, 0, 0
    start = time.perf_counter()
    accepted_history, comfortable = [], 0
    blocks_since_probe = args.probe_every - 1
    stop_reason = "cycle budget reached"

    with (args.outdir / "results.csv").open("w", newline="",
                                            encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        while cycle < args.total_cycles:
            tic = time.perf_counter()
            step = min(delta_n, args.total_cycles - cycle)
            rejections = 0
            probe_due = blocks_since_probe >= args.probe_every - 1

            try:
                if args.mode == "controlled" and not probe_due:
                    # An amortised block: the size most recently certified by a
                    # probe is reused and only the full step is taken.  Nothing
                    # is verified here, which is the price of the amortisation.
                    state, info = model.advance(state, step)
                    errors, ratio, accepted = {}, float("nan"), step
                    total_solves += info["solves"]
                    blocks_since_probe += 1
                elif args.mode == "controlled":
                    while True:
                        step = min(step, args.total_cycles - cycle)
                        full, info_full = model.advance(state, step)
                        if step >= 2:
                            first, info_first = model.advance(state, step // 2)
                            half, info_half = model.advance(first, step - step // 2)
                            errors = block_error(model, full, half, info_full,
                                                 info_half, args)
                            solves = (info_full["solves"] + info_first["solves"]
                                      + info_half["solves"])
                            ratio = max(errors.values())
                        else:
                            half, info_half, errors, ratio = full, info_full, {}, 0.0
                            solves = info_full["solves"]
                        total_solves += solves
                        if not np.isfinite(ratio):
                            raise FloatingPointError("non-finite block error")
                        if ratio <= 1.0 or step <= args.min_delta_n:
                            state, info = half, info_half
                            break
                        rejections += 1
                        total_rejects += 1
                        step = max(args.min_delta_n, step // 2)
                    if ratio <= args.growth_margin and rejections == 0:
                        comfortable += 1
                    else:
                        comfortable = 0
                    accepted = step
                    blocks_since_probe = 0
                    if comfortable >= args.growth_patience:
                        delta_n = min(args.max_delta_n, 2 * accepted)
                        comfortable = 0
                    else:
                        delta_n = accepted
                else:
                    state, info = model.advance(state, step)
                    errors, ratio, accepted = {}, float("nan"), step
                    total_solves += info["solves"]
            except Exception:
                # Once the ligament is broken the equilibrium tangent is
                # singular, so a solver failure there is the end of the
                # analysis rather than a defect.  Anything else is re-raised.
                if failed(state):
                    stop_reason = "ligament exhausted"
                    print(f"stopping at cycle {cycle}: {stop_reason}", flush=True)
                    break
                raise

            cycle += accepted
            block += 1
            damage = np.asarray(jax.device_get(state["damage"]))
            concentration = np.asarray(jax.device_get(state["c"]))
            zeta = np.asarray(jax.device_get(state["zeta"]))
            elapsed = time.perf_counter() - tic
            cumulative = time.perf_counter() - start
            row = dict(
                block=block, cycle=cycle, delta_n=accepted,
                crack_area_mm2=info["crack_area"],
                crack_tip_x_mm=model.crack_tip_x(state["damage"]),
                max_damage=float(damage.max()),
                min_hydrogen_factor=float(zeta.min()),
                max_concentration_mol_mm3=float(concentration.max()),
                hydrogen_inventory_mol_per_mm=model.inventory(state["c"]),
                block_error_ratio=ratio,
                error_crack=errors.get("crack", float("nan")),
                error_phase=errors.get("phase", float("nan")),
                error_concentration=errors.get("concentration", float("nan")),
                error_memory=errors.get("memory", float("nan")),
                rejections=rejections, solves=total_solves,
                stagger_iterations=info["stagger_iterations"],
                stagger_residual=info["stagger_residual"],
                u_linear_residual=info["u_residual"],
                d_linear_residual=info["d_residual"],
                c_linear_residual=info["c_residual"],
                block_wall_s=elapsed, cumulative_wall_s=cumulative)
            writer.writerow(row)
            handle.flush()
            accepted_history.append(row)
            if accepted > 1 or block % 250 == 0:
                print(f"block={block:5d} cycle={cycle:7d} dN={accepted:5d} "
                      f"A={info['crack_area']:.6e} maxd={damage.max():.5f} "
                      f"E={ratio:.3f} rej={rejections} "
                      f"wall={elapsed:.2f}s cum={cumulative:.0f}s", flush=True)

            if args.stop_crack_area and info["crack_area"] > args.stop_crack_area:
                stop_reason = "crack area target reached"
                break
            if failed(state):
                stop_reason = "ligament exhausted"
                print(f"stopping at cycle {cycle}: {stop_reason}", flush=True)
                break
            if args.wall_budget_s and cumulative > args.wall_budget_s:
                stop_reason = "wall-time budget reached"
                break

    total = time.perf_counter() - start
    summary = {
        "driver": "yang_error_controlled_driver",
        "purpose": ("error-controlled cycle blocks applied to the Yang et al. "
                    "(2026) constitutive model without retuning"),
        "mode": args.mode,
        "mesh": str(args.mesh),
        "nodes": model.nnode,
        "triangles": model.ncell,
        "hydrogen": model.hydrogen,
        "hydrogen_law": args.hydrogen_law,
        "hydrogen_memory_tau_s": args.hydrogen_memory_tau_s,
        "frequency_hz": args.frequency_hz,
        "loading": args.loading,
        "pin_load_n": args.pin_load_n if args.loading == "pin" else None,
        "precharge_hours": args.precharge_hours,
        "tolerances": {"crack_rtol": args.crack_rtol,
                       "crack_atol_mm2": args.crack_atol,
                       "phase_rtol": args.phase_rtol,
                       "concentration_rtol": args.concentration_rtol,
                       "memory_rtol": args.memory_rtol},
        "cycles_completed": cycle,
        "blocks": block,
        "coupled_solves": total_solves,
        "rejections": total_rejects,
        "solves_per_cycle": total_solves / max(cycle, 1),
        "mean_accepted_delta_n": cycle / max(block, 1),
        "probe_every": args.probe_every,
        "final_crack_area_mm2": accepted_history[-1]["crack_area_mm2"]
        if accepted_history else None,
        "wall_s": total,
        "stop_reason": stop_reason,
    }
    (args.outdir / "summary.json").write_text(json.dumps(summary, indent=2),
                                              encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
