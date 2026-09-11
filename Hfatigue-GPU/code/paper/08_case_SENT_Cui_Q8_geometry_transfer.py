#!/usr/bin/env python3
"""Q8 SENT adapter for the unified hydrogen-fatigue block algorithm.

This is a numerical geometry-migration example, not an experimental
validation.  It transfers the time-consistent Cui cycle-block algorithm to
the Golahmar cracked-square/SENT mesh and displacement boundary conditions.

The Q8 kernels, cuDSS backend and constitutive block primitives are imported
from ``cui2024_fig6_paris_curves_ct_q8.py``.  This adapter does not modify that
CT solver or any of its stored results.
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
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np

import cui2024_fig6_paris_curves_ct_q8 as core


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MESH = (
    ROOT
    / "outputs"
    / "hydrogen_fatigue_cracked_square"
    / "q8_jax"
    / "mesh_q8_paper.msh"
)
DEFAULT_OUT = ROOT / "outputs" / "unified_model_paper" / "sent"
WIDTH, HEIGHT, INITIAL_CRACK = 1.0, 1.0, 0.5
VERSION = "unified_model_sent_q8_geometry_adapter_v1"
SOURCE_FILES = (
    "code/reproductions/golahmar2022/golahmar2022_fig2_3_cracked_square_q8_cudss.py",
    "code/reproductions/cui2024/cui2024_fig6_paris_curves_ct_q8.py",
)


def configure_sent_material() -> None:
    """Use the cracked-square regularisation and transport scales."""
    core.E, core.NU = 210_000.0, 0.3
    core.LAMBDA = (
        core.E * core.NU
        / ((1.0 + core.NU) * (1.0 - 2.0 * core.NU))
    )
    core.MU = core.E / (2.0 * (1.0 + core.NU))
    core.K_BULK = core.LAMBDA + 2.0 * core.MU / 3.0
    core.GC = 2.7
    core.LENGTH_SCALE = 0.004
    core.KAPPA = 1.0e-7
    core.DIFFUSIVITY = 0.0127
    core.SIGMA_C = 4.0 * core.SIGMA_Y
    core.EPSILON_C = np.sqrt(
        core.GC / (3.0 * core.LENGTH_SCALE * core.E)
    )
    core.ALPHA_N = core.SIGMA_C * core.EPSILON_C / 2.0


def load_sent_mesh(path: Path):
    """Read Q8 cells and the bottom/top/crack physical groups."""
    mesh = core.meshio.read(path)
    nodes = np.asarray(mesh.points[:, :2], dtype=np.float64)
    blocks = [
        np.asarray(block.data, dtype=np.int64)
        for block in mesh.cells
        if block.type == "quad8"
    ]
    if not blocks:
        raise RuntimeError("mesh contains no quad8 cells")
    elements = np.concatenate(blocks)
    tags = {
        name.lower(): int(value[0])
        for name, value in mesh.field_data.items()
    }
    missing = {"bottom", "top", "crack"} - tags.keys()
    if missing:
        raise RuntimeError(
            f"mesh is missing physical groups: {sorted(missing)}"
        )
    groups: dict[str, list[int]] = {
        name: [] for name in ("bottom", "top", "crack")
    }
    physical = mesh.cell_data.get("gmsh:physical", [])
    for block_index, block in enumerate(mesh.cells):
        if block.type != "line3":
            continue
        for edge, tag in zip(block.data, physical[block_index]):
            for name in groups:
                if int(tag) == tags[name]:
                    groups[name].extend(map(int, edge))
    arrays = {
        name: np.unique(values) for name, values in groups.items()
    }
    if any(len(values) == 0 for values in arrays.values()):
        raise RuntimeError(
            "empty SENT physical group: "
            + ", ".join(
                f"{name}={len(values)}"
                for name, values in arrays.items()
            )
        )
    return nodes, elements, arrays


def crack_extension(nodes: np.ndarray, damage: np.ndarray) -> float:
    """Interpolate the contiguous d=0.95 centreline crack tip."""
    line = np.flatnonzero(
        np.isclose(nodes[:, 1], 0.0, atol=1.0e-10)
        & (nodes[:, 0] >= INITIAL_CRACK - 1.0e-10)
    )
    line = line[np.argsort(nodes[line, 0])]
    if len(line) == 0:
        raise RuntimeError("no centreline nodes found ahead of the crack")
    last = 0
    for index in range(1, len(line)):
        if damage[line[index]] < 0.95:
            break
        last = index
    tip = float(nodes[line[last], 0])
    if last + 1 < len(line):
        d0 = float(damage[line[last]])
        d1 = float(damage[line[last + 1]])
        if d0 >= 0.95 > d1 and abs(d1 - d0) > 1.0e-14:
            fraction = (0.95 - d0) / (d1 - d0)
            tip += fraction * float(
                nodes[line[last + 1], 0] - nodes[line[last], 0]
            )
    return min(WIDTH - INITIAL_CRACK, max(0.0, tip - INITIAL_CRACK))


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    result.add_argument(
        "--case",
        action="append",
        choices=("air", "hydrogen", "all"),
        help="minimal environment case (repeatable; default: air)",
    )
    result.add_argument(
        "--platform", choices=("auto", "cpu", "gpu"), default="gpu"
    )
    result.add_argument(
        "--linear-backend",
        choices=("jax-cg", "cudss"),
        default="cudss",
    )
    result.add_argument("--mesh", type=Path, default=DEFAULT_MESH)
    result.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    result.add_argument(
        "--integration", choices=("reduced", "full"), default="reduced"
    )
    result.add_argument("--max-cycles", type=int, default=100)
    result.add_argument("--cycle-jump", type=int, default=10)
    result.add_argument("--min-cycle-jump", type=int, default=1)
    result.add_argument(
        "--fixed-cycle-jump",
        action="store_true",
        help="keep the requested block size (apart from the final remainder)",
    )
    result.add_argument(
        "--block-error-control",
        action="store_true",
        help="use a full-step/two-half-step estimate and rollback on rejection",
    )
    result.add_argument("--frequency-hz", type=float, default=1.0)
    result.add_argument("--load-ratio", type=float, default=0.1)
    result.add_argument(
        "--maximum-displacement-mm", type=float, default=2.0e-3
    )
    result.add_argument("--hydrogen-wppm", type=float, default=1.0)
    result.add_argument("--hydrogen-memory-tau-s", type=float, default=0.0)
    result.add_argument("--precharge-hours", type=float, default=0.0)
    result.add_argument("--precharge-steps", type=int, default=1)
    result.add_argument(
        "--diffusion-time-integrator",
        choices=("block", "subcycle"),
        default="block",
    )
    result.add_argument("--diffusion-max-step-s", type=float, default=3600.0)
    result.add_argument(
        "--diffusion-cycle-mode",
        choices=("average", "split"),
        default="average",
    )
    result.add_argument("--crack-penalty", type=float, default=1.0e5)
    result.add_argument("--crack-penalty-threshold", type=float, default=0.75)
    result.add_argument("--fatigue-n", type=float, default=1.25)
    result.add_argument("--alpha-0", type=float, default=10.5)
    result.add_argument(
        "--fatigue-degradation-exponent", type=float, default=2.0
    )
    result.add_argument(
        "--fatigue-energy",
        choices=("degraded", "undamaged"),
        default="degraded",
    )
    result.add_argument(
        "--fatigue-cycle-driving",
        choices=("maximum", "range"),
        default="maximum",
    )
    result.add_argument(
        "--hydrogen-scaled-alpha-n",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    result.add_argument("--block-crack-increment-rtol", type=float, default=0.005)
    result.add_argument(
        "--block-crack-increment-atol-mm", type=float, default=5.0e-5
    )
    result.add_argument("--block-phase-rtol", type=float, default=0.01)
    result.add_argument("--block-concentration-rtol", type=float, default=0.005)
    result.add_argument("--block-memory-rtol", type=float, default=0.005)
    result.add_argument("--max-damage-increment", type=float, default=0.05)
    result.add_argument("--max-stagger", type=int, default=50)
    result.add_argument("--stagger-tol", type=float, default=1.0e-3)
    result.add_argument("--linear-tol", type=float, default=1.0e-8)
    result.add_argument("--linear-maxiter", type=int, default=8000)
    result.add_argument("--linear-residual-tol", type=float, default=5.0e-5)
    result.add_argument("--print-every", type=int, default=1)
    return result


def validate_args(args: argparse.Namespace) -> None:
    positive = {
        "frequency": args.frequency_hz,
        "cycle jump": args.cycle_jump,
        "minimum cycle jump": args.min_cycle_jump,
        "diffusion maximum step": args.diffusion_max_step_s,
        "fatigue n": args.fatigue_n,
        "alpha_0": args.alpha_0,
        "fatigue degradation exponent": args.fatigue_degradation_exponent,
        "linear tolerance": args.linear_tol,
        "linear residual tolerance": args.linear_residual_tol,
    }
    bad = [name for name, value in positive.items() if value <= 0]
    if bad:
        raise ValueError(f"positive values required for: {', '.join(bad)}")
    if args.max_cycles < 1:
        raise ValueError("--max-cycles must be positive")
    if args.min_cycle_jump > args.cycle_jump:
        raise ValueError("--min-cycle-jump cannot exceed --cycle-jump")
    if not 0.0 <= args.load_ratio < 1.0:
        raise ValueError("--load-ratio must satisfy 0 <= R < 1")
    if args.hydrogen_wppm < 0 or args.hydrogen_memory_tau_s < 0:
        raise ValueError("hydrogen concentration and memory tau cannot be negative")
    if args.precharge_hours < 0 or args.precharge_steps < 1:
        raise ValueError("invalid precharge duration/step count")
    tolerances = (
        args.block_crack_increment_rtol,
        args.block_crack_increment_atol_mm,
        args.block_phase_rtol,
        args.block_concentration_rtol,
        args.block_memory_rtol,
    )
    if any(value <= 0 for value in tolerances):
        raise ValueError("block error-control tolerances must be positive")
    if args.fixed_cycle_jump and args.block_error_control:
        raise ValueError(
            "choose fixed blocks or rejecting block-error control, not both"
        )


def run_case(name: str, args: argparse.Namespace) -> None:
    configure_sent_material()
    total_started = time.perf_counter()
    mesh_path = args.mesh.expanduser().resolve()
    if not mesh_path.is_file():
        raise FileNotFoundError(f"required mesh does not exist: {mesh_path}")
    case_wppm = 0.0 if name == "air" else args.hydrogen_wppm
    outdir = args.outdir.expanduser().resolve() / name
    outdir.mkdir(parents=True, exist_ok=True)

    core.jax.config.update("jax_enable_x64", True)
    dtype = core.jnp.float64
    devices = core.jax.devices()
    if args.platform == "gpu" and not any(
        device.platform == "gpu" for device in devices
    ):
        raise RuntimeError(f"GPU requested; available devices={devices}")

    nodes, elements, groups = load_sent_mesh(mesh_path)
    shape, grad, bmat, dofs, det, mass, stiff, hourglass = core.preprocess(
        nodes, elements, integration=args.integration
    )
    nnode, ndof = len(nodes), 2 * len(nodes)
    anchor = groups["bottom"][np.argmax(nodes[groups["bottom"], 0])]
    fixed_u = np.unique(
        np.r_[
            2 * groups["bottom"] + 1,
            2 * groups["top"] + 1,
            2 * anchor,
        ]
    )
    free_u = np.ones(ndof)
    free_u[fixed_u] = 0.0
    fixed_d = np.zeros(nnode)
    fixed_d[groups["crack"]] = 1.0

    solve_u, solve_d, solve_c, qois = core.make_solvers(
        nodes,
        elements,
        shape,
        grad,
        bmat,
        dofs,
        det,
        mass,
        stiff,
        hourglass,
        free_u,
        fixed_d,
        dtype,
        args.linear_tol,
        args.linear_maxiter,
        linear_backend=args.linear_backend,
        fatigue_energy=args.fatigue_energy,
    )
    top_y = 2 * groups["top"] + 1
    minimum_boundary_np = np.zeros(ndof)
    maximum_boundary_np = np.zeros(ndof)
    maximum_boundary_np[top_y] = args.maximum_displacement_mm
    minimum_boundary_np[top_y] = (
        args.load_ratio * args.maximum_displacement_mm
    )
    minimum_boundary = core.jnp.asarray(minimum_boundary_np, dtype=dtype)
    maximum_boundary = core.jnp.asarray(maximum_boundary_np, dtype=dtype)
    outer_mask = (
        np.isclose(nodes[:, 0], 0.0, atol=1.0e-10)
        | np.isclose(nodes[:, 0], WIDTH, atol=1.0e-10)
        | np.isclose(nodes[:, 1], -HEIGHT / 2.0, atol=1.0e-10)
        | np.isclose(nodes[:, 1], HEIGHT / 2.0, atol=1.0e-10)
    )
    outer_mask_j = core.jnp.asarray(outer_mask)
    element_nodes = core.jnp.asarray(elements)
    shape_values = core.jnp.asarray(shape, dtype=dtype)

    unit_q = core.jnp.ones((len(elements), len(shape)), dtype=dtype)
    zero_q = core.jnp.zeros_like(unit_q)
    initial_d = core.jnp.asarray(fixed_d, dtype=dtype)
    equilibrated, equilibrium_residual = solve_d(
        zero_q, unit_q, initial_d, unit_q
    )
    state = {
        "u": core.jnp.zeros(ndof, dtype=dtype),
        "d": core.jnp.clip(
            core.jnp.maximum(equilibrated, initial_d), 0.0, 1.0
        ),
        "history": zero_q,
        "alpha_bar": zero_q,
        "peak_seen": zero_q,
        "c": core.jnp.zeros(nnode, dtype=dtype),
        "hydrogen_memory": zero_q,
    }
    c_env = case_wppm * 1.0e-6 * core.RHO_M / core.M_H
    max_linear_residual = float(core.jax.device_get(equilibrium_residual))
    if max_linear_residual > args.linear_residual_tol:
        raise RuntimeError(
            "initial AT2 equilibrium residual exceeded limit: "
            f"{max_linear_residual:.3e}"
        )

    def advance_concentration(cold, hydro, elapsed_s, damage):
        if case_wppm <= 0.0 or elapsed_s <= 0.0:
            return cold, core.jnp.asarray(0.0, dtype=dtype), 0
        substeps = (
            max(1, math.ceil(elapsed_s / args.diffusion_max_step_s))
            if args.diffusion_time_integrator == "subcycle"
            else 1
        )
        step_s = elapsed_s / substeps
        concentration = cold
        maximum_residual = core.jnp.asarray(0.0, dtype=dtype)
        for _ in range(substeps):
            concentration, residual = solve_c(
                concentration,
                hydro,
                outer_mask_j,
                c_env,
                step_s,
                damage,
                args.crack_penalty,
                args.crack_penalty_threshold,
            )
            maximum_residual = core.jnp.maximum(
                maximum_residual, residual
            )
        return concentration, maximum_residual, substeps

    def advance_memory(memory_start, concentration_q, elapsed_s):
        target_wppm = concentration_q * (
            1.0e6 * core.M_H / core.RHO_M
        )
        if args.hydrogen_memory_tau_s <= 0.0:
            return target_wppm
        survival = core.jnp.exp(
            -core.jnp.asarray(
                elapsed_s / args.hydrogen_memory_tau_s, dtype=dtype
            )
        )
        return target_wppm + (memory_start - target_wppm) * survival

    if case_wppm > 0.0 and args.precharge_hours > 0.0:
        precharge_step_s = (
            args.precharge_hours * 3600.0 / args.precharge_steps
        )
        for _ in range(args.precharge_steps):
            state["c"], residual, _ = advance_concentration(
                state["c"], zero_q, precharge_step_s, state["d"]
            )
            concentration_q = core.jnp.einsum(
                "qi,ei->eq",
                shape_values,
                state["c"][element_nodes],
            )
            state["hydrogen_memory"] = advance_memory(
                state["hydrogen_memory"],
                concentration_q,
                precharge_step_s,
            )
            max_linear_residual = max(
                max_linear_residual,
                float(core.jax.device_get(residual)),
            )

    def advance_cycle_block(initial_state: dict, block_jump: int):
        """Return a trial state; the input is unchanged and can be rolled back."""
        block_state = dict(initial_state)
        d_start = initial_state["d"]
        history_start = initial_state["history"]
        alpha_bar_start = initial_state["alpha_bar"]
        peak_seen_start = initial_state["peak_seen"]
        concentration_start = initial_state["c"]
        memory_start = initial_state["hydrogen_memory"]
        elapsed_s = block_jump / args.frequency_hz

        block_state["u"], residual_u_min = solve_u(
            d_start, minimum_boundary, initial_state["u"]
        )
        _, alpha_min, _, hydro_min = qois(block_state["u"], d_start)
        block_state["u"], residual_u_max = solve_u(
            d_start, maximum_boundary, block_state["u"]
        )

        residual_d = core.jnp.asarray(0.0, dtype=dtype)
        residual_c = core.jnp.asarray(0.0, dtype=dtype)
        stagger_error = np.inf
        diffusion_substeps = 0
        fatigue_q = core.jnp.ones_like(alpha_bar_start)
        hydrogen_q = core.jnp.ones_like(alpha_bar_start)
        for stagger in range(1, args.max_stagger + 1):
            old_d, old_c = block_state["d"], block_state["c"]
            psi, alpha, _, hydro_max = qois(
                block_state["u"], block_state["d"]
            )
            history_trial = core.jnp.maximum(history_start, psi)
            if args.diffusion_cycle_mode == "split":
                half_c, residual_c_min, steps_min = advance_concentration(
                    concentration_start,
                    hydro_min,
                    0.5 * elapsed_s,
                    block_state["d"],
                )
                block_state["c"], residual_c_max, steps_max = (
                    advance_concentration(
                        half_c,
                        hydro_max,
                        0.5 * elapsed_s,
                        block_state["d"],
                    )
                )
                residual_c = core.jnp.maximum(
                    residual_c_min, residual_c_max
                )
                diffusion_substeps = steps_min + steps_max
            else:
                block_state["c"], residual_c, diffusion_substeps = (
                    advance_concentration(
                        concentration_start,
                        0.5 * (hydro_min + hydro_max),
                        elapsed_s,
                        block_state["d"],
                    )
                )
            concentration_q = core.jnp.einsum(
                "qi,ei->eq",
                shape_values,
                block_state["c"][element_nodes],
            )
            block_state["hydrogen_memory"] = advance_memory(
                memory_start, concentration_q, elapsed_s
            )
            hydrogen_q = core.hydrogen_factor_from_wppm(
                block_state["hydrogen_memory"]
            )
            peak_seen_trial = core.jnp.maximum(peak_seen_start, alpha)
            alpha_n_scale = (
                hydrogen_q
                if args.hydrogen_scaled_alpha_n
                else core.jnp.ones_like(hydrogen_q)
            )
            fatigue_alpha = (
                core.jnp.maximum(alpha - alpha_min, 0.0)
                if args.fatigue_cycle_driving == "range"
                else alpha
            )
            alpha_bar_trial = alpha_bar_start + core.fatigue_increment(
                fatigue_alpha,
                peak_seen_trial,
                block_jump,
                args.load_ratio,
                args.fatigue_n,
                alpha_n_scale,
            )
            fatigue_q = core.fatigue_factor(
                alpha_bar_trial,
                args.alpha_0,
                args.fatigue_degradation_exponent,
            )
            trial_d, residual_d = solve_d(
                history_trial,
                fatigue_q,
                block_state["d"],
                hydrogen_q,
            )
            block_state["d"] = core.jnp.clip(
                core.jnp.maximum(trial_d, d_start), 0.0, 1.0
            )
            block_state["u"], residual_u_max = solve_u(
                block_state["d"],
                maximum_boundary,
                block_state["u"],
            )
            damage_error = core.jnp.linalg.norm(
                block_state["d"] - old_d
            ) / core.jnp.maximum(
                core.jnp.linalg.norm(block_state["d"]), 1.0e-30
            )
            concentration_error = core.jnp.linalg.norm(
                block_state["c"] - old_c
            ) / core.jnp.maximum(
                core.jnp.linalg.norm(block_state["c"]), 1.0e-30
            )
            stagger_error = float(
                core.jax.device_get(
                    core.jnp.maximum(damage_error, concentration_error)
                )
            )
            if stagger_error < args.stagger_tol:
                break
        block_state["history"] = history_trial
        block_state["alpha_bar"] = alpha_bar_trial
        block_state["peak_seen"] = peak_seen_trial

        residuals = [
            float(core.jax.device_get(value))
            for value in (
                residual_u_min,
                residual_u_max,
                residual_d,
                residual_c,
            )
        ]
        return block_state, {
            "residuals": residuals,
            "stagger": stagger,
            "stagger_error": stagger_error,
            "elapsed_s": elapsed_s,
            "diffusion_substeps": diffusion_substeps,
            "fatigue_q": fatigue_q,
            "hydrogen_q": hydrogen_q,
        }

    def combine_diagnostics(first: dict, second: dict):
        return {
            "residuals": [
                max(left, right)
                for left, right in zip(
                    first["residuals"], second["residuals"]
                )
            ],
            "stagger": first["stagger"] + second["stagger"],
            "stagger_error": max(
                first["stagger_error"], second["stagger_error"]
            ),
            "elapsed_s": first["elapsed_s"] + second["elapsed_s"],
            "diffusion_substeps": (
                first["diffusion_substeps"]
                + second["diffusion_substeps"]
            ),
            "fatigue_q": second["fatigue_q"],
            "hydrogen_q": second["hydrogen_q"],
        }

    print("=" * 78)
    print("Unified-model Q8 SENT numerical geometry-migration adapter")
    print(
        f"case={name}; {nnode} nodes; {len(elements)} Q8 elements; "
        f"f={args.frequency_hz:g} Hz; tau={args.hydrogen_memory_tau_s:g} s"
    )
    print(
        f"JAX {core.jax.__version__}; Python {platform.python_version()}; "
        f"devices={devices}"
    )
    print("NOT AN EXPERIMENTAL VALIDATION")
    print("=" * 78)

    cycle = 0
    adaptive_jump = args.cycle_jump
    accepted_blocks = 0
    rejected_blocks = 0
    stable_accepts = 0
    rows: list[dict] = []
    while cycle < args.max_cycles:
        jump = min(adaptive_jump, args.max_cycles - cycle)
        state_before = dict(state)
        initial_extension = crack_extension(
            nodes, np.asarray(core.jax.device_get(state_before["d"]))
        )
        full_state, full_diagnostics = advance_cycle_block(
            state_before, jump
        )
        trial_diagnostics = [full_diagnostics]
        block_error_ratio = 0.0
        crack_error_mm = 0.0
        phase_error = 0.0
        concentration_error = 0.0
        memory_error = 0.0
        trial_solves = 1

        if args.block_error_control and jump >= 2:
            first_jump = jump // 2
            second_jump = jump - first_jump
            half_state, first_diagnostics = advance_cycle_block(
                state_before, first_jump
            )
            candidate_state, second_diagnostics = advance_cycle_block(
                half_state, second_jump
            )
            diagnostics = combine_diagnostics(
                first_diagnostics, second_diagnostics
            )
            trial_diagnostics.extend(
                (first_diagnostics, second_diagnostics)
            )
            trial_solves = 3
            full_extension = crack_extension(
                nodes,
                np.asarray(core.jax.device_get(full_state["d"])),
            )
            candidate_extension = crack_extension(
                nodes,
                np.asarray(core.jax.device_get(candidate_state["d"])),
            )
            full_increment = full_extension - initial_extension
            candidate_increment = candidate_extension - initial_extension
            crack_error_mm = abs(full_increment - candidate_increment)
            phase_error = float(
                core.jax.device_get(
                    core.jnp.linalg.norm(
                        full_state["d"] - candidate_state["d"]
                    )
                    / core.jnp.maximum(
                        core.jnp.linalg.norm(candidate_state["d"]),
                        1.0e-30,
                    )
                )
            )
            concentration_error = float(
                core.jax.device_get(
                    core.jnp.linalg.norm(
                        full_state["c"] - candidate_state["c"]
                    )
                    / core.jnp.maximum(
                        core.jnp.linalg.norm(candidate_state["c"]),
                        1.0e-30,
                    )
                )
            )
            memory_error = float(
                core.jax.device_get(
                    core.jnp.linalg.norm(
                        full_state["hydrogen_memory"]
                        - candidate_state["hydrogen_memory"]
                    )
                    / core.jnp.maximum(
                        core.jnp.linalg.norm(
                            candidate_state["hydrogen_memory"]
                        ),
                        1.0e-30,
                    )
                )
            )
            ratios = [
                crack_error_mm
                / (
                    args.block_crack_increment_atol_mm
                    + args.block_crack_increment_rtol
                    * abs(candidate_increment)
                ),
                phase_error / args.block_phase_rtol,
                concentration_error / args.block_concentration_rtol,
            ]
            if args.hydrogen_memory_tau_s > 0.0:
                ratios.append(memory_error / args.block_memory_rtol)
            block_error_ratio = max(ratios)
            if not np.isfinite(block_error_ratio):
                block_error_ratio = np.inf
            state = candidate_state
        else:
            state, diagnostics = full_state, full_diagnostics

        for trial in trial_diagnostics:
            max_linear_residual = max(
                max_linear_residual, *trial["residuals"]
            )
            if not all(
                np.isfinite(value)
                and value <= args.linear_residual_tol
                for value in trial["residuals"]
            ):
                raise RuntimeError(
                    "linear residual limit exceeded: "
                    f"{trial['residuals']}"
                )

        maximum_damage_increment = float(
            core.jax.device_get(
                core.jnp.max(state["d"] - state_before["d"])
            )
        )
        error_reject = (
            args.block_error_control
            and jump >= 2
            and block_error_ratio > 1.0
        )
        convergence_reject = (
            maximum_damage_increment > args.max_damage_increment
            or diagnostics["stagger_error"] >= args.stagger_tol
        ) and not args.fixed_cycle_jump
        if (
            (error_reject or convergence_reject)
            and jump > args.min_cycle_jump
        ):
            state = state_before
            adaptive_jump = max(args.min_cycle_jump, jump // 2)
            rejected_blocks += 1
            stable_accepts = 0
            print(
                f"rollback N={cycle}+{jump}: "
                f"block_err={block_error_ratio:.3e}; "
                f"coupled_err={diagnostics['stagger_error']:.3e}; "
                f"retry={adaptive_jump}",
                flush=True,
            )
            continue
        if error_reject:
            raise RuntimeError(
                "block error exceeded tolerance at minimum jump: "
                f"{block_error_ratio:.3e}"
            )
        if diagnostics["stagger_error"] >= args.stagger_tol:
            raise RuntimeError(
                "coupled iteration failed at minimum jump: "
                f"{diagnostics['stagger_error']:.3e}"
            )

        cycle += jump
        accepted_blocks += 1
        if (
            not args.fixed_cycle_jump
            and maximum_damage_increment
            < 0.25 * args.max_damage_increment
        ):
            stable_accepts += 1
            if stable_accepts >= 4:
                adaptive_jump = min(
                    args.cycle_jump, max(jump + 1, 2 * jump)
                )
                stable_accepts = 0
        else:
            stable_accepts = 0

        damage_host = np.asarray(core.jax.device_get(state["d"]))
        concentration_host = np.asarray(core.jax.device_get(state["c"]))
        memory_host = np.asarray(
            core.jax.device_get(state["hydrogen_memory"])
        )
        extension = crack_extension(nodes, damage_host)
        row = {
            "cycle": cycle,
            "crack_extension_mm": extension,
            "maximum_phase_field": float(damage_host.max()),
            "maximum_hydrogen_wppm": float(
                concentration_host.max()
                * 1.0e6
                * core.M_H
                / core.RHO_M
            ),
            "maximum_hydrogen_memory_wppm": float(memory_host.max()),
            "cycle_jump": jump,
            "elapsed_block_s": diagnostics["elapsed_s"],
            "coupled_iterations": diagnostics["stagger"],
            "coupled_error": diagnostics["stagger_error"],
            "maximum_damage_increment": maximum_damage_increment,
            "block_error_ratio": block_error_ratio,
            "block_crack_increment_error_mm": crack_error_mm,
            "block_phase_error": phase_error,
            "block_concentration_error": concentration_error,
            "block_hydrogen_memory_error": memory_error,
            "block_trial_solves": trial_solves,
            "maximum_linear_residual": max(
                diagnostics["residuals"]
            ),
            "wall_s": time.perf_counter() - total_started,
        }
        rows.append(row)
        write_rows(outdir / "results.csv", rows)
        if accepted_blocks % args.print_every == 0:
            print(
                f"N={cycle:5d} da={extension:.6f} mm "
                f"jump={jump:4d} block_err={block_error_ratio:.2e} "
                f"linear={row['maximum_linear_residual']:.2e}",
                flush=True,
            )

    np.savez_compressed(
        outdir / "final_state.npz",
        nodes=nodes,
        elements=elements,
        displacement=np.asarray(core.jax.device_get(state["u"])).reshape(
            -1, 2
        ),
        phase_field=np.asarray(core.jax.device_get(state["d"])),
        concentration_mol_mm3=np.asarray(
            core.jax.device_get(state["c"])
        ),
        hydrogen_memory_wppm=np.asarray(
            core.jax.device_get(state["hydrogen_memory"])
        ),
        alpha_bar=np.asarray(core.jax.device_get(state["alpha_bar"])),
    )
    wall_s = time.perf_counter() - total_started
    summary = {
        "version": VERSION,
        "purpose": "Q8 SENT numerical geometry-migration example",
        "validation_status": (
            "not experimental validation; no SENT data were fitted or compared"
        ),
        "case": name,
        "source_files": list(SOURCE_FILES),
        "command": [sys.executable, *sys.argv],
        "mesh": str(mesh_path),
        "nodes": nnode,
        "q8_elements": len(elements),
        "integration": args.integration,
        "device": str(devices[0]),
        "linear_backend": args.linear_backend,
        "frequency_hz": args.frequency_hz,
        "cycles": cycle,
        "accepted_block_count": accepted_blocks,
        "rejected_block_count": rejected_blocks,
        "block_control": {
            "fixed_cycle_jump": args.fixed_cycle_jump,
            "block_error_control": args.block_error_control,
            "requested_cycle_jump": args.cycle_jump,
            "minimum_cycle_jump": args.min_cycle_jump,
            "crack_increment_rtol": args.block_crack_increment_rtol,
            "crack_increment_atol_mm": (
                args.block_crack_increment_atol_mm
            ),
            "phase_rtol": args.block_phase_rtol,
            "concentration_rtol": args.block_concentration_rtol,
            "memory_rtol": args.block_memory_rtol,
        },
        "coupling_and_linear_control": {
            "maximum_damage_increment": args.max_damage_increment,
            "maximum_stagger_iterations": args.max_stagger,
            "stagger_relative_tolerance": args.stagger_tol,
            "iterative_linear_tolerance": args.linear_tol,
            "iterative_linear_max_iterations": args.linear_maxiter,
            "reported_linear_residual_limit": args.linear_residual_tol,
        },
        "hydrogen": {
            "environment_wppm": case_wppm,
            "memory_tau_s": args.hydrogen_memory_tau_s,
            "diffusion_time_mapping": "Delta t = Delta N / frequency",
        },
        "material_and_regularisation": {
            "E_mpa": core.E,
            "nu": core.NU,
            "Gc_n_per_mm": core.GC,
            "ell_mm": core.LENGTH_SCALE,
            "D_mm2_s": core.DIFFUSIVITY,
            "note": (
                "Cracked-square Gc, ell and diffusivity are retained so the "
                "existing SENT mesh remains numerically resolved."
            ),
        },
        "maximum_linear_residual": max_linear_residual,
        "wall_s": wall_s,
    }
    (outdir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"completed numerical SENT case: {outdir}", flush=True)


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    validate_args(args)
    selected = args.case or ["air"]
    if "all" in selected:
        selected = ["air", "hydrogen"]
    for name in dict.fromkeys(selected):
        run_case(name, args)


if __name__ == "__main__":
    main()
