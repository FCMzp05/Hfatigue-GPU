#!/usr/bin/env python3
"""GPU reproduction of Yang et al. (2026) JMRT welded-pipeline case.

The driver uses the repository's JAX/P1/cuDSS phase-field solver, not MOOSE.
Released Q4 elements are split deterministically into P1 triangles, while the
released TRI3 elements are retained. Units are mm, N, MPa, s, and mol.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[3]
YANG_DIR = REPO / "code" / "reproductions" / "yang2026"
if str(YANG_DIR) not in sys.path:
    sys.path.insert(0, str(YANG_DIR))

import yang2026_fig4_sent_p1_cudss as yang  # noqa: E402

from audit_released_mesh import MESH, parse_abaqus  # noqa: E402


HAZ_FIELDS = (
    REPO
    / "outputs"
    / "yang_2026_jmrt_pipeline"
    / "haz_map"
    / "haz_map_fields.npz"
)
DEFAULT_OUT = (
    REPO
    / "outputs"
    / "yang_2026_jmrt_pipeline_gpu"
    / "fig14"
    / "p10"
)
VERSION = "yang2026_jmrt_pipeline_p1_cudss_v1"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--pressure-mpa", type=float, default=10.0)
    result.add_argument("--blocks", type=int, default=82)
    result.add_argument("--delta-n", type=int, default=100)
    result.add_argument("--uniform-refine", type=int, default=0)
    result.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    result.add_argument(
        "--restart-state",
        type=Path,
        help="state.npz from a matching mesh and pressure",
    )
    result.add_argument(
        "--start-cycle",
        type=int,
        default=0,
        help="physical cycle represented by --restart-state",
    )
    result.add_argument("--mesh-check", action="store_true")
    result.add_argument("--max-stagger", type=int, default=20)
    result.add_argument("--stagger-tol", type=float, default=1.0e-6)
    result.add_argument("--linear-residual-tol", type=float, default=1.0e-4)
    result.add_argument("--phase-field-residual-tol", type=float, default=1.0e-3)
    result.add_argument(
        "--stop-at-max-d",
        type=float,
        default=0.0,
        help="stop once max(d) reaches this value; zero disables early stopping",
    )
    result.add_argument(
        "--time-mode",
        choices=("source", "physical"),
        default="source",
        help="source uses the verified 1e7 diffusivity acceleration per block",
    )
    result.add_argument(
        "--snapshot-cycles",
        type=int,
        nargs="*",
        default=(6000, 7000, 8100, 8200),
    )
    result.add_argument("--save-state", action="store_true")
    return result


def boundary_edges(
    elements: dict[int, tuple[int, ...]], boundary: set[int]
) -> list[tuple[int, int]]:
    found: set[tuple[int, int]] = set()
    for connectivity in elements.values():
        for first, second in zip(
            connectivity, connectivity[1:] + connectivity[:1], strict=True
        ):
            if first in boundary and second in boundary:
                found.add(tuple(sorted((first, second))))
    return sorted(found)


def load_pipeline():
    nodes_by_id, elements, _, node_sets, _ = parse_abaqus(MESH)
    with np.load(HAZ_FIELDS) as fields:
        node_ids = np.asarray(fields["node_ids"], dtype=np.int64)
        nodes = np.asarray(fields["xy"], dtype=np.float64)
        triangles = np.asarray(fields["triangles"], dtype=np.int64)
        nodal_fields = {
            name: np.asarray(fields[name], dtype=np.float64)
            for name in ("Gc", "n", "alpha_T", "D_eff", "eta_paper")
        }
    index = {int(node_id): position for position, node_id in enumerate(node_ids)}
    expected = np.asarray([nodes_by_id[int(node_id)] for node_id in node_ids])
    if not np.allclose(nodes, expected, rtol=0, atol=1.0e-12):
        raise RuntimeError("HAZ map and released mesh coordinates do not match")

    xy = nodes[triangles]
    twice_area = (
        (xy[:, 1, 0] - xy[:, 0, 0]) * (xy[:, 2, 1] - xy[:, 0, 1])
        - (xy[:, 2, 0] - xy[:, 0, 0]) * (xy[:, 1, 1] - xy[:, 0, 1])
    )
    flip = twice_area < 0
    triangles[flip] = triangles[flip][:, [0, 2, 1]]
    areas = np.abs(twice_area) / 2
    if np.any(areas <= 0):
        raise RuntimeError("pipeline triangulation contains non-positive areas")

    groups = {
        name: np.asarray(
            sorted(index[node_id] for node_id in node_sets[name]), dtype=np.int64
        )
        for name in ("internal", "left", "right", "base")
    }
    boundaries = {
        name: [
            (index[first], index[second])
            for first, second in boundary_edges(elements, node_sets[name])
        ]
        for name in ("internal", "left", "right")
    }
    if not boundaries["internal"]:
        raise RuntimeError("no internal-pressure boundary edges were recovered")

    return nodes, triangles, areas, groups, boundaries, nodal_fields


def refine_once(
    nodes: np.ndarray,
    triangles: np.ndarray,
    groups: dict[str, np.ndarray],
    boundaries: dict[str, list[tuple[int, int]]],
    nodal_fields: dict[str, np.ndarray],
):
    raw_edges = np.concatenate(
        (
            triangles[:, (0, 1)],
            triangles[:, (1, 2)],
            triangles[:, (2, 0)],
        )
    )
    sorted_edges = np.sort(raw_edges, axis=1)
    unique_edges, inverse = np.unique(
        sorted_edges, axis=0, return_inverse=True
    )
    midpoint_ids = np.arange(
        len(nodes), len(nodes) + len(unique_edges), dtype=np.int64
    )
    edge_to_midpoint = {
        tuple(map(int, edge)): int(midpoint)
        for edge, midpoint in zip(
            unique_edges, midpoint_ids, strict=True
        )
    }
    midpoint_coordinates = nodes[unique_edges].mean(axis=1)
    refined_nodes = np.vstack((nodes, midpoint_coordinates))
    refined_fields = {
        name: np.concatenate((values, values[unique_edges].mean(axis=1)))
        for name, values in nodal_fields.items()
    }

    count = len(triangles)
    mab = midpoint_ids[inverse[:count]]
    mbc = midpoint_ids[inverse[count : 2 * count]]
    mca = midpoint_ids[inverse[2 * count :]]
    a, b, c = triangles.T
    refined_triangles = np.concatenate(
        (
            np.column_stack((a, mab, mca)),
            np.column_stack((mab, b, mbc)),
            np.column_stack((mca, mbc, c)),
            np.column_stack((mab, mbc, mca)),
        )
    )

    refined_groups: dict[str, np.ndarray] = {}
    for name, values in groups.items():
        members = np.zeros(len(nodes), dtype=bool)
        members[values] = True
        midpoint_members = members[unique_edges].all(axis=1)
        refined_groups[name] = np.concatenate(
            (values, midpoint_ids[midpoint_members])
        )

    refined_boundaries: dict[str, list[tuple[int, int]]] = {}
    for name, edges in boundaries.items():
        split_edges: list[tuple[int, int]] = []
        for first, second in edges:
            midpoint = edge_to_midpoint[tuple(sorted((first, second)))]
            split_edges.extend(((first, midpoint), (midpoint, second)))
        refined_boundaries[name] = split_edges

    xy = refined_nodes[refined_triangles]
    twice_area = (
        (xy[:, 1, 0] - xy[:, 0, 0]) * (xy[:, 2, 1] - xy[:, 0, 1])
        - (xy[:, 2, 0] - xy[:, 0, 0]) * (xy[:, 1, 1] - xy[:, 0, 1])
    )
    if np.any(twice_area <= 0):
        raise RuntimeError("uniform refinement produced non-positive areas")
    return (
        refined_nodes,
        refined_triangles,
        twice_area / 2,
        refined_groups,
        refined_boundaries,
        refined_fields,
    )


def pressure_force(
    nodes: np.ndarray, edges: list[tuple[int, int]], pressure: float
) -> np.ndarray:
    force = np.zeros(2 * len(nodes), dtype=np.float64)
    for first, second in edges:
        start, end = nodes[first], nodes[second]
        tangent = end - start
        length = float(np.linalg.norm(tangent))
        normal = np.array((tangent[1], -tangent[0])) / length
        midpoint = 0.5 * (start + end)
        if np.dot(normal, midpoint) < 0:
            normal *= -1
        nodal = 0.5 * pressure * length * normal
        force[2 * first : 2 * first + 2] += nodal
        force[2 * second : 2 * second + 2] += nodal
    return force


def outer_surface_nodes(
    triangles: np.ndarray,
    boundaries: dict[str, list[tuple[int, int]]],
) -> np.ndarray:
    raw_edges = np.sort(
        np.concatenate(
            (
                triangles[:, (0, 1)],
                triangles[:, (1, 2)],
                triangles[:, (2, 0)],
            )
        ),
        axis=1,
    )
    unique_edges, counts = np.unique(raw_edges, axis=0, return_counts=True)
    exterior = unique_edges[counts == 1]
    excluded = {
        tuple(sorted(edge))
        for name in ("internal", "left", "right")
        for edge in boundaries[name]
    }
    outer_edges = [
        edge for edge in exterior if tuple(map(int, edge)) not in excluded
    ]
    if not outer_edges:
        raise RuntimeError("no external pipe surface was recovered")
    return np.unique(np.asarray(outer_edges, dtype=np.int64))


def penetration_mm(
    nodes: np.ndarray, damage: np.ndarray, outer_nodes: np.ndarray
) -> float:
    broken = damage >= 0.95
    if not np.any(broken):
        return 0.0
    radii = np.linalg.norm(nodes, axis=1)
    angles = np.arctan2(nodes[:, 1], nodes[:, 0])
    order = np.argsort(angles[outer_nodes])
    surface_angles = angles[outer_nodes][order]
    surface_radii = radii[outer_nodes][order]
    local_outer = np.interp(
        angles[broken], surface_angles, surface_radii
    )
    return max(0.0, float(np.max(local_outer - radii[broken])))


def main() -> None:
    args = parser().parse_args()
    if (
        args.pressure_mpa <= 0
        or args.blocks < 1
        or args.delta_n < 1
        or args.uniform_refine < 0
        or args.start_cycle < 0
    ):
        raise ValueError("pressure, blocks, and delta-n must be positive")
    if args.start_cycle and args.restart_state is None:
        raise ValueError("--start-cycle requires --restart-state")

    nodes, tri, areas, groups, boundaries, nodal_fields = load_pipeline()
    for _ in range(args.uniform_refine):
        (
            nodes,
            tri,
            areas,
            groups,
            boundaries,
            nodal_fields,
        ) = refine_once(
            nodes, tri, groups, boundaries, nodal_fields
        )
    fields = {
        name: values[tri].mean(axis=1)
        for name, values in nodal_fields.items()
    }
    edges = boundaries["internal"]
    outer_nodes = outer_surface_nodes(tri, boundaries)
    mesh_summary = {
        "nodes": len(nodes),
        "triangles": len(tri),
        "uniform_refine": args.uniform_refine,
        "internal_nodes": len(groups["internal"]),
        "internal_edges": len(edges),
        "outer_surface_nodes": len(outer_nodes),
        "area_mm2": float(areas.sum()),
        "Gc_range": [float(fields["Gc"].min()), float(fields["Gc"].max())],
        "alpha_T_range": [
            float(fields["alpha_T"].min()),
            float(fields["alpha_T"].max()),
        ],
        "n_range": [float(fields["n"].min()), float(fields["n"].max())],
        "D_eff_range_mm2_s": [
            float(fields["D_eff"].min()),
            float(fields["D_eff"].max()),
        ],
    }
    if args.mesh_check:
        print(json.dumps(mesh_summary, indent=2))
        return

    yang.load_runtime()
    yang.jax.config.update("jax_enable_x64", True)
    devices = yang.jax.devices()
    if not any(device.platform == "gpu" for device in devices):
        raise RuntimeError(f"GPU required, available devices: {devices}")
    dtype = yang.jnp.float64
    yang.L0 = 0.2
    yang.LOAD_RATIO = 0.5
    yang.FATIGUE_N = 2.85
    yang.TEMPERATURE = 295.0

    bmat, dofs, mass, grad = yang.preprocess(nodes, tri, areas)
    ndof = 2 * len(nodes)
    fixed_nodes = np.unique(np.concatenate((groups["left"], groups["right"])))
    fixed_dofs = np.column_stack(
        (2 * fixed_nodes, 2 * fixed_nodes + 1)
    ).reshape(-1)
    free_u = np.ones(ndof, dtype=np.float64)
    free_u[fixed_dofs] = 0
    external_force = pressure_force(nodes, edges, args.pressure_mpa)
    boundary = yang.jnp.zeros(ndof, dtype=dtype)

    c_wppm = 0.243 * np.sqrt(args.pressure_mpa / 10.0)
    c_environment = c_wppm * yang.C_ENV
    if args.time_mode == "source":
        diffusion_dt = 1.0
        diffusivity = fields["D_eff"] * 1.0e7
    else:
        diffusion_dt = args.delta_n / 1.0e-5
        diffusivity = fields["D_eff"]

    bridge = yang.FixedTopologyCudssBridge()
    bridge.install(yang)
    try:
        (
            solve_u,
            qois,
            fatigue_factor,
            solve_d,
            solve_c,
            hydrogen_factor,
        ) = yang.make_solvers(
            nodes,
            tri,
            areas,
            bmat,
            dofs,
            mass,
            grad,
            free_u,
            "felino_stress",
            dtype,
            1.0e-7,
            5000,
            diffusivity,
            fields["Gc"],
            0.89,
            c_environment,
            3.0e4,
            external_force_np=external_force,
            hydrogen_law="jmrt_pipeline",
            fatigue_energy="mean_load",
            multiply_by_degradation=False,
            alpha_t_cell=fields["alpha_T"],
            fatigue_n_cell=fields["n"],
            damage_zero_ids=groups["base"],
        )

        outdir = args.outdir.resolve()
        outdir.mkdir(parents=True, exist_ok=True)
        snapshots = outdir / "snapshots"
        snapshot_cycles = set(map(int, args.snapshot_cycles))
        fields_csv = (
            "block",
            "cycle",
            "max_damage",
            "crack_area_mm2",
            "penetration_mm",
            "max_alpha_bar_mpa",
            "min_fatigue_factor",
            "min_hydrogen_factor",
            "max_concentration_mol_mm3",
            "min_concentration_mol_mm3",
            "max_hydrostatic_stress_mpa",
            "min_hydrostatic_stress_mpa",
            "stagger_iterations",
            "stagger_residual",
            "u_residual",
            "d_residual",
            "c_residual",
            "block_wall_s",
            "cumulative_wall_s",
        )

        if args.restart_state is None:
            u = yang.jnp.zeros(ndof, dtype=dtype)
            damage = yang.jnp.zeros(len(nodes), dtype=dtype)
            history = yang.jnp.zeros(len(tri), dtype=dtype)
            alpha_bar = yang.jnp.zeros(len(tri), dtype=dtype)
            concentration = yang.jnp.zeros(len(nodes), dtype=dtype)
        else:
            restart_path = args.restart_state.resolve()
            with np.load(restart_path) as restart:
                restart_nodes = np.asarray(restart["nodes"])
                restart_triangles = np.asarray(restart["triangles"])
                if (
                    restart_nodes.shape != nodes.shape
                    or not np.allclose(
                        restart_nodes, nodes, rtol=0, atol=1.0e-12
                    )
                    or restart_triangles.shape != tri.shape
                    or not np.array_equal(restart_triangles, tri)
                ):
                    raise ValueError(
                        "restart state does not match the requested mesh"
                    )
                u = yang.jnp.asarray(
                    np.asarray(restart["displacement"]).reshape(-1),
                    dtype=dtype,
                )
                damage = yang.jnp.asarray(restart["damage"], dtype=dtype)
                history = yang.jnp.asarray(restart["history"], dtype=dtype)
                alpha_bar = yang.jnp.asarray(
                    restart["alpha_bar"], dtype=dtype
                )
                concentration = yang.jnp.asarray(
                    restart["concentration"], dtype=dtype
                )
        initial_damage_max = float(
            np.asarray(yang.jax.device_get(damage)).max()
        )
        initial_hydrogen_max = float(
            np.asarray(yang.jax.device_get(concentration)).max()
        )
        rows: list[dict[str, float | int]] = []
        start = time.perf_counter()

        with (outdir / "results.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=fields_csv)
            writer.writeheader()
            for block in range(1, args.blocks + 1):
                tic = time.perf_counter()
                damage_start = damage
                alpha_start = alpha_bar
                concentration_start = concentration
                history_start = history

                u, ur, _ = solve_u(damage, boundary, u)
                psi, psi_eff, _, sigma_h = qois(u, damage)
                alpha_trial = alpha_start + args.delta_n * psi_eff
                fatigue_trial = fatigue_factor(alpha_trial)
                concentration_trial, cr = solve_c(
                    concentration_start,
                    sigma_h,
                    damage,
                    diffusion_dt,
                    groups["internal"],
                    1,
                )
                hydrogen_trial = hydrogen_factor(concentration_trial)
                factor = fatigue_trial * hydrogen_trial
                history_trial = yang.jnp.maximum(history_start, psi)
                stagger_residual = float("inf")

                for stagger in range(1, args.max_stagger + 1):
                    old_damage = damage
                    old_alpha = alpha_trial
                    old_concentration = concentration_trial
                    damage, dr, _ = solve_d(
                        history_trial, factor, damage_start
                    )
                    damage = yang.jnp.maximum(damage_start, damage)
                    u, ur, _ = solve_u(damage, boundary, u)
                    psi, psi_eff, crack_area, sigma_h = qois(u, damage)
                    alpha_trial = alpha_start + args.delta_n * psi_eff
                    fatigue_trial = fatigue_factor(alpha_trial)
                    concentration_trial, cr = solve_c(
                        concentration_start,
                        sigma_h,
                        damage,
                        diffusion_dt,
                        groups["internal"],
                        1,
                    )
                    hydrogen_trial = hydrogen_factor(concentration_trial)
                    factor = fatigue_trial * hydrogen_trial
                    history_trial = yang.jnp.maximum(history_start, psi)
                    residuals = (
                        yang.jnp.linalg.norm(damage - old_damage)
                        / yang.jnp.maximum(
                            yang.jnp.linalg.norm(damage), 1.0e-30
                        ),
                        yang.jnp.linalg.norm(alpha_trial - old_alpha)
                        / yang.jnp.maximum(
                            yang.jnp.linalg.norm(alpha_trial), 1.0e-30
                        ),
                        yang.jnp.linalg.norm(
                            concentration_trial - old_concentration
                        )
                        / yang.jnp.maximum(
                            yang.jnp.linalg.norm(concentration_trial), 1.0e-30
                        ),
                    )
                    stagger_residual = float(
                        yang.jax.device_get(
                            yang.jnp.maximum(
                                yang.jnp.maximum(residuals[0], residuals[1]),
                                residuals[2],
                            )
                        )
                    )
                    if stagger_residual < args.stagger_tol:
                        break

                alpha_bar = alpha_trial
                concentration = concentration_trial
                history = history_trial
                urf, drf, crf = (
                    float(yang.jax.device_get(value))
                    for value in (ur, dr, cr)
                )
                if (
                    not np.isfinite(stagger_residual)
                    or max(urf, crf) > args.linear_residual_tol
                    or drf > args.phase_field_residual_tol
                ):
                    raise RuntimeError(
                        f"residual failure at block {block}: "
                        f"stagger={stagger_residual:.3e}, u={urf:.3e}, "
                        f"d={drf:.3e}, c={crf:.3e}"
                    )

                damage_np = np.asarray(yang.jax.device_get(damage))
                alpha_np = np.asarray(yang.jax.device_get(alpha_bar))
                fatigue_np = np.asarray(yang.jax.device_get(fatigue_trial))
                hydrogen_np = np.asarray(yang.jax.device_get(hydrogen_trial))
                concentration_np = np.asarray(
                    yang.jax.device_get(concentration)
                )
                sigma_np = np.asarray(yang.jax.device_get(sigma_h))
                cumulative = time.perf_counter() - start
                cycle = args.start_cycle + block * args.delta_n
                row = {
                    "block": block,
                    "cycle": cycle,
                    "max_damage": float(damage_np.max()),
                    "crack_area_mm2": float(yang.jax.device_get(crack_area)),
                    "penetration_mm": penetration_mm(
                        nodes, damage_np, outer_nodes
                    ),
                    "max_alpha_bar_mpa": float(alpha_np.max()),
                    "min_fatigue_factor": float(fatigue_np.min()),
                    "min_hydrogen_factor": float(hydrogen_np.min()),
                    "max_concentration_mol_mm3": float(
                        concentration_np.max()
                    ),
                    "min_concentration_mol_mm3": float(
                        concentration_np.min()
                    ),
                    "max_hydrostatic_stress_mpa": float(sigma_np.max()),
                    "min_hydrostatic_stress_mpa": float(sigma_np.min()),
                    "stagger_iterations": stagger,
                    "stagger_residual": stagger_residual,
                    "u_residual": urf,
                    "d_residual": drf,
                    "c_residual": crf,
                    "block_wall_s": time.perf_counter() - tic,
                    "cumulative_wall_s": cumulative,
                }
                writer.writerow(row)
                stream.flush()
                rows.append(row)

                if cycle in snapshot_cycles:
                    snapshots.mkdir(exist_ok=True)
                    np.savez_compressed(
                        snapshots / f"cycle{cycle:05d}.npz",
                        nodes=nodes,
                        triangles=tri,
                        displacement=np.asarray(
                            yang.jax.device_get(u)
                        ).reshape((-1, 2)),
                        damage=damage_np,
                        history=np.asarray(
                            yang.jax.device_get(history)
                        ),
                        alpha_bar=alpha_np,
                        concentration=concentration_np,
                        hydrostatic_stress=sigma_np,
                        eta_haz=fields["eta_paper"],
                        cycle=cycle,
                    )
                eta = cumulative / block * (args.blocks - block)
                print(
                    f"block={block:3d}/{args.blocks} cycle={cycle:5d} "
                    f"maxd={row['max_damage']:.5f} "
                    f"pen={row['penetration_mm']:.3f} mm "
                    f"res={stagger_residual:.2e}/{urf:.2e}/{drf:.2e} "
                    f"wall={row['block_wall_s']:.2f}s ETA={eta:.1f}s",
                    flush=True,
                )
                if (
                    args.stop_at_max_d > 0
                    and row["max_damage"] >= args.stop_at_max_d
                ):
                    print(
                        f"stopping at cycle {cycle}: "
                        f"max(d)={row['max_damage']:.6f}",
                        flush=True,
                    )
                    break

        if args.save_state:
            np.savez_compressed(
                outdir / "state.npz",
                nodes=nodes,
                triangles=tri,
                displacement=np.asarray(yang.jax.device_get(u)).reshape((-1, 2)),
                damage=np.asarray(yang.jax.device_get(damage)),
                history=np.asarray(yang.jax.device_get(history)),
                alpha_bar=np.asarray(yang.jax.device_get(alpha_bar)),
                concentration=np.asarray(yang.jax.device_get(concentration)),
                cycle=np.asarray(
                    args.start_cycle + len(rows) * args.delta_n
                ),
            )
        summary = {
            "version": VERSION,
            "solver": "repository JAX/P1 solver with NVIDIA cuDSS bridge",
            "device": str(devices[0]),
            "mesh": mesh_summary,
            "pressure_mpa": args.pressure_mpa,
            "environment_wppm": c_wppm,
            "restart_state": (
                str(args.restart_state.resolve())
                if args.restart_state is not None
                else None
            ),
            "start_cycle": args.start_cycle,
            "blocks_completed": len(rows),
            "cycles_completed": rows[-1]["cycle"],
            "time_mode": args.time_mode,
            "diffusion_dt_s": diffusion_dt,
            "diffusivity_acceleration": (
                1.0e7 if args.time_mode == "source" else 1.0
            ),
            "boundary_conditions": (
                "consistent vector pressure on internal edges; "
                "released left/right node sets clamped"
            ),
            "initial_damage": initial_damage_max,
            "initial_hydrogen": initial_hydrogen_max,
            "final": rows[-1],
            "wall_s": time.perf_counter() - start,
            "cudss": bridge.stats(),
        }
        (outdir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        print(f"completed: {outdir}", flush=True)
    finally:
        bridge.close()


if __name__ == "__main__":
    main()
