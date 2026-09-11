"""Independent P1-FEM check of the released harmonic HAZ mapping field."""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from audit_released_mesh import MESH, OUT as MESH_OUT, parse_abaqus


OUT = MESH_OUT.parent / "haz_map"


def triangles_from_elements(elements: dict[int, tuple[int, ...]]) -> list[tuple[int, int, int]]:
    triangles: list[tuple[int, int, int]] = []
    for connectivity in elements.values():
        if len(connectivity) == 3:
            triangles.append(connectivity)
        elif len(connectivity) == 4:
            triangles.extend(
                [
                    (connectivity[0], connectivity[1], connectivity[2]),
                    (connectivity[0], connectivity[2], connectivity[3]),
                ]
            )
        else:
            raise ValueError(f"Unsupported element with {len(connectivity)} nodes")
    return triangles


def main() -> None:
    nodes, elements, _, node_sets, _ = parse_abaqus(MESH)
    node_ids = np.array(sorted(nodes))
    index = {node_id: position for position, node_id in enumerate(node_ids)}
    xy = np.array([nodes[node_id] for node_id in node_ids])
    triangles_by_id = triangles_from_elements(elements)
    triangles = np.array([[index[node_id] for node_id in tri] for tri in triangles_by_id])

    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    for triangle in triangles:
        points = xy[triangle]
        signed_double_area = np.linalg.det(
            np.array(
                [
                    [1.0, points[0, 0], points[0, 1]],
                    [1.0, points[1, 0], points[1, 1]],
                    [1.0, points[2, 0], points[2, 1]],
                ]
            )
        )
        area = abs(signed_double_area) / 2.0
        if area <= 0:
            raise ValueError("Degenerate triangle in released mesh")
        b = np.array(
            [
                points[1, 1] - points[2, 1],
                points[2, 1] - points[0, 1],
                points[0, 1] - points[1, 1],
            ]
        )
        c = np.array(
            [
                points[2, 0] - points[1, 0],
                points[0, 0] - points[2, 0],
                points[1, 0] - points[0, 0],
            ]
        )
        local = (np.outer(b, b) + np.outer(c, c)) / (4.0 * area)
        for local_row, global_row in enumerate(triangle):
            for local_column, global_column in enumerate(triangle):
                rows.append(int(global_row))
                columns.append(int(global_column))
                values.append(float(local[local_row, local_column]))

    stiffness = coo_matrix(
        (values, (rows, columns)), shape=(len(node_ids), len(node_ids))
    ).tocsr()

    # The released deck uses the inverse of the paper's convention:
    # eta_deck=1 in BM and eta_deck=0 in WM.
    base_nodes = set(node_sets["base"])
    weld_nodes = set(node_sets["weld1"]) | set(node_sets["weld2"])
    conflicting = base_nodes & weld_nodes
    if conflicting:
        raise ValueError(f"BM/WM Dirichlet sets overlap at {len(conflicting)} nodes")

    fixed_ids = base_nodes | weld_nodes
    fixed = np.array(sorted(index[node_id] for node_id in fixed_ids), dtype=int)
    free = np.setdiff1d(np.arange(len(node_ids)), fixed)
    eta_deck = np.zeros(len(node_ids))
    eta_deck[[index[node_id] for node_id in base_nodes]] = 1.0
    right_hand_side = -(stiffness[free][:, fixed] @ eta_deck[fixed])
    eta_deck[free] = spsolve(stiffness[free][:, free], right_hand_side)
    residual = stiffness @ eta_deck
    free_residual = residual[free]
    raw_eta_min = float(eta_deck.min())
    raw_eta_max = float(eta_deck.max())
    eta_deck = np.clip(eta_deck, 0.0, 1.0)
    eta_paper = 1.0 - eta_deck

    bm = {"Gc": 33.0, "n": 2.85, "alpha_T": 60.0, "D_eff": 4.5e-4}
    wm = {"Gc": 50.0, "n": 2.95, "alpha_T": 15.0, "D_eff": 3.0e-4}
    fields = {
        name: bm[name] + (wm[name] - bm[name]) * eta_paper
        for name in bm
    }

    summary = {
        "method": "independent P1 finite-element solve; released quads split into triangles",
        "node_count": len(node_ids),
        "triangle_count_after_split": len(triangles),
        "fixed_BM_nodes": len(base_nodes),
        "fixed_WM_nodes": len(weld_nodes),
        "free_HAZ_nodes": len(free),
        "deck_eta": {
            "minimum": float(eta_deck.min()),
            "maximum": float(eta_deck.max()),
            "raw_minimum_before_roundoff_clip": raw_eta_min,
            "raw_maximum_before_roundoff_clip": raw_eta_max,
            "free_residual_linf": float(np.max(np.abs(free_residual))),
        },
        "paper_eta": {
            "minimum": float(eta_paper.min()),
            "maximum": float(eta_paper.max()),
            "convention": "0 on BM, 1 on WM",
        },
        "mapped_field_ranges": {
            name: {"minimum": float(field.min()), "maximum": float(field.max())}
            for name, field in fields.items()
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "haz_map_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        OUT / "haz_map_fields.npz",
        node_ids=node_ids,
        xy=xy,
        triangles=triangles,
        eta_deck=eta_deck,
        eta_paper=eta_paper,
        **fields,
    )

    triangulation = mtri.Triangulation(xy[:, 0], xy[:, 1], triangles)
    plot_fields = [
        ("$\\eta_\\mathrm{HAZ}$", eta_paper, (0.0, 1.0)),
        ("$G_c$ (N/mm)", fields["Gc"], (33.0, 50.0)),
        ("$n$", fields["n"], (2.85, 2.95)),
        ("$\\alpha_T$ (MPa)", fields["alpha_T"], (15.0, 60.0)),
        ("$D_\\mathrm{eff}$ (mm$^2$/s)", fields["D_eff"], (3.0e-4, 4.5e-4)),
    ]
    fig, axes = plt.subplots(
        len(plot_fields), 1, figsize=(10.2, 10.8), constrained_layout=True
    )
    for axis, (label, field, limits) in zip(axes, plot_fields, strict=True):
        upper_plot_limit = np.nextafter(limits[1], np.inf)
        contour = axis.tricontourf(
            triangulation,
            field,
            levels=np.linspace(limits[0], upper_plot_limit, 21),
            cmap="viridis",
            extend="neither",
        )
        axis.triplot(triangulation, color="white", linewidth=0.035, alpha=0.18)
        axis.set_xlim(-31, 31)
        axis.set_ylim(332, 357)
        axis.set_aspect("equal")
        axis.set_ylabel("$y$ (mm)")
        axis.set_title(label, loc="left")
        fig.colorbar(contour, ax=axis, pad=0.01, aspect=18)
    axes[-1].set_xlabel("$x$ (mm)")
    fig.suptitle("Independent harmonic HAZ-map verification (paper convention)")
    fig.savefig(OUT / "haz_map_and_properties.png", dpi=260)
    fig.savefig(OUT / "haz_map_and_properties.pdf")
    plt.close(fig)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
