"""Audit and plot the released Yang et al. (2026) JMRT pipeline mesh."""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[3]
MESH = (
    REPO
    / "open_source_code"
    / "phase-field-hydrogen-fatigue"
    / "phase-field-hydrogen-fatigue-main"
    / "03_JMRT_2026"
    / "examples"
    / "weld5.inp"
)
OUT = REPO / "outputs" / "yang_2026_jmrt_pipeline" / "mesh_audit"

REGIONS = ("base", "HAZ1", "HAZ2", "weld1", "weld2")
COLORS = {
    "base": "#d8dee9",
    "HAZ1": "#f0c36e",
    "HAZ2": "#e69f5b",
    "weld1": "#5e81ac",
    "weld2": "#81a1c1",
    "unassigned": "#bf616a",
}


def _options(header: str) -> dict[str, str | bool]:
    result: dict[str, str | bool] = {}
    for token in header[1:].split(",")[1:]:
        token = token.strip()
        if "=" in token:
            key, value = token.split("=", 1)
            result[key.strip().lower()] = value.strip()
        elif token:
            result[token.lower()] = True
    return result


def parse_abaqus(path: Path):
    nodes: dict[int, tuple[float, float]] = {}
    elements: dict[int, tuple[int, ...]] = {}
    element_types: dict[int, str] = {}
    node_sets: dict[str, set[int]] = defaultdict(set)
    element_sets: dict[str, set[int]] = defaultdict(set)

    section = ""
    set_name = ""
    generate = False
    element_type = ""

    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("**"):
            continue
        if line.startswith("*"):
            keyword = line[1:].split(",", 1)[0].strip().lower()
            opts = _options(line)
            section = keyword
            set_name = ""
            generate = "generate" in opts
            if keyword == "element":
                element_type = str(opts.get("type", "unknown"))
            elif keyword == "nset":
                set_name = str(opts["nset"])
            elif keyword == "elset":
                set_name = str(opts["elset"])
            continue

        values = [item.strip() for item in line.split(",") if item.strip()]
        if section == "node":
            node_id = int(values[0])
            nodes[node_id] = (float(values[1]), float(values[2]))
        elif section == "element":
            element_id = int(values[0])
            elements[element_id] = tuple(map(int, values[1:]))
            element_types[element_id] = element_type
        elif section in {"nset", "elset"} and set_name:
            target = node_sets[set_name] if section == "nset" else element_sets[set_name]
            integers = list(map(int, values))
            if generate:
                if len(integers) != 3:
                    raise ValueError(f"Invalid generate set line: {line}")
                target.update(range(integers[0], integers[1] + 1, integers[2]))
            else:
                target.update(integers)

    return nodes, elements, element_types, node_sets, element_sets


def polygon_area(points: np.ndarray) -> float:
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def main() -> None:
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.patches import Patch

    nodes, elements, element_types, node_sets, element_sets = parse_abaqus(MESH)
    OUT.mkdir(parents=True, exist_ok=True)

    coordinates = np.array(list(nodes.values()))
    radii = np.linalg.norm(coordinates, axis=1)
    edge_lengths: list[float] = []
    areas: list[float] = []
    element_region: dict[int, str] = {}

    for element_id, connectivity in elements.items():
        points = np.array([nodes[node_id] for node_id in connectivity])
        areas.append(polygon_area(points))
        for index in range(len(points)):
            edge_lengths.append(float(np.linalg.norm(points[index] - points[index - 1])))
        memberships = [region for region in REGIONS if element_id in element_sets[region]]
        element_region[element_id] = memberships[0] if len(memberships) == 1 else "unassigned"

    internal_radii = np.array(
        [math.hypot(*nodes[node_id]) for node_id in node_sets["internal"]]
    )
    base_radii = np.array([math.hypot(*nodes[node_id]) for node_id in node_sets["base"]])
    nominal_outer_radius = float(base_radii.max())
    weld_cap_radius = float(radii.max())
    inner_radius = float(np.median(internal_radii))
    edge_array = np.array(edge_lengths)

    summary = {
        "source": str(MESH.relative_to(REPO)),
        "node_count": len(nodes),
        "element_count": len(elements),
        "element_types": {
            name: sum(value == name for value in element_types.values())
            for name in sorted(set(element_types.values()))
        },
        "region_element_counts": {
            region: sum(value == region for value in element_region.values())
            for region in (*REGIONS, "unassigned")
        },
        "coordinate_bounds_mm": {
            "x_min": float(coordinates[:, 0].min()),
            "x_max": float(coordinates[:, 0].max()),
            "y_min": float(coordinates[:, 1].min()),
            "y_max": float(coordinates[:, 1].max()),
        },
        "radius_mm": {
            "internal_median": inner_radius,
            "internal_min": float(internal_radii.min()),
            "internal_max": float(internal_radii.max()),
            "nominal_outer_from_base": nominal_outer_radius,
            "weld_cap_max": weld_cap_radius,
            "weld_cap_reinforcement": weld_cap_radius - nominal_outer_radius,
            "nominal_wall_from_radii": nominal_outer_radius - inner_radius,
            "weld_root_intrusion": inner_radius - float(internal_radii.min()),
        },
        "paper_geometry_mm": {
            "outer_radius": 711.0 / 2.0,
            "inner_radius": 711.0 / 2.0 - 17.5,
            "wall_thickness": 17.5,
        },
        "geometry_absolute_error_mm": {
            "outer_radius": abs(nominal_outer_radius - 711.0 / 2.0),
            "inner_radius": abs(inner_radius - (711.0 / 2.0 - 17.5)),
            "wall_thickness": abs((nominal_outer_radius - inner_radius) - 17.5),
        },
        "edge_length_mm": {
            "minimum": float(edge_array.min()),
            "p01": float(np.quantile(edge_array, 0.01)),
            "median": float(np.median(edge_array)),
            "maximum": float(edge_array.max()),
        },
        "element_area_mm2": {
            "minimum": float(np.min(areas)),
            "median": float(np.median(areas)),
            "maximum": float(np.max(areas)),
            "total": float(np.sum(areas)),
        },
        "named_node_sets": {name: len(ids) for name, ids in sorted(node_sets.items())},
        "named_element_sets": {
            name: len(ids) for name, ids in sorted(element_sets.items())
        },
    }
    (OUT / "mesh_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.4), constrained_layout=True)
    for axis, zoom in zip(axes, (False, True), strict=True):
        polygons = []
        facecolors = []
        for element_id, connectivity in elements.items():
            polygons.append([nodes[node_id] for node_id in connectivity])
            facecolors.append(COLORS[element_region[element_id]])
        collection = PolyCollection(
            polygons,
            facecolors=facecolors,
            edgecolors="#333842",
            linewidths=0.08 if not zoom else 0.16,
            rasterized=True,
        )
        axis.add_collection(collection)
        axis.autoscale()
        axis.set_aspect("equal")
        axis.set_xlabel("$x$ (mm)")
        axis.set_ylabel("$y$ (mm)")
        axis.set_title("Released 1/8 pipe mesh" if not zoom else "Weld-region detail")
        if zoom:
            axis.set_xlim(-32, 32)
            axis.set_ylim(332, 357)
        axis.grid(False)

    axes[0].legend(
        handles=[Patch(facecolor=COLORS[name], label=name) for name in REGIONS],
        frameon=False,
        loc="lower center",
        ncol=3,
    )
    fig.suptitle(
        f"Yang et al. (2026) released mesh: {len(nodes):,} nodes, "
        f"{len(elements):,} elements; nominal wall = "
        f"{nominal_outer_radius - inner_radius:.3f} mm"
    )
    fig.savefig(OUT / "released_mesh_audit.png", dpi=260)
    fig.savefig(OUT / "released_mesh_audit.pdf")
    plt.close(fig)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
