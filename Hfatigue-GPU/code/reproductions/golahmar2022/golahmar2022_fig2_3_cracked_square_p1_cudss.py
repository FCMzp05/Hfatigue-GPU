#!/usr/bin/env python3
"""Matrix-free JAX FEM reproduction of Golahmar et al. (2022), Figs. 2-3.

The model is a 1 x 1 mm plane-strain cracked square under fully reversed
piecewise-linear cyclic displacement.  It combines AT2 phase-field fracture,
the Amor volumetric/deviatoric split, the Carrara fatigue history/degradation
law, and the hydrogen-dependent fracture energy used by Golahmar et al.

The first reproduction target uses uniform pre-charging (0, 0.1, 0.5 and
1 wt-ppm).  Transient stress-assisted transport is intentionally left for the
frequency-effect benchmark; using a steady diffusion solve at 400 Hz would be
physically incorrect.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import re
import time
from pathlib import Path


def _requested_platform() -> str:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--platform", choices=("gpu",), default="gpu")
    return p.parse_known_args()[0].platform


_PLATFORM = _requested_platform()
if _PLATFORM != "auto":
    os.environ.setdefault("JAX_PLATFORMS", "cuda" if _PLATFORM == "gpu" else "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np  # noqa: E402

try:
    import jax  # noqa: E402
    import jax.numpy as jnp  # noqa: E402
    from jax.scipy.sparse.linalg import cg  # noqa: E402
except ImportError as exc:  # pragma: no cover
    raise SystemExit(f"JAX is required: {exc}") from exc


# Fixed-topology CSR backend for NVIDIA nvmath cuDSS, inlined verbatim from
# the shared ``cudss_backend`` module so this entry point has no repo-local
# imports.  CUDA dependencies are imported only when a solver is constructed.
class FixedTopologyCudssSolver:
    """Assemble element matrices into one reusable GPU CSR topology."""

    def __init__(self, connectivity: np.ndarray, free: np.ndarray):
        try:
            import cupy as cp
            import nvmath
            from cupyx.scipy.sparse import csr_matrix
        except ImportError as exc:
            raise RuntimeError(
                "cuDSS requires cupy-cuda12x and nvmath-python[cu12]"
            ) from exc

        self.cp = cp
        self.nvmath = nvmath
        size, width = len(free), connectivity.shape[1]
        rows = np.repeat(connectivity, width, axis=1).reshape(-1).astype(np.int64)
        cols = np.tile(connectivity, (1, width)).reshape(-1).astype(np.int64)
        keys = rows * size + cols
        unique, inverse = np.unique(keys, return_inverse=True)
        unique_rows, unique_cols = unique // size, unique % size
        indptr = np.searchsorted(
            unique_rows, np.arange(size + 1), side="left"
        ).astype(np.int32)
        fixed = np.flatnonzero(free == 0).astype(np.int64)

        self.inverse = cp.asarray(inverse, dtype=cp.int32)
        self.indices = cp.asarray(unique_cols, dtype=cp.int32)
        self.indptr = cp.asarray(indptr, dtype=cp.int32)
        self.free = cp.asarray(free, dtype=cp.float64)
        self.entry_mask = cp.asarray(free[unique_rows] * free[unique_cols])
        self.fixed_diagonal = cp.asarray(
            np.searchsorted(unique, fixed * (size + 1)), dtype=cp.int32
        )
        self.data = cp.zeros(len(unique), dtype=cp.float64)
        self.constrained_data = cp.zeros_like(self.data)
        self.matrix = csr_matrix(
            (self.data, self.indices, self.indptr), shape=(size, size)
        )
        self.constrained = csr_matrix(
            (self.constrained_data, self.indices, self.indptr),
            shape=(size, size),
        )
        self.rhs = cp.zeros(size, dtype=cp.float64)
        self.solver = None

    def solve(self, element_matrix, boundary, assembled_force):
        """Assemble, factor, solve, and return JAX solution and residual."""
        import jax
        import jax.numpy as jnp

        cp = self.cp
        jax.block_until_ready(element_matrix)
        values = cp.from_dlpack(element_matrix).reshape(-1)
        boundary_gpu = cp.from_dlpack(boundary)
        force_gpu = cp.from_dlpack(assembled_force)
        self.data.fill(0)
        cp.add.at(self.data, self.inverse, values)
        self.rhs[...] = self.free * (force_gpu - self.matrix @ boundary_gpu)
        self.constrained_data[...] = self.data * self.entry_mask
        self.constrained_data[self.fixed_diagonal] = 1.0
        if self.solver is None:
            self.solver = self.nvmath.sparse.advanced.DirectSolver(
                self.constrained, self.rhs
            )
            self.solver.plan()
        self.solver.factorize()
        solution = self.solver.solve()
        cp.cuda.Stream.null.synchronize()
        residual = cp.linalg.norm(
            self.constrained @ solution - self.rhs
        ) / (cp.linalg.norm(self.rhs) + 1.0e-30)
        full = boundary_gpu + self.free * solution
        return jnp.from_dlpack(full), jnp.asarray(float(residual.get()))

    def close(self) -> None:
        """Release the cuDSS plan explicitly."""
        if self.solver is not None:
            self.solver.free()
            self.solver = None


# Golahmar et al. (2022), cracked-square benchmark; mm, N, MPa, s and mol.
WIDTH, HEIGHT, INITIAL_CRACK = 1.0, 1.0, 0.5
NOTCH_WIDTH = 0.002
NOTCH_TAPER_LENGTH = 0.002
E, NU = 210000.0, 0.3
LAMBDA = E * NU / ((1.0 + NU) * (1.0 - 2.0 * NU))
MU = E / (2.0 * (1.0 + NU))
K_BULK = LAMBDA + 2.0 * MU / 3.0
GC0, L0, KAPPA = 2.7, 0.004, 1.0e-7
ALPHA_T = 56.25
DISPLACEMENT_RANGE = 4.0e-3
LOAD_AMPLITUDE = 0.5 * DISPLACEMENT_RANGE
LOAD_RATIO, FREQUENCY = -1.0, 400.0
D_H, V_H = 0.0127, 2000.0
STEPS_PER_CYCLE, NOMINAL_CYCLES = 20, 420
MAX_STAGGER, STAGGER_TOL = 80, 1.0e-6
# 1 wt-ppm = 1e-6 mass fraction.  Converting to the H/Fe atomic ratio gives
# (1e-6 / M_H) / (1 / M_Fe) ~= 55.5e-6, not 5.5e-6.
CH_ENV_1WPPM = 5.5e-5
DELTA_GB, HYDROGEN_CHI = 30000.0, 0.89
R_GAS, TEMPERATURE = 8.314, 300.0
VERSION = "golahmar2022_cracked_square_jax_v3_full_energy_fatigue"


def script_root() -> Path:
    here = Path(__file__).resolve().parent
    return Path(__file__).resolve().parents[3]


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--platform", choices=("gpu",), default="gpu")
    p.add_argument("--dtype", choices=("float64",), default="float64")
    p.add_argument(
        "--fatigue-law",
        choices=("rational", "logarithmic"),
        default="rational",
    )
    p.add_argument("--fatigue-kappa", type=float, default=0.15)
    p.add_argument(
        "--fatigue-energy",
        choices=("total", "tensile"),
        default="total",
    )
    p.add_argument("--delta-gb-j-per-mol", type=float, default=DELTA_GB)
    p.add_argument("--no-dynamic-crack-fix", action="store_true")
    p.add_argument(
        "--initial-phase-crack",
        action="store_true",
        help="fix d=1 on the embedded crack and equilibrate its AT2 profile",
    )
    p.add_argument("--hydrogen-wtppm", type=float, default=0.0)
    p.add_argument(
        "--hydrogen-mode",
        choices=("uniform", "transient"),
        default="transient",
        help="uniform pre-charge only or paper's stress-assisted transient diffusion",
    )
    p.add_argument(
        "--split",
        choices=("isotropic", "amor", "spectral"),
        default="amor",
        help="strain-energy split used for the three Fig. 2 curves.",
    )
    p.add_argument("--cycles", type=int, default=NOMINAL_CYCLES)
    p.add_argument("--steps-per-cycle", type=int, default=STEPS_PER_CYCLE)
    p.add_argument("--mesh", type=Path, required=True)
    p.add_argument("--outdir", type=Path)
    p.add_argument("--max-stagger", type=int, default=MAX_STAGGER)
    p.add_argument("--stagger-tol", type=float, default=STAGGER_TOL)
    p.add_argument("--cg-tol", type=float, default=1.0e-6)
    p.add_argument("--cg-maxiter", type=int, default=5000)
    p.add_argument("--linear-residual-tol", type=float, default=5.0e-4)
    p.add_argument("--print-every", type=int, default=10, help="cycle reporting interval")
    p.add_argument("--checkpoint-every", type=int, default=10)
    p.add_argument("--resume", action="store_true")
    return p


def concentration_tag(value: float) -> str:
    text = f"{value:.8g}".replace("-", "m").replace(".", "p")
    return f"{text}wppm"


def device_slug(device) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", device.device_kind.lower()).strip("_")
    return slug or device.platform.lower()






def load_mesh(path: Path):
    try:
        import meshio
    except ImportError as exc:
        raise SystemExit("meshio is required: python -m pip install meshio") from exc
    mesh = meshio.read(str(path))
    nodes = np.asarray(mesh.points[:, :2], dtype=np.float64)
    triangles = [np.asarray(c.data, dtype=np.int64) for c in mesh.cells if c.type == "triangle"]
    if not triangles:
        raise RuntimeError("mesh contains no linear triangles")
    tri = np.concatenate(triangles)
    tags_by_name = {name.lower(): int(value[0]) for name, value in mesh.field_data.items()}
    expected = {"domain", "bottom", "top", "crack"}
    if not expected.issubset(tags_by_name):
        raise RuntimeError(f"missing physical groups: {sorted(expected - tags_by_name.keys())}")
    boundaries = {name: [] for name in ("bottom", "top", "crack")}
    physical = mesh.cell_data.get("gmsh:physical", [])
    for ib, block in enumerate(mesh.cells):
        if block.type != "line":
            continue
        for edge, tag in zip(block.data, physical[ib]):
            for name in boundaries:
                if int(tag) == tags_by_name[name]:
                    boundaries[name].append(np.asarray(edge[:2], dtype=np.int64))
    edges = {key: np.asarray(value, dtype=np.int64).reshape((-1, 2))
             for key, value in boundaries.items()}
    groups = {key: np.unique(value) for key, value in edges.items()}
    xy = nodes[tri]
    twice_area = (
        (xy[:, 1, 0] - xy[:, 0, 0]) * (xy[:, 2, 1] - xy[:, 0, 1])
        - (xy[:, 2, 0] - xy[:, 0, 0]) * (xy[:, 1, 1] - xy[:, 0, 1])
    )
    flip = twice_area < 0
    tri[flip] = tri[flip][:, [0, 2, 1]]
    xy = nodes[tri]
    areas = 0.5 * (
        (xy[:, 1, 0] - xy[:, 0, 0]) * (xy[:, 2, 1] - xy[:, 0, 1])
        - (xy[:, 2, 0] - xy[:, 0, 0]) * (xy[:, 1, 1] - xy[:, 0, 1])
    )
    if np.any(areas <= 0):
        raise RuntimeError("nonpositive triangle area")
    return nodes, tri, areas, edges, groups


def boundary_cells(tri: np.ndarray, edges: np.ndarray) -> np.ndarray:
    lookup = {}
    for ie, cell in enumerate(tri):
        for a, b in ((cell[0], cell[1]), (cell[1], cell[2]), (cell[2], cell[0])):
            lookup[tuple(sorted((int(a), int(b))))] = ie
    return np.asarray([lookup[tuple(sorted(map(int, edge)))] for edge in edges], dtype=np.int64)


def preprocess(nodes, tri, areas):
    x, y = nodes[:, 0], nodes[:, 1]
    i, j, k = tri.T
    gx = np.stack((y[j] - y[k], y[k] - y[i], y[i] - y[j]), axis=1) / (2 * areas[:, None])
    gy = np.stack((x[k] - x[j], x[i] - x[k], x[j] - x[i]), axis=1) / (2 * areas[:, None])
    b = np.zeros((len(tri), 3, 6))
    for a in range(3):
        b[:, 0, 2 * a] = gx[:, a]
        b[:, 1, 2 * a + 1] = gy[:, a]
        b[:, 2, 2 * a] = gy[:, a]
        b[:, 2, 2 * a + 1] = gx[:, a]
    dofs = np.empty((len(tri), 6), dtype=np.int64)
    dofs[:, 0::2], dofs[:, 1::2] = 2 * tri, 2 * tri + 1
    mass = areas[:, None, None] * np.array([[2, 1, 1], [1, 2, 1], [1, 1, 2]]) / 12
    grad = areas[:, None, None] * (
        gx[:, :, None] * gx[:, None, :] + gy[:, :, None] * gy[:, None, :]
    )
    return gx, gy, b, dofs, mass, grad


def make_solvers(*, nodes_np, tri_np, areas_np, gx_np, gy_np, b_np, dofs_np,
                 mass_np, grad_np, free_u_np, top_edges_np, top_cells_np,
                 fixed_d_np, split, dtype, tol, maxiter, fatigue_law="rational",
                 fatigue_kappa=0.15, dynamic_crack_fix=True,
                 fatigue_energy="total"):
    tri, dofs = jnp.asarray(tri_np), jnp.asarray(dofs_np)
    arrays = (areas_np, gx_np, gy_np, b_np, mass_np, grad_np)
    areas, gx, gy, b, mass, grad = (jnp.asarray(a, dtype=dtype) for a in arrays)
    free_u = jnp.asarray(free_u_np, dtype=dtype)
    fixed_d = jnp.asarray(fixed_d_np, dtype=dtype)
    nodes = jnp.asarray(nodes_np, dtype=dtype)
    nnode, ndof = len(nodes_np), 2 * len(nodes_np)
    top_edges, top_cells = jnp.asarray(top_edges_np), jnp.asarray(top_cells_np)
    edge_len = jnp.linalg.norm(nodes[top_edges[:, 1]] - nodes[top_edges[:, 0]], axis=1)
    tiny = jnp.asarray(1.0e-30, dtype=dtype)
    mechanics_direct = FixedTopologyCudssSolver(dofs_np, free_u_np)
    zero_force = jnp.zeros(ndof, dtype=dtype)

    def scatter(ids, values, size):
        return jnp.zeros(size, dtype=dtype).at[ids.reshape(-1)].add(values.reshape(-1))

    def cell_state(u, d):
        strain = jnp.einsum("eij,ej->ei", b, u[dofs])
        tr = strain[:, 0] + strain[:, 1]
        eps2 = strain[:, 0] ** 2 + strain[:, 1] ** 2 + 0.5 * strain[:, 2] ** 2
        psi_total = 0.5 * LAMBDA * tr * tr + MU * eps2
        dev2 = (
            (strain[:, 0] - tr / 3) ** 2
            + (strain[:, 1] - tr / 3) ** 2
            + (tr / 3) ** 2 + 0.5 * strain[:, 2] ** 2
        )
        psi_plus = 0.5 * K_BULK * jnp.maximum(tr, 0) ** 2 + MU * dev2
        dm = jnp.mean(jnp.clip(d[tri], 0, 1), axis=1)
        degradation = (1 - dm) ** 2 + KAPPA
        if split == "isotropic":
            psi_plus = 0.5 * LAMBDA * tr * tr + MU * eps2
            c11 = degradation * (LAMBDA + 2 * MU)
            c12 = degradation * LAMBDA
        elif split == "spectral":
            mean = 0.5 * tr
            radius = jnp.sqrt(
                (0.5 * (strain[:, 0] - strain[:, 1])) ** 2
                + (0.5 * strain[:, 2]) ** 2
            )
            principal_1, principal_2 = mean + radius, mean - radius
            psi_plus = (
                0.5 * LAMBDA * jnp.maximum(tr, 0) ** 2
                + MU * (
                    jnp.maximum(principal_1, 0) ** 2
                    + jnp.maximum(principal_2, 0) ** 2
                )
            )
            # Hybrid implementation: the spectral split drives fracture,
            # while equilibrium uses the robust isotropically degraded stress.
            c11 = degradation * (LAMBDA + 2 * MU)
            c12 = degradation * LAMBDA
        else:
            psi_plus = 0.5 * K_BULK * jnp.maximum(tr, 0) ** 2 + MU * dev2
            kv = K_BULK * jnp.where(tr >= 0, degradation, 1.0)
            c11 = kv + 4 * MU * degradation / 3
            c12 = kv - 2 * MU * degradation / 3
        c66 = MU * degradation
        zero = jnp.zeros_like(degradation)
        constit = jnp.stack((
            jnp.stack((c11, c12, zero), axis=1),
            jnp.stack((c12, c11, zero), axis=1),
            jnp.stack((zero, zero, c66), axis=1),
        ), axis=1)
        stress = jnp.einsum("eij,ej->ei", constit, strain)
        # Equation (10) of Golahmar et al. uses the complete undamaged
        # elastic energy for fatigue accumulation, alpha=g(phi)*psi_0.
        # The tension-compression split applies only to fracture driving.
        alpha = degradation * (
            psi_plus if fatigue_energy == "tensile" else psi_total
        )
        if split == "amor":
            hydrostatic = K_BULK * jnp.where(
                tr >= 0, degradation, 1.0
            ) * tr
        else:
            hydrostatic = degradation * K_BULK * tr
        return psi_plus, alpha, constit, stress, hydrostatic

    @jax.jit
    def assemble_u(d, boundary, x0):
        _, _, constit, _, _ = cell_state(x0 + boundary, d)
        return areas[:, None, None] * jnp.einsum(
            "eia,eij,ejb->eab", b, constit, b
        )

    def solve_u(d, boundary, x0):
        solution, residual = mechanics_direct.solve(
            assemble_u(d, boundary, x0), boundary, zero_force
        )
        return solution, residual, jnp.asarray(0, dtype=jnp.int32)

    def fatigue_factor(alpha_bar):
        if fatigue_law == "logarithmic":
            ratio = jnp.maximum(alpha_bar / ALPHA_T, 1.0)
            # Eq. (11) bounds the active range by alpha_T*10**(1/kappa),
            # which fixes the logarithm as base 10.
            degraded = jnp.maximum(
                1.0 - fatigue_kappa * jnp.log10(ratio), 0.0
            ) ** 2
            return jnp.where(alpha_bar <= ALPHA_T, 1.0, degraded)
        ratio = 2 * ALPHA_T / jnp.maximum(alpha_bar + ALPHA_T, tiny)
        return jnp.where(alpha_bar <= ALPHA_T, 1.0, ratio * ratio)

    def update_fatigue(alpha, alpha_previous, alpha_bar):
        delta = alpha - alpha_previous
        increment = jnp.where(alpha * delta > 0, jnp.abs(delta), 0.0)
        return alpha_bar + increment

    def solve_d(history, toughness_factor, x0):
        gd = GC0 * toughness_factor
        coeff = gd / L0 + 2 * history
        ke = gd[:, None, None] * L0 * grad + coeff[:, None, None] * mass
        rhs = scatter(tri, jnp.broadcast_to((2 * history * areas / 3)[:, None], tri.shape), nnode)

        def kmv(x):
            return scatter(tri, jnp.einsum("eij,ej->ei", ke, x[tri]), nnode)

        # Crack-set irreversibility used by the reference implementation:
        # nodes reaching phi >= 0.95 join the Dirichlet crack set at phi = 1.
        # Assigning their current value (rather than one) artificially blunts
        # the crack and severely delays the tracked phi=0.95 crack tip.
        fixed = (
            jnp.maximum(fixed_d, (x0 >= 0.95).astype(dtype))
            if dynamic_crack_fix else fixed_d
        )
        free = 1.0 - fixed
        boundary = fixed
        rhs_free = free * (rhs - kmv(boundary))

        def op(x):
            return free * kmv(free * x) + fixed * x

        diag = free * scatter(
            tri, jnp.diagonal(ke, axis1=1, axis2=2), nnode
        ) + fixed
        sol_free, _ = cg(op, rhs_free, x0=free * x0, tol=tol, atol=0.0, maxiter=maxiter,
                    M=lambda r: r / jnp.maximum(diag, tiny))
        sol = boundary + free * sol_free
        # The phase-field right-hand side is exactly zero before loading.
        # Normalising only by ||rhs|| would report a huge meaningless number
        # even when CG has reduced the initial-crack residual to round-off.
        scale = jnp.maximum(
            jnp.maximum(jnp.linalg.norm(rhs_free), jnp.linalg.norm(op(free * x0))),
            tiny,
        )
        residual = jnp.linalg.norm(op(sol_free) - rhs_free) / scale
        return sol, residual

    def qois(u, d):
        psi, alpha, _, stress, hydrostatic = cell_state(u, d)
        reaction = jnp.sum(stress[top_cells, 1] * edge_len)
        dv = d[tri]
        gd_x, gd_y = jnp.sum(dv * gx, axis=1), jnp.sum(dv * gy, axis=1)
        gamma = jnp.sum(areas * (
            jnp.mean(dv * dv, axis=1) / (2 * L0)
            + 0.5 * L0 * (gd_x * gd_x + gd_y * gd_y)
        ))
        return psi, alpha, reaction, gamma, hydrostatic

    return tuple(jax.jit(f) for f in (
        solve_u, fatigue_factor, update_fatigue, solve_d, qois
    ))


def make_diffusion_solver(*, tri_np, areas_np, gx_np, gy_np, mass_np,
                          grad_np, dtype, tol, maxiter, steps_per_cycle):
    """Implicit stress-assisted diffusion, Eqs. (12)-(14) of the paper."""
    tri = jnp.asarray(tri_np)
    arrays = (areas_np, gx_np, gy_np, mass_np, grad_np)
    areas, gx, gy, mass, grad = (jnp.asarray(a, dtype=dtype) for a in arrays)
    nnode = int(np.max(tri_np)) + 1
    tiny = jnp.asarray(1.0e-30, dtype=dtype)
    dt = jnp.asarray(1.0 / (FREQUENCY * steps_per_cycle), dtype=dtype)
    beta = jnp.asarray(V_H / (R_GAS * TEMPERATURE * 1000.0), dtype=dtype)
    nodal_weight = jnp.zeros(nnode, dtype=dtype).at[tri.reshape(-1)].add(
        jnp.broadcast_to((areas / 3)[:, None], tri.shape).reshape(-1)
    )
    def scatter(ids, values):
        return jnp.zeros(nnode, dtype=dtype).at[ids.reshape(-1)].add(values.reshape(-1))

    def solve_c(c_old, sigma_h, fixed, c_environment):
        # Recover continuous nodal hydrostatic stress before differentiating.
        sigma_node = scatter(
            tri,
            jnp.broadcast_to((sigma_h * areas / 3)[:, None], tri.shape),
        ) / jnp.maximum(nodal_weight, tiny)
        # Slotboom variable z=C*exp(-beta*sigma_H) converts stress-driven
        # diffusion to an SPD system and prevents explicit-drift overshoots.
        sigma_factor_node = jnp.exp(jnp.clip(beta * sigma_node, -20, 20))
        sigma_factor_cell = jnp.mean(sigma_factor_node[tri], axis=1)
        ae = sigma_factor_cell[:, None, None] * (mass + dt * D_H * grad)

        def matvec(x):
            return scatter(tri, jnp.einsum("eij,ej->ei", ae, x[tri]))

        diag_base = scatter(tri, jnp.diagonal(ae, axis1=1, axis2=2))
        rhs = scatter(tri, jnp.einsum("eij,ej->ei", mass, c_old[tri]))

        fixed_f = fixed.astype(dtype)
        free = 1.0 - fixed_f
        boundary = fixed_f * c_environment / jnp.maximum(sigma_factor_node, tiny)
        rhs_free = free * (rhs - matvec(boundary))

        def op(x):
            return free * matvec(free * x) + fixed_f * x

        diag = free * diag_base + fixed_f
        sol_free, _ = cg(
            op, rhs_free,
            x0=free * c_old / jnp.maximum(sigma_factor_node, tiny),
            tol=tol, atol=0.0,
            maxiter=maxiter, M=lambda r: r / jnp.maximum(diag, tiny),
        )
        concentration = sigma_factor_node * (boundary + free * sol_free)
        scale = jnp.maximum(jnp.linalg.norm(rhs_free), tiny)
        residual = jnp.linalg.norm(op(sol_free) - rhs_free) / scale
        return jnp.maximum(concentration, 0.0), residual

    return jax.jit(solve_c)


def hydrogen_factor(wtppm: float, delta_gb: float = DELTA_GB) -> float:
    concentration = wtppm * CH_ENV_1WPPM
    theta = concentration / (
        concentration + np.exp(-delta_gb / (R_GAS * TEMPERATURE))
    )
    return float(np.clip(1.0 - HYDROGEN_CHI * theta, 0.0, 1.0))


def cyclic_displacement(step_in_cycle: int, steps_per_cycle: int) -> float:
    phase = step_in_cycle / steps_per_cycle
    if phase <= 0.25:
        return 4 * LOAD_AMPLITUDE * phase
    if phase <= 0.75:
        return 2 * LOAD_AMPLITUDE - 4 * LOAD_AMPLITUDE * phase
    return -4 * LOAD_AMPLITUDE + 4 * LOAD_AMPLITUDE * phase


def crack_extension(nodes: np.ndarray, tri: np.ndarray, d: np.ndarray) -> float:
    del tri  # nodal threshold avoids a mesh-dependent cell-average offset
    mask = (np.abs(nodes[:, 1]) <= 3 * L0) & (d >= 0.95)
    tip = max(INITIAL_CRACK, float(np.max(nodes[mask, 0])) if np.any(mask) else INITIAL_CRACK)
    return max(0.0, tip - INITIAL_CRACK)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def checkpoint(path: Path, **state) -> None:
    np.savez_compressed(path, **state)


def main() -> None:
    args = parser().parse_args()
    if args.hydrogen_wtppm < 0:
        raise ValueError("--hydrogen-wtppm must be non-negative")
    if args.cycles < 1 or args.steps_per_cycle < 4 or args.steps_per_cycle % 4:
        raise ValueError("cycles must be positive and steps-per-cycle a positive multiple of 4")

    jax.config.update("jax_enable_x64", args.dtype == "float64")
    dtype = jnp.float64 if args.dtype == "float64" else jnp.float32
    devices = jax.devices()
    if args.platform == "gpu" and not any(device.platform == "gpu" for device in devices):
        raise RuntimeError(f"GPU requested, available devices: {devices}")

    case_root = script_root() / "outputs" / "hydrogen_fatigue_cracked_square" / "jax"
    run_name = f"{args.split}_{concentration_tag(args.hydrogen_wtppm)}"
    args.outdir = args.outdir or case_root / run_name
    args.outdir.mkdir(parents=True, exist_ok=True)
    mesh_path = args.mesh.expanduser().resolve()
    if not mesh_path.is_file():
        raise FileNotFoundError(f"required existing mesh not found: {mesh_path}")

    nodes, tri, areas, edges, groups = load_mesh(mesh_path)
    gx, gy, b, dofs, mass, grad = preprocess(nodes, tri, areas)
    top_cells = boundary_cells(tri, edges["top"])
    nnode, ndof = len(nodes), 2 * len(nodes)

    # Exact support idealisation in Fig. 1: bottom vertical roller plus one
    # horizontal pin to remove rigid translation.
    anchor = groups["bottom"][np.argmax(nodes[groups["bottom"], 0])]
    fixed_u = np.unique(np.concatenate((
        2 * groups["bottom"] + 1,
        2 * groups["top"] + 1,
        np.asarray([2 * anchor]),
    )))
    free_u = np.ones(ndof)
    free_u[fixed_u] = 0
    fixed_d = np.zeros(nnode)
    if args.initial_phase_crack:
        fixed_d[groups["crack"]] = 1.0
    top_y = 2 * groups["top"] + 1

    solve_u, fatigue_factor_fn, update_fatigue, solve_d, qois = make_solvers(
        nodes_np=nodes, tri_np=tri, areas_np=areas, gx_np=gx, gy_np=gy,
        b_np=b, dofs_np=dofs, mass_np=mass, grad_np=grad, free_u_np=free_u,
        top_edges_np=edges["top"], top_cells_np=top_cells, dtype=dtype,
        fixed_d_np=fixed_d, split=args.split, tol=args.cg_tol,
        maxiter=args.cg_maxiter, fatigue_law=args.fatigue_law,
        fatigue_kappa=args.fatigue_kappa,
        dynamic_crack_fix=not args.no_dynamic_crack_fix,
        fatigue_energy=args.fatigue_energy,
    )
    solve_c = make_diffusion_solver(
        tri_np=tri, areas_np=areas, gx_np=gx, gy_np=gy,
        mass_np=mass, grad_np=grad, dtype=dtype, tol=args.cg_tol,
        maxiter=args.cg_maxiter, steps_per_cycle=args.steps_per_cycle,
    )
    outer_c_np = (
        np.isclose(nodes[:, 0], 0.0)
        | np.isclose(nodes[:, 0], WIDTH)
        | np.isclose(nodes[:, 1], -0.5 * HEIGHT)
        | np.isclose(nodes[:, 1], 0.5 * HEIGHT)
    )
    outer_c = jnp.asarray(outer_c_np)
    crack_c_np = np.zeros(nnode, dtype=bool)
    crack_c_np[groups["crack"]] = True
    crack_c = jnp.asarray(crack_c_np)
    c_environment = args.hydrogen_wtppm * CH_ENV_1WPPM

    cp_path = args.outdir / "checkpoint_latest.npz"
    csv_path = args.outdir / "results.csv"
    d0 = fixed_d.copy()
    if args.initial_phase_crack:
        d_equilibrated, d_initial_residual = solve_d(
            jnp.zeros(len(tri), dtype=dtype),
            jnp.ones(len(tri), dtype=dtype),
            jnp.asarray(d0, dtype=dtype),
        )
        jax.block_until_ready(d_equilibrated)
        d0 = np.asarray(jax.device_get(d_equilibrated))
        if float(jax.device_get(d_initial_residual)) > args.linear_residual_tol:
            raise RuntimeError(
                "initial AT2 crack equilibrium failed: "
                f"residual={float(jax.device_get(d_initial_residual)):.3e}"
            )
    _, _, _, initial_gamma_device, _ = qois(
        jnp.zeros(ndof, dtype=dtype), jnp.asarray(d0, dtype=dtype)
    )
    initial_gamma = float(jax.device_get(initial_gamma_device))
    if args.resume and cp_path.is_file():
        saved = np.load(cp_path, allow_pickle=False)
        completed_steps = int(saved["completed_steps"])
        u, d, history = (jnp.asarray(saved[name], dtype=dtype) for name in ("u", "d", "history"))
        alpha_previous = jnp.asarray(saved["alpha_previous"], dtype=dtype)
        alpha_bar = jnp.asarray(saved["alpha_bar"], dtype=dtype)
        concentration = jnp.asarray(
            saved["concentration"] if "concentration" in saved
            else np.full(nnode, c_environment),
            dtype=dtype,
        )
        rows = json.loads(str(saved["rows_json"]))
    else:
        completed_steps, rows = 0, []
        u = jnp.zeros(ndof, dtype=dtype)
        # The initial defect is represented by the geometric notch itself.
        d = jnp.asarray(d0, dtype=dtype)
        history = jnp.zeros(len(tri), dtype=dtype)
        alpha_previous = jnp.zeros(len(tri), dtype=dtype)
        alpha_bar = jnp.zeros(len(tri), dtype=dtype)
        concentration = jnp.full(nnode, c_environment, dtype=dtype)

    h_factor = hydrogen_factor(
        args.hydrogen_wtppm, args.delta_gb_j_per_mol
    )
    total_steps = args.cycles * args.steps_per_cycle
    start_time = time.perf_counter()
    cycle_start_time = start_time
    max_linear_residual = 0.0
    print("=" * 78)
    print("Golahmar 2022 cracked square — JAX phase-field hydrogen fatigue")
    print(f"JAX {jax.__version__}; Python {platform.python_version()}; devices={devices}")
    print(f"mesh={mesh_path}: {nnode} nodes, {len(tri)} triangles")
    print(f"split={args.split}; H={args.hydrogen_wtppm:g} wt-ppm; "
          f"mode={args.hydrogen_mode}; f_H(env)={h_factor:.6f}; "
          f"cycles={args.cycles}; increments/cycle={args.steps_per_cycle}")
    print("=" * 78)

    for global_step in range(completed_steps, total_steps):
        tic = time.perf_counter()
        step_in_cycle = global_step % args.steps_per_cycle + 1
        cycle = global_step // args.steps_per_cycle + 1
        if step_in_cycle == 1:
            cycle_start_time = tic
            cycle_u_cg_iterations = jnp.asarray(0, dtype=jnp.int32)
            cycle_u_solve_calls = 0
        target = cyclic_displacement(step_in_cycle, args.steps_per_cycle)
        boundary_np = np.zeros(ndof)
        boundary_np[top_y] = target
        boundary = jnp.asarray(boundary_np, dtype=dtype)
        d_step = d

        # Update the fatigue history exactly once per physical load increment.
        u, ur, uit = solve_u(d, boundary, u)
        cycle_u_cg_iterations = cycle_u_cg_iterations + uit
        cycle_u_solve_calls += 1
        psi, alpha, _, _, hydrostatic = qois(u, d)
        if args.hydrogen_mode == "transient" and c_environment > 0:
            fixed_c = outer_c | crack_c | (d >= 0.95)
            solid_fraction = (
                1 - jnp.mean(jnp.clip(d[tri], 0, 1), axis=1)
            ) ** 2
            concentration, cr = solve_c(
                concentration, hydrostatic * solid_fraction,
                fixed_c, c_environment
            )
        else:
            cr = jnp.asarray(0.0, dtype=dtype)
        c_cells = jnp.mean(concentration[tri], axis=1)
        theta_cells = c_cells / (
            c_cells
            + jnp.exp(-args.delta_gb_j_per_mol / (R_GAS * TEMPERATURE))
        )
        h_cells = jnp.clip(1.0 - HYDROGEN_CHI * theta_cells, 0.0, 1.0)
        alpha_bar_step = update_fatigue(alpha, alpha_previous, alpha_bar)
        f_factor = fatigue_factor_fn(alpha_bar_step)
        toughness = h_cells * f_factor
        history_step = jnp.maximum(history, psi)

        error = np.inf
        for iteration in range(1, args.max_stagger + 1):
            d_old = d
            # The first mechanical state is already available from the
            # physical-increment update above.  Re-solving it here used to
            # duplicate one expensive displacement CG solve per increment.
            if iteration > 1:
                u, ur, uit = solve_u(d, boundary, u)
                cycle_u_cg_iterations = cycle_u_cg_iterations + uit
                cycle_u_solve_calls += 1
                psi, _, _, _, hydrostatic = qois(u, d)
            history_trial = jnp.maximum(history_step, psi)
            d_trial, dr = solve_d(history_trial, toughness, d)
            d = jnp.clip(jnp.maximum(d_trial, d_step), 0, 1)
            history_step = history_trial
            jax.block_until_ready(d)
            error = float(jax.device_get(
                jnp.linalg.norm(d - d_old) / (jnp.linalg.norm(d) + 1.0e-30)
            ))
            if error < args.stagger_tol:
                break

        residuals = tuple(float(jax.device_get(value)) for value in (ur, dr, cr))
        max_linear_residual = max(max_linear_residual, *residuals)
        if not all(np.isfinite(value) and value <= args.linear_residual_tol for value in residuals):
            raise RuntimeError(
                "linear residual limit exceeded: "
                f"u={residuals[0]:.3e}, d={residuals[1]:.3e}, "
                f"c={residuals[2]:.3e}"
            )
        if not np.isfinite(error):
            raise RuntimeError("non-finite stagger error")

        history = history_step
        alpha_bar = alpha_bar_step
        alpha_previous = alpha

        if step_in_cycle == args.steps_per_cycle:
            _, _, reaction, gamma, _ = qois(u, d)
            dh = np.asarray(jax.device_get(d))
            abar_h = np.asarray(jax.device_get(alpha_bar))
            f_h = np.asarray(jax.device_get(f_factor))
            c_h = np.asarray(jax.device_get(concentration))
            hc_h = np.asarray(jax.device_get(h_cells))
            gamma_value = float(jax.device_get(gamma))
            row = {
                "cycle": cycle,
                "hydrogen_wtppm": args.hydrogen_wtppm,
                # The reference implementation tracks the furthest integration
                # point with phi >= 0.95.  The regularised surface increment is
                # retained as a separate energy diagnostic.
                "crack_extension_mm": crack_extension(nodes, tri, dh),
                "crack_surface_extension_mm": max(0.0, gamma_value - initial_gamma),
                "crack_surface_gamma_mm": gamma_value,
                "max_damage": float(dh.max()),
                "max_alpha_bar_mpa": float(abar_h.max()),
                "min_fatigue_factor": float(f_h.min()),
                "hydrogen_factor": h_factor,
                "minimum_local_hydrogen_factor": float(hc_h.min()),
                "maximum_hydrogen_wtppm": float(c_h.max() / CH_ENV_1WPPM),
                "reaction_at_cycle_end_N_per_mm": float(jax.device_get(reaction)),
                "last_stagger_iterations": iteration,
                "last_stagger_error": error,
                "u_solve_calls": cycle_u_solve_calls,
                "u_cg_iterations": int(jax.device_get(cycle_u_cg_iterations)),
                "cycle_wall_s": time.perf_counter() - cycle_start_time,
                "cumulative_wall_s": time.perf_counter() - start_time,
            }
            rows.append(row)
            write_csv(csv_path, rows)
            if cycle <= 3 or cycle % args.print_every == 0:
                print(
                    f"cycle={cycle:4d} da={row['crack_extension_mm']:.6f} mm "
                    f"dmax={row['max_damage']:.4f} abar={row['max_alpha_bar_mpa']:.3e} "
                    f"fFmin={row['min_fatigue_factor']:.4f} fp={iteration} "
                    f"uCG={row['u_cg_iterations']} dt={row['cycle_wall_s']:.2f}s"
                )
            if cycle % args.checkpoint_every == 0 or cycle == args.cycles:
                checkpoint(
                    cp_path, completed_steps=np.asarray(global_step + 1),
                    u=np.asarray(jax.device_get(u)), d=dh,
                    history=np.asarray(jax.device_get(history)),
                    alpha_previous=np.asarray(jax.device_get(alpha_previous)),
                    alpha_bar=abar_h, concentration=c_h,
                    rows_json=np.asarray(json.dumps(rows)),
                )
            if row["crack_extension_mm"] >= 0.49:
                print("early stop: crack traversed the ligament")
                break

    final_state = args.outdir / "final_state.npz"
    checkpoint(
        final_state, nodes=nodes, triangles=tri,
        u=np.asarray(jax.device_get(u)).reshape((-1, 2)),
        d=np.asarray(jax.device_get(d)),
        history=np.asarray(jax.device_get(history)),
        alpha_bar=np.asarray(jax.device_get(alpha_bar)),
        concentration=np.asarray(jax.device_get(concentration)),
    )
    first_growth = next(
        (row for row in rows if float(row["crack_extension_mm"]) > 0.0),
        None,
    )
    summary = {
        "version": VERSION,
        "paper": "Golahmar et al., International Journal of Fatigue 154 (2022) 106521",
        "doi": "10.1016/j.ijfatigue.2021.106521",
        "jax_version": jax.__version__,
        "device": str(devices[0]),
        "dtype": args.dtype,
        "split": args.split,
        "nodes": nnode,
        "triangles": len(tri),
        "hydrogen_wtppm": args.hydrogen_wtppm,
        "hydrogen_mode": args.hydrogen_mode,
        "hydrogen_factor": h_factor,
        "cycles_requested": args.cycles,
        "cycles_completed": len(rows),
        "first_growth_cycle": int(first_growth["cycle"]) if first_growth else None,
        "final_crack_extension_mm": float(rows[-1]["crack_extension_mm"]) if rows else None,
        "steps_per_cycle": args.steps_per_cycle,
        "maximum_linear_residual": max_linear_residual,
        "total_wall_s": time.perf_counter() - start_time,
        "output_files": ["results.csv", "checkpoint_latest.npz", "final_state.npz", "summary.json"],
        "limitations": [
            "P1 triangles replace the paper's reduced-integration Q8 elements.",
            "Figs. 2-3 use uniform pre-charging; transient diffusion is reserved for the frequency benchmark.",
        ],
    }
    (args.outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"completed; outputs: {args.outdir}")


if __name__ == "__main__":
    main()
