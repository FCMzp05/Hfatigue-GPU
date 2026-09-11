#!/usr/bin/env python3
"""Golahmar et al. (2022) Figs. 2-3 cracked-square Q8 solver on NVIDIA cuDSS.

Self-contained FP64 Q8 hydrogen-fatigue reproduction using fixed-topology
CSR and NVIDIA cuDSS.  Mechanics and phase-field systems are assembled by
JAX and solved only by cuDSS; transient hydrogen diffusion remains a JAX
BiCGSTAB solve.  The parser defaults reproduce the historical Golahmar
Figs. 2-3 wrapper (430 cycles, 20 steps/cycle, transient hydrogen).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

def load_jax_runtime() -> None:
    """Load simulation-only dependencies after CLI parsing."""
    global np, jax, jnp, bicgstab
    try:
        import numpy as numpy_module
        import jax as jax_module
        import jax.numpy as jax_numpy
        from jax.scipy.sparse.linalg import bicgstab as jax_bicgstab
    except ImportError as exc:
        raise RuntimeError(
            "the accelerated solver requires NumPy and JAX with CUDA"
        ) from exc
    np, jax, jnp, bicgstab = (
        numpy_module, jax_module, jax_numpy, jax_bicgstab
    )

WIDTH, HEIGHT, A0 = 1.0, 1.0, 0.5
E, NU = 210000.0, 0.3
LAMBDA = E * NU / ((1 + NU) * (1 - 2 * NU))
MU = E / (2 * (1 + NU))
K_BULK = LAMBDA + 2 * MU / 3
GC0, L0, KAPPA = 2.7, 0.004, 1.0e-7
ALPHA_T, UAMP = 56.25, 2.0e-3
FREQUENCY, D_H, V_H = 400.0, 0.0127, 2000.0
R_GAS, TEMPERATURE = 8.314, 300.0
DELTA_GB, CHI, CH_1WPPM = 30000.0, 0.89, 5.5e-5
VERSION = "q8_fp64_fixed_csr_nvmath_cudss_v1"


def repository_root() -> Path:
    """Resolve the repository root from code/accelerate."""
    return Path(__file__).resolve().parents[3]


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--platform", choices=("gpu",), default="gpu")
    result.add_argument("--dtype", choices=("float64",), default="float64")
    result.add_argument("--split", choices=("isotropic", "amor", "spectral"), default="amor")
    result.add_argument("--hydrogen-wtppm", type=float, default=0.0)
    result.add_argument("--hydrogen-mode", choices=("uniform", "transient"), default="transient")
    result.add_argument("--delta-gb-j-per-mol", type=float, default=DELTA_GB)
    result.add_argument("--cycles", type=int, default=430)
    result.add_argument("--steps-per-cycle", type=int, default=20)
    result.add_argument("--mesh", type=Path, required=True)
    result.add_argument("--outdir", type=Path)
    result.add_argument("--diffusion-tol", type=float, default=1.0e-6)
    result.add_argument("--diffusion-maxiter", type=int, default=6000)
    result.add_argument("--linear-residual-tol", type=float, default=1.0e-4)
    result.add_argument("--max-stagger", type=int, default=80)
    result.add_argument("--stagger-tol", type=float, default=1.0e-6)
    result.add_argument("--print-every", type=int, default=1)
    result.add_argument("--checkpoint-every", type=int, default=1)
    result.add_argument("--resume", action="store_true")
    result.add_argument("--paper-label", default="Golahmar et al. (2022), Figs. 2-3")
    result.add_argument("--case-label", default="golahmar2022_fig2_3_cracked_square_q8")
    return result


class FixedTopologyCudssSolver:
    """Assemble element matrices into reusable GPU CSR topology."""

    def __init__(self, connectivity: np.ndarray, free: np.ndarray):
        try:
            import cupy as cp
            import nvmath
            from cupyx.scipy.sparse import csr_matrix
        except ImportError as exc:
            raise RuntimeError(
                "requires cupy-cuda12x and nvmath-python[cu12]"
            ) from exc
        self.cp, self.nvmath = cp, nvmath
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
        if self.solver is not None:
            self.solver.free()
            self.solver = None


def load_mesh(path: Path):
    import meshio

    mesh = meshio.read(path)
    nodes = np.asarray(mesh.points[:, :2], dtype=np.float64)
    blocks = [
        np.asarray(cell.data, dtype=np.int64)
        for cell in mesh.cells if cell.type == "quad8"
    ]
    if not blocks:
        raise RuntimeError("mesh contains no quad8 cells")
    elements = np.concatenate(blocks)
    tags = {name.lower(): int(value[0]) for name, value in mesh.field_data.items()}
    missing = {"bottom", "top", "crack"} - tags.keys()
    if missing:
        raise RuntimeError(f"mesh is missing physical groups: {sorted(missing)}")
    groups = {name: [] for name in ("bottom", "top", "crack")}
    physical = mesh.cell_data.get("gmsh:physical", [])
    for block_index, block in enumerate(mesh.cells):
        if block.type != "line3":
            continue
        for edge, tag in zip(block.data, physical[block_index]):
            for name in groups:
                if int(tag) == tags[name]:
                    groups[name].extend(map(int, edge))
    return nodes, elements, {key: np.unique(value) for key, value in groups.items()}


def _q8_at(points: np.ndarray):
    shape = np.empty((len(points), 8))
    derivative = np.empty((len(points), 8, 2))
    for q, (x, y) in enumerate(points):
        shape[q] = (
            .25*(1-x)*(1-y)*(-x-y-1), .25*(1+x)*(1-y)*(x-y-1),
            .25*(1+x)*(1+y)*(x+y-1), .25*(1-x)*(1+y)*(-x+y-1),
            .5*(1-x*x)*(1-y), .5*(1+x)*(1-y*y),
            .5*(1-x*x)*(1+y), .5*(1-x)*(1-y*y),
        )
        derivative[q, :, 0] = (
            .25*(1-y)*(2*x+y), .25*(1-y)*(2*x-y),
            .25*(1+y)*(2*x+y), .25*(1+y)*(2*x-y),
            -x*(1-y), .5*(1-y*y), -x*(1+y), -.5*(1-y*y),
        )
        derivative[q, :, 1] = (
            .25*(1-x)*(x+2*y), .25*(1+x)*(-x+2*y),
            .25*(1+x)*(x+2*y), .25*(1-x)*(-x+2*y),
            -.5*(1-x*x), -(1+x)*y, .5*(1-x*x), -(1-x)*y,
        )
    return shape, derivative


def q8_shapes():
    a = 1 / np.sqrt(3)
    points = np.asarray([[-a, -a], [a, -a], [a, a], [-a, a]])
    return points, *_q8_at(points)


def preprocess(nodes: np.ndarray, elements: np.ndarray):
    _, shape, derivative = q8_shapes()
    coordinates = nodes[elements]
    jacobian = np.einsum("qia,eib->eqab", derivative, coordinates)
    det = np.linalg.det(jacobian)
    if np.any(det <= 0):
        raise RuntimeError(f"nonpositive Q8 Jacobian: {det.min()}")
    grad = np.einsum("qia,eqba->eqib", derivative, np.linalg.inv(jacobian))
    b = np.zeros((len(elements), 4, 3, 16))
    for inode in range(8):
        b[:, :, 0, 2*inode] = grad[:, :, inode, 0]
        b[:, :, 1, 2*inode+1] = grad[:, :, inode, 1]
        b[:, :, 2, 2*inode] = grad[:, :, inode, 1]
        b[:, :, 2, 2*inode+1] = grad[:, :, inode, 0]
    dofs = np.empty((len(elements), 16), dtype=np.int64)
    dofs[:, 0::2], dofs[:, 1::2] = 2*elements, 2*elements+1
    mass = np.einsum("qi,qj,eq->eij", shape, shape, det)
    stiff = np.einsum("eqia,eqja,eq->eij", grad, grad, det)
    a = np.sqrt(3/5)
    points = np.asarray([(x, y) for y in (-a, 0, a) for x in (-a, 0, a)])
    weights_1d = (5/9, 8/9, 5/9)
    weights = np.asarray([
        weights_1d[ix] * weights_1d[iy]
        for iy in range(3) for ix in range(3)
    ])
    _, full_derivative = _q8_at(points)
    full_jac = np.einsum("qia,eib->eqab", full_derivative, coordinates)
    full_det = np.linalg.det(full_jac) * weights[None, :]
    if np.any(full_det <= 0):
        raise RuntimeError(f"nonpositive full Q8 Jacobian: {full_det.min()}")
    full_grad = np.einsum(
        "qia,eqba->eqib", full_derivative, np.linalg.inv(full_jac)
    )
    full_b = np.zeros((len(elements), 9, 3, 16))
    for inode in range(8):
        full_b[:, :, 0, 2*inode] = full_grad[:, :, inode, 0]
        full_b[:, :, 1, 2*inode+1] = full_grad[:, :, inode, 1]
        full_b[:, :, 2, 2*inode] = full_grad[:, :, inode, 1]
        full_b[:, :, 2, 2*inode+1] = full_grad[:, :, inode, 0]
    elastic = np.asarray((
        (LAMBDA+2*MU, LAMBDA, 0), (LAMBDA, LAMBDA+2*MU, 0), (0, 0, MU)
    ))
    full_ke = np.einsum(
        "eqia,ij,eqjb,eq->eab", full_b, elastic, full_b, full_det
    )
    hourglass = np.zeros((len(elements), 16, 16))
    for index, (element_b, xy) in enumerate(zip(b, coordinates)):
        _, _, vh = np.linalg.svd(element_b.reshape(12, 16), full_matrices=True)
        null_projector = vh[12:].T @ vh[12:]
        rigid = np.zeros((16, 3))
        rigid[0::2, 0], rigid[1::2, 1] = 1, 1
        rigid[0::2, 2], rigid[1::2, 2] = -xy[:, 1], xy[:, 0]
        qr, _ = np.linalg.qr(rigid)
        candidate = null_projector - qr @ qr.T
        values, vectors = np.linalg.eigh(.5*(candidate+candidate.T))
        projector = (vectors*np.clip(values, 0, None)) @ vectors.T
        stabilization = projector @ full_ke[index] @ projector
        hourglass[index] = .5*(stabilization+stabilization.T)
    return shape, grad, b, dofs, det, mass, stiff, hourglass


def make_solvers(
    nodes_np, elem_np, shape_np, grad_np, b_np, dofs_np, det_np,
    mass_np, stiff_np, hourglass_np, free_u_np, fixed_d_np, split,
    diffusion_tol, diffusion_maxiter, steps_per_cycle,
):
    dtype = jnp.float64
    elem, dofs = jnp.asarray(elem_np), jnp.asarray(dofs_np)
    shape, grad, b, det, mass, stiff, hourglass = (
        jnp.asarray(value, dtype=dtype)
        for value in (shape_np, grad_np, b_np, det_np, mass_np, stiff_np, hourglass_np)
    )
    fixed_d = jnp.asarray(fixed_d_np, dtype=dtype)
    nnode, ndof = len(nodes_np), 2*len(nodes_np)
    tiny = jnp.asarray(1e-30, dtype=dtype)
    hydro_extrapolation = jnp.asarray(np.linalg.pinv(shape_np), dtype=dtype)
    mechanics = FixedTopologyCudssSolver(dofs_np, free_u_np)
    phase = FixedTopologyCudssSolver(elem_np, 1-fixed_d_np)

    def scatter(indices, values, size):
        return jnp.zeros(size, dtype=dtype).at[indices.reshape(-1)].add(
            values.reshape(-1)
        )

    def state(u, d):
        strain = jnp.einsum("eqij,ej->eqi", b, u[dofs])
        trace = strain[..., 0]+strain[..., 1]
        strain2 = strain[..., 0]**2+strain[..., 1]**2+.5*strain[..., 2]**2
        dev2 = (
            (strain[..., 0]-trace/3)**2+(strain[..., 1]-trace/3)**2
            +(trace/3)**2+.5*strain[..., 2]**2
        )
        phi = jnp.einsum("qi,ei->eq", shape, d[elem])
        degradation = (1-jnp.clip(phi, 0, 1))**2+KAPPA
        if split == "isotropic":
            psi = .5*LAMBDA*trace**2+MU*strain2
            c11, c12 = degradation*(LAMBDA+2*MU), degradation*LAMBDA
            hydro = degradation*K_BULK*trace
        elif split == "spectral":
            mean = trace/2
            radius = jnp.sqrt(
                ((strain[..., 0]-strain[..., 1])/2)**2+(strain[..., 2]/2)**2
            )
            e1, e2 = mean+radius, mean-radius
            psi = .5*LAMBDA*jnp.maximum(trace, 0)**2 + MU*(
                jnp.maximum(e1, 0)**2+jnp.maximum(e2, 0)**2
            )
            c11, c12 = degradation*(LAMBDA+2*MU), degradation*LAMBDA
            hydro = degradation*K_BULK*trace
        else:
            psi = .5*K_BULK*jnp.maximum(trace, 0)**2+MU*dev2
            volumetric = K_BULK*jnp.where(trace >= 0, degradation, 1)
            c11 = volumetric+4*MU*degradation/3
            c12 = volumetric-2*MU*degradation/3
            hydro = volumetric*trace
        zero = jnp.zeros_like(degradation)
        constitutive = jnp.stack((
            jnp.stack((c11, c12, zero), -1),
            jnp.stack((c12, c11, zero), -1),
            jnp.stack((zero, zero, MU*degradation), -1),
        ), -2)
        return psi, degradation*psi, constitutive, hydro

    @jax.jit
    def assemble_u(d, boundary, previous):
        _, _, constitutive, _ = state(previous+boundary, d)
        ke = jnp.einsum("eqia,eqij,eqjb,eq->eab", b, constitutive, b, det)
        phi = jnp.einsum("qi,ei->eq", shape, d[elem])
        mean_g = jnp.mean((1-jnp.clip(phi, 0, 1))**2+KAPPA, axis=1)
        return ke+mean_g[:, None, None]*hourglass

    zero_force = jnp.zeros(ndof, dtype=dtype)

    def solve_u(d, boundary, previous):
        return mechanics.solve(assemble_u(d, boundary, previous), boundary, zero_force)

    @jax.jit
    def fatigue_factor(alpha_bar):
        ratio = 2*ALPHA_T/jnp.maximum(alpha_bar+ALPHA_T, tiny)
        return jnp.where(alpha_bar <= ALPHA_T, 1, ratio**2)

    @jax.jit
    def update_fatigue(alpha, previous, alpha_bar):
        increment = alpha-previous
        return alpha_bar+jnp.where(alpha*increment > 0, jnp.abs(increment), 0)

    @jax.jit
    def assemble_d(history, toughness):
        coefficient = GC0*toughness/L0+2*history
        ke = jnp.einsum(
            "eq,eqia,eqja,eq->eij", GC0*toughness*L0, grad, grad, det
        )
        ke += jnp.einsum("eq,qi,qj,eq->eij", coefficient, shape, shape, det)
        rhs_e = jnp.einsum("eq,qi,eq->ei", 2*history, shape, det)
        return ke, scatter(elem, rhs_e, nnode)

    def solve_d(history, toughness, previous):
        del previous
        ke, force = assemble_d(history, toughness)
        return phase.solve(ke, fixed_d, force)

    area = jnp.sum(det, axis=1)
    nodal_weight = scatter(
        elem, jnp.broadcast_to((area/8)[:, None], elem.shape), nnode
    )
    dt = jnp.asarray(1/(FREQUENCY*steps_per_cycle), dtype=dtype)
    beta = jnp.asarray(V_H/(R_GAS*TEMPERATURE*1000), dtype=dtype)

    @jax.jit
    def solve_c(cold, hydro, fixed, cenv):
        local_hydro = jnp.einsum("iq,eq->ei", hydro_extrapolation, hydro)
        nodal_hydro = scatter(
            elem, local_hydro*(area/8)[:, None], nnode
        )/jnp.maximum(nodal_weight, tiny)
        hydro_gradient = jnp.einsum("eqia,ei->eqa", grad, nodal_hydro[elem])
        drift_test = jnp.einsum("eqia,eqa->eqi", grad, hydro_gradient)
        ae = mass+dt*D_H*stiff-dt*D_H*beta*jnp.einsum(
            "eqi,qj,eq->eij", drift_test, shape, det
        )
        rhs_e = jnp.einsum("eij,ej->ei", mass, cold[elem])
        fixed_float = fixed.astype(dtype)
        free, boundary = 1-fixed_float, fixed_float*cenv

        def kmv(x):
            return scatter(elem, jnp.einsum("eij,ej->ei", ae, x[elem]), nnode)

        rhs = free*(scatter(elem, rhs_e, nnode)-kmv(boundary))
        operator = lambda x: free*kmv(free*x)+fixed_float*x
        diagonal = free*scatter(
            elem, jnp.diagonal(ae, axis1=1, axis2=2), nnode
        )+fixed_float
        solution, _ = bicgstab(
            operator, rhs, x0=free*cold, tol=diffusion_tol, atol=0,
            maxiter=diffusion_maxiter,
            M=lambda residual: residual/jnp.maximum(jnp.abs(diagonal), tiny),
        )
        residual = jnp.linalg.norm(operator(solution)-rhs)/jnp.maximum(
            jnp.linalg.norm(rhs), tiny
        )
        return jnp.maximum(boundary+free*solution, 0), residual

    @jax.jit
    def qois(u, d):
        psi, alpha, _, hydro = state(u, d)
        phi = jnp.einsum("qi,ei->eq", shape, d[elem])
        phi_gradient = jnp.einsum("eqia,ei->eqa", grad, d[elem])
        gamma = jnp.sum(
            det*(phi**2/(2*L0)+.5*L0*jnp.sum(phi_gradient**2, axis=-1))
        )
        return psi, alpha, gamma, hydro

    return (
        solve_u, fatigue_factor, update_fatigue, solve_d, solve_c, qois,
        mechanics, phase,
    )


def displacement(step: int, steps_per_cycle: int) -> float:
    phase = step/steps_per_cycle
    if phase <= .25:
        return 4*UAMP*phase
    if phase <= .75:
        return 2*UAMP-4*UAMP*phase
    return -4*UAMP+4*UAMP*phase


def hfactor(concentration, delta_gb=DELTA_GB):
    theta = concentration/(concentration+jnp.exp(-delta_gb/(R_GAS*TEMPERATURE)))
    return jnp.clip(1-CHI*theta, 0, 1)


def crack_extension(nodes, elements, shape, damage) -> float:
    del shape
    coordinates = nodes[elements]
    # Ma et al. Eq. (25) defines the crack surface by d=0.95.  Evaluate
    # that contour directly on every Q8 edge lying on the propagation
    # centreline; checking only the four Gauss points systematically lags it.
    on_line = np.isclose(coordinates[..., 1], 0, atol=1e-12)
    line_elements = np.count_nonzero(on_line, axis=1) >= 2
    xi = np.linspace(-1, 1, 65)
    points = np.concatenate((
        np.column_stack((xi, -np.ones_like(xi))),
        np.column_stack((np.ones_like(xi), xi)),
        np.column_stack((xi, np.ones_like(xi))),
        np.column_stack((-np.ones_like(xi), xi)),
    ))
    edge_shape, _ = _q8_at(points)
    line_coordinates = coordinates[line_elements]
    xyz = np.einsum("qi,eij->eqj", edge_shape, line_coordinates)
    phi = np.einsum(
        "qi,ei->eq", edge_shape, damage[elements[line_elements]]
    )
    mask = (
        np.isclose(xyz[..., 1], 0, atol=1e-10)
        & (xyz[..., 0] >= A0-1e-12)
        & (xyz[..., 0] <= WIDTH+1e-12)
    )
    x = xyz[..., 0][mask]
    values = phi[mask]
    if not np.any(values >= .95):
        return 0.0
    # Duplicate samples from the upper/lower elements are harmless: use the
    # furthest reconstructed point on the d=0.95 contour.
    tip_index = np.argmax(np.where(values >= .95, x, -np.inf))
    tip = float(x[tip_index])
    ahead = (x > tip) & (values < .95)
    if np.any(ahead):
        next_index = np.argmin(np.where(ahead, x, np.inf))
        x1, d1 = tip, float(values[tip_index])
        x2, d2 = float(x[next_index]), float(values[next_index])
        if x2 > x1 and d1 != d2:
            tip = x1 + (x2-x1)*(d1-.95)/(d1-d2)
    return min(.5, max(0, tip-A0))


def write_csv(path: Path, rows: list[dict]) -> None:
    if rows:
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    if args.cycles < 1 or args.steps_per_cycle % 4:
        raise ValueError("cycles > 0 and steps-per-cycle divisible by four are required")
    mesh_path = args.mesh.expanduser().resolve()
    if not mesh_path.is_file():
        raise FileNotFoundError(f"required mesh does not exist: {mesh_path}")
    load_jax_runtime()
    jax.config.update("jax_enable_x64", True)
    devices = jax.devices()
    if not any(device.platform == "gpu" for device in devices):
        raise RuntimeError(f"NVIDIA GPU required; devices={devices}")
    out = (
        args.outdir.expanduser().resolve() if args.outdir else
        repository_root()/"outputs"/"accelerated"/args.case_label
        /f"q8_{args.hydrogen_wtppm:g}wppm"
    )
    out.mkdir(parents=True, exist_ok=True)
    nodes, elements, groups = load_mesh(mesh_path)
    shape, grad, b, dofs, det, mass, stiff, hourglass = preprocess(nodes, elements)
    nnode, ndof = len(nodes), 2*len(nodes)
    anchor = groups["bottom"][np.argmax(nodes[groups["bottom"], 0])]
    fixed_u = np.unique(np.r_[2*groups["bottom"]+1, 2*groups["top"]+1, 2*anchor])
    free_u = np.ones(ndof)
    free_u[fixed_u] = 0
    fixed_d = np.zeros(nnode)
    fixed_d[groups["crack"]] = 1
    top_y = 2*groups["top"]+1
    outer = (
        np.isclose(nodes[:, 0], 0) | np.isclose(nodes[:, 0], WIDTH)
        | np.isclose(nodes[:, 1], -HEIGHT/2) | np.isclose(nodes[:, 1], HEIGHT/2)
    )
    solvers = make_solvers(
        nodes, elements, shape, grad, b, dofs, det, mass, stiff, hourglass,
        free_u, fixed_d, args.split, args.diffusion_tol,
        args.diffusion_maxiter, args.steps_per_cycle,
    )
    solve_u, fatigue_factor, update_fatigue, solve_d, solve_c, qois = solvers[:6]
    direct_solvers = solvers[6:]
    cenv = args.hydrogen_wtppm*CH_1WPPM
    initial_q = jnp.zeros((len(elements), len(shape)), dtype=jnp.float64)
    d0_j, d0_residual = solve_d(
        initial_q, jnp.ones_like(initial_q), jnp.asarray(fixed_d)
    )
    jax.block_until_ready(d0_j)
    initial_residual = float(jax.device_get(d0_residual))
    if initial_residual > args.linear_residual_tol:
        raise RuntimeError(
            f"initial AT2 crack equilibrium failed: residual={initial_residual:.3e}"
        )
    d0 = np.asarray(jax.device_get(jnp.clip(d0_j, 0, 1)))
    _, _, gamma0_j, _ = qois(jnp.zeros(ndof), jnp.asarray(d0))
    gamma0 = float(jax.device_get(gamma0_j))
    checkpoint_path, csv_path = out/"checkpoint_latest.npz", out/"results.csv"
    if args.resume and checkpoint_path.is_file():
        checkpoint = np.load(checkpoint_path)
        completed = int(checkpoint["completed_steps"])
        u, d = jnp.asarray(checkpoint["u"]), jnp.asarray(checkpoint["d"])
        history = jnp.asarray(checkpoint["history"])
        previous = jnp.asarray(checkpoint["alpha_previous"])
        alpha_bar = jnp.asarray(checkpoint["alpha_bar"])
        concentration = jnp.asarray(checkpoint["concentration"])
        rows = json.loads(str(checkpoint["rows_json"]))
    else:
        completed, rows = 0, []
        u, d = jnp.zeros(ndof), jnp.asarray(d0)
        history = jnp.zeros((len(elements), len(shape)))
        previous, alpha_bar = jnp.zeros_like(history), jnp.zeros_like(history)
        concentration = jnp.full(nnode, cenv)
    print("="*78)
    print("Q8 FP64 fixed-topology CSR + NVIDIA nvmath cuDSS")
    print(f"JAX {jax.__version__}; Python {platform.python_version()}; {devices}")
    print(
        f"{nnode} nodes, {len(elements)} Q8 elements; split={args.split}; "
        f"H={args.hydrogen_wtppm:g} wt-ppm; mode={args.hydrogen_mode}"
    )
    print("="*78)
    started = cycle_started = time.perf_counter()
    maximum_residual = 0.0
    try:
        for istep in range(completed, args.cycles*args.steps_per_cycle):
            local, cycle = istep % args.steps_per_cycle+1, istep//args.steps_per_cycle+1
            if local == 1:
                cycle_started = time.perf_counter()
            boundary_np = np.zeros(ndof)
            boundary_np[top_y] = displacement(local, args.steps_per_cycle)
            boundary = jnp.asarray(boundary_np)
            damage_at_step_start = d
            u, mechanics_residual = solve_u(d, boundary, u)
            psi, alpha, _, hydro = qois(u, d)
            if args.hydrogen_mode == "transient" and cenv > 0:
                phi_q = jnp.einsum("qi,ei->eq", jnp.asarray(shape), d[jnp.asarray(elements)])
                concentration, concentration_residual = solve_c(
                    concentration, hydro*(1-jnp.clip(phi_q, 0, 1))**2,
                    jnp.asarray(outer)|(d >= .95), cenv,
                )
            else:
                concentration_residual = jnp.asarray(0.0)
            concentration_q = jnp.einsum(
                "qi,ei->eq", jnp.asarray(shape), concentration[jnp.asarray(elements)]
            )
            hydrogen_factor = hfactor(concentration_q, args.delta_gb_j_per_mol)
            alpha_bar_step = update_fatigue(alpha, previous, alpha_bar)
            fatigue = fatigue_factor(alpha_bar_step)
            history_step = jnp.maximum(history, psi)
            stagger_error = np.inf
            for iteration in range(1, args.max_stagger+1):
                old_damage = d
                if iteration > 1:
                    u, mechanics_residual = solve_u(d, boundary, u)
                    psi, _, _, _ = qois(u, d)
                    history_step = jnp.maximum(history_step, psi)
                trial, damage_residual = solve_d(
                    history_step, hydrogen_factor*fatigue, d
                )
                d = jnp.clip(jnp.maximum(trial, damage_at_step_start), 0, 1)
                jax.block_until_ready(d)
                stagger_error = float(jax.device_get(
                    jnp.linalg.norm(d-old_damage)/(jnp.linalg.norm(d)+1e-30)
                ))
                if stagger_error < args.stagger_tol:
                    break
            residuals = [
                float(jax.device_get(value)) for value in
                (mechanics_residual, damage_residual, concentration_residual)
            ]
            maximum_residual = max(maximum_residual, *residuals)
            if not all(
                np.isfinite(value) and value <= args.linear_residual_tol
                for value in residuals
            ):
                raise RuntimeError(
                    f"residual limit: u={residuals[0]:.3e}, "
                    f"d={residuals[1]:.3e}, c={residuals[2]:.3e}"
                )
            history, alpha_bar, previous = history_step, alpha_bar_step, alpha
            if local == args.steps_per_cycle:
                _, _, gamma_j, _ = qois(u, d)
                damage_host = np.asarray(jax.device_get(d))
                concentration_host = np.asarray(jax.device_get(concentration))
                alpha_bar_host = np.asarray(jax.device_get(alpha_bar))
                fatigue_host = np.asarray(jax.device_get(fatigue))
                hydrogen_host = np.asarray(jax.device_get(hydrogen_factor))
                row = {
                    "cycle": cycle, "hydrogen_wtppm": args.hydrogen_wtppm,
                    "crack_extension_mm": crack_extension(
                        nodes, elements, shape, damage_host
                    ),
                    "crack_surface_extension_mm": max(
                        0, float(jax.device_get(gamma_j))-gamma0
                    ),
                    "maximum_hydrogen_wtppm": (
                        float(concentration_host.max()/CH_1WPPM) if cenv else 0
                    ),
                    "minimum_local_hydrogen_factor": float(hydrogen_host.min()),
                    "max_alpha_bar_mpa": float(alpha_bar_host.max()),
                    "min_fatigue_factor": float(fatigue_host.min()),
                    "last_stagger_iterations": iteration,
                    "last_stagger_error": stagger_error,
                    "cycle_wall_s": time.perf_counter()-cycle_started,
                    "cumulative_wall_s": time.perf_counter()-started,
                }
                rows.append(row)
                write_csv(csv_path, rows)
                if cycle <= 3 or cycle % args.print_every == 0:
                    print(
                        f"cycle={cycle:4d} "
                        f"da_gamma={row['crack_surface_extension_mm']:.6f} mm "
                        f"da_tip95={row['crack_extension_mm']:.6f} mm "
                        f"Cmax={row['maximum_hydrogen_wtppm']:.3f} wt-ppm "
                        f"fp={iteration} dt={row['cycle_wall_s']:.2f}s", flush=True
                    )
                if cycle % args.checkpoint_every == 0 or cycle == args.cycles:
                    np.savez_compressed(
                        checkpoint_path, completed_steps=istep+1, u=np.asarray(u),
                        d=damage_host, history=np.asarray(history),
                        alpha_previous=np.asarray(previous), alpha_bar=alpha_bar_host,
                        concentration=concentration_host,
                        rows_json=np.asarray(json.dumps(rows)),
                    )
                if row["crack_extension_mm"] >= .49:
                    print("early stop: crack traversed ligament", flush=True)
                    break
    finally:
        for direct_solver in direct_solvers:
            direct_solver.close()
    np.savez_compressed(
        out/"final_state.npz", nodes=nodes, elements=elements,
        u=np.asarray(u).reshape(-1, 2), d=np.asarray(d), history=np.asarray(history),
        alpha_bar=np.asarray(alpha_bar), concentration=np.asarray(concentration),
    )
    summary = {
        "version": VERSION, "paper": args.paper_label,
        "device": str(devices[0]), "element": "Q8 serendipity",
        "integration": "2x2 Gauss reduced integration",
        "linear_solver": "fixed CSR + nvmath cuDSS (BiCGSTAB diffusion)",
        "dtype": "float64", "mesh": str(mesh_path), "nodes": nnode,
        "elements": len(elements), "split": args.split,
        "hydrogen_wtppm": args.hydrogen_wtppm,
        "hydrogen_mode": args.hydrogen_mode, "cycles_completed": len(rows),
        "final_crack_extension_mm": (
            rows[-1]["crack_extension_mm"] if rows else None
        ),
        "final_crack_surface_extension_mm": (
            rows[-1]["crack_surface_extension_mm"] if rows else None
        ),
        "maximum_linear_residual": maximum_residual,
        "wall_s": time.perf_counter()-started,
    }
    (out/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"completed: {out}", flush=True)


if __name__ == "__main__":
    main()
