#!/usr/bin/env python3
"""Full-GPU P1/cuDSS reproduction of Yang et al. (2026) JMRT pipeline.

JAX assembles every element residual, tangent, and transport operator in FP64
on the GPU. CuPy consumes those arrays through DLPack and updates three
fixed-topology CSR systems whose cuDSS symbolic plans are reused for mechanics,
phase field, and nonsymmetric stress-assisted diffusion. Units are mm, N, MPa,
s, and mol.

Irreversibility uses the history field and a post-solve projection
max(d_at_step_start, d_trial). Base-metal (BM) nodes are always fixed at d=0.
No threshold-based d=1 crack set is imposed because that discontinuous update
causes a spurious stiffness loss during rapid crack propagation.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from yang2026_pipeline_p1_cudss import (  # noqa: E402
    load_pipeline,
    outer_surface_nodes,
    penetration_mm,
    pressure_force,
    refine_once,
)


REPO = Path(__file__).resolve().parents[3]
DEFAULT_OUT = (
    REPO
    / "outputs"
    / "yang_2026_jmrt_pipeline_fullgpu"
    / "fig14"
    / "p10"
)
VERSION = "yang2026_jmrt_pipeline_p1_fullgpu_v2"

E, NU = 210000.0, 0.3
LAMBDA = E * NU / ((1.0 + NU) * (1.0 - 2.0 * NU))
MU = E / (2.0 * (1.0 + NU))
K_BULK = LAMBDA + 2.0 * MU / 3.0
L0, KAPPA = 0.2, 1.0e-6
LOAD_RATIO, TEMPERATURE = 0.5, 295.0
RHO_M, M_H = 7.85e-6, 1.008e-3
C_ENV = 1.0e-6 * RHO_M / M_H
V_H, R_GAS_NMM = 2000.0, 8.314e3
CRACK_DIFFUSIVITY_FACTOR = 1.0e5
PHI_THRESHOLD, PHI_WIDTH = 0.95, 0.02
NEWTON_TOL, NEWTON_MAXITER = 1.0e-8, 40


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--pressure-mpa", type=float, default=10.0)
    result.add_argument("--blocks", type=int, default=82)
    result.add_argument("--delta-n", type=int, default=100)
    result.add_argument("--uniform-refine", type=int, default=0)
    result.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    result.add_argument("--restart-state", type=Path)
    result.add_argument("--start-cycle", type=int, default=0)
    result.add_argument("--max-stagger", type=int, default=20)
    result.add_argument("--stagger-tol", type=float, default=1.0e-6)
    result.add_argument(
        "--stagger-relaxation",
        type=float,
        default=1.0,
        help="fixed-point relaxation in (0, 1]; one is the original iteration",
    )
    result.add_argument(
        "--accept-unconverged",
        action="store_true",
        help="accept the last stagger iterate at the cap, matching released policy",
    )
    result.add_argument("--linear-residual-tol", type=float, default=1.0e-4)
    result.add_argument(
        "--phase-field-residual-tol", type=float, default=1.0e-3
    )
    result.add_argument("--snapshot-cycles", type=int, nargs="*", default=())
    result.add_argument("--save-state", action="store_true")
    result.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="save restart_checkpoint.npz every N accepted blocks; zero disables",
    )
    result.add_argument("--mesh-check", action="store_true")
    result.add_argument(
        "--stop-at-max-d",
        type=float,
        default=0.0,
        help="stop once max(d) reaches this value; zero disables early stop",
    )
    result.add_argument(
        "--time-mode",
        choices=("source", "physical"),
        default="source",
        help="source applies the released 1e7 diffusivity acceleration",
    )
    return result


def validate_arguments(args: argparse.Namespace) -> None:
    if (
        args.pressure_mpa <= 0.0
        or args.blocks < 1
        or args.delta_n < 1
        or args.uniform_refine < 0
        or args.start_cycle < 0
        or args.max_stagger < 1
        or args.stagger_tol <= 0.0
        or not 0.0 < args.stagger_relaxation <= 1.0
        or args.checkpoint_every < 0
    ):
        raise ValueError("invalid positive load, step, refinement, or tolerance")
    if args.start_cycle and args.restart_state is None:
        raise ValueError("--start-cycle requires --restart-state")
    if not 0.0 <= args.stop_at_max_d <= 1.0:
        raise ValueError("--stop-at-max-d must be in [0, 1]")


def load_runtime() -> None:
    """Import CUDA-only dependencies after the mesh-only exit point."""
    global jax, jnp
    try:
        import jax as jax_module
        import jax.numpy as jax_numpy
    except ImportError as exc:
        raise RuntimeError("the full-GPU driver requires CUDA-enabled JAX") from exc
    jax, jnp = jax_module, jax_numpy


class FixedTopologyCudssSolver:
    """GPU CSR assembly and reusable cuDSS plan for one FE topology."""

    def __init__(
        self,
        name: str,
        connectivity: np.ndarray,
        static_fixed: np.ndarray,
        *,
        diagonal_regularization: float = 0.0,
    ) -> None:
        try:
            import cupy as cp
            import nvmath
            from cupyx.scipy.sparse import csr_matrix
        except ImportError as exc:
            raise RuntimeError(
                "full-GPU solves require cupy-cuda12x and nvmath-python[cu12]"
            ) from exc

        self.name = name
        self.cp = cp
        self.nvmath = nvmath
        self.diagonal_regularization = float(diagonal_regularization)
        size, width = len(static_fixed), connectivity.shape[1]
        rows = np.repeat(connectivity, width, axis=1).reshape(-1)
        cols = np.tile(connectivity, (1, width)).reshape(-1)
        keys = rows.astype(np.int64) * size + cols
        unique, inverse = np.unique(keys, return_inverse=True)
        unique_rows, unique_cols = unique // size, unique % size
        indptr = np.searchsorted(
            unique_rows, np.arange(size + 1), side="left"
        ).astype(np.int32)
        diagonal = np.searchsorted(unique, np.arange(size) * (size + 1))
        if np.any(unique[diagonal] != np.arange(size) * (size + 1)):
            raise RuntimeError(f"{name} topology lacks one or more diagonal entries")

        self.size = size
        self.inverse = cp.asarray(inverse, dtype=cp.int32)
        self.indices = cp.asarray(unique_cols, dtype=cp.int32)
        self.indptr = cp.asarray(indptr, dtype=cp.int32)
        self.unique_rows = cp.asarray(unique_rows, dtype=cp.int32)
        self.unique_cols = cp.asarray(unique_cols, dtype=cp.int32)
        self.all_diagonal = cp.asarray(diagonal, dtype=cp.int32)
        self.static_fixed = cp.asarray(static_fixed, dtype=cp.bool_)
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
        self.solve_calls = 0
        self.plan_calls = 0
        self.assembly_s = 0.0
        self.plan_s = 0.0
        self.factor_s = 0.0
        self.solve_s = 0.0

    def solve(
        self,
        element_matrix,
        boundary,
        assembled_force,
        *,
        dynamic_fixed=None,
    ):
        """Solve with static/dynamic Dirichlet rows and a true relative residual."""
        cp = self.cp
        jax.block_until_ready(element_matrix)
        tic = time.perf_counter()
        values = cp.from_dlpack(element_matrix).reshape(-1)
        boundary_gpu = cp.from_dlpack(boundary)
        force_gpu = cp.from_dlpack(assembled_force)
        fixed = self.static_fixed
        if dynamic_fixed is not None:
            fixed = fixed | cp.from_dlpack(dynamic_fixed).astype(cp.bool_)
        free = (~fixed).astype(cp.float64)
        entry_mask = free[self.unique_rows] * free[self.unique_cols]
        fixed_diagonal = self.all_diagonal[fixed]

        self.data.fill(0.0)
        cp.add.at(self.data, self.inverse, values)
        self.rhs[...] = free * (force_gpu - self.matrix @ boundary_gpu)
        self.constrained_data[...] = self.data * entry_mask
        self.constrained_data[fixed_diagonal] = 1.0
        if self.diagonal_regularization:
            diagonal_values = cp.abs(self.constrained_data[self.all_diagonal])
            scale = cp.maximum(cp.max(diagonal_values), 1.0)
            self.constrained_data[self.all_diagonal[~fixed]] += (
                self.diagonal_regularization * scale
            )
        cp.cuda.Stream.null.synchronize()
        self.assembly_s += time.perf_counter() - tic

        if self.solver is None:
            tic = time.perf_counter()
            self.solver = self.nvmath.sparse.advanced.DirectSolver(
                self.constrained, self.rhs
            )
            self.solver.plan()
            cp.cuda.Stream.null.synchronize()
            self.plan_s += time.perf_counter() - tic
            self.plan_calls += 1

        tic = time.perf_counter()
        self.solver.factorize()
        cp.cuda.Stream.null.synchronize()
        self.factor_s += time.perf_counter() - tic
        tic = time.perf_counter()
        solution = self.solver.solve()
        cp.cuda.Stream.null.synchronize()
        self.solve_s += time.perf_counter() - tic
        self.solve_calls += 1
        residual = cp.linalg.norm(self.constrained @ solution - self.rhs)
        residual /= cp.maximum(cp.linalg.norm(self.rhs), 1.0e-30)
        full = boundary_gpu + free * solution
        return jnp.from_dlpack(full), jnp.from_dlpack(residual)

    def stats(self) -> dict[str, float | int | str]:
        return {
            "name": self.name,
            "size": self.size,
            "nnz": int(self.data.size),
            "solve_calls": self.solve_calls,
            "plan_calls": self.plan_calls,
            "assembly_s": self.assembly_s,
            "plan_s": self.plan_s,
            "factor_s": self.factor_s,
            "solve_s": self.solve_s,
        }

    def close(self) -> None:
        if self.solver is not None:
            self.solver.free()
            self.solver = None


def preprocess(
    nodes: np.ndarray, tri: np.ndarray, areas: np.ndarray
) -> tuple[np.ndarray, ...]:
    """Create constant-strain P1 operators on the host before timestepping."""
    x, y = nodes[:, 0], nodes[:, 1]
    i, j, k = tri.T
    gx = np.stack((y[j] - y[k], y[k] - y[i], y[i] - y[j]), axis=1)
    gy = np.stack((x[k] - x[j], x[i] - x[k], x[j] - x[i]), axis=1)
    gx /= 2.0 * areas[:, None]
    gy /= 2.0 * areas[:, None]
    bmat = np.zeros((len(tri), 3, 6), dtype=np.float64)
    for local in range(3):
        bmat[:, 0, 2 * local] = gx[:, local]
        bmat[:, 1, 2 * local + 1] = gy[:, local]
        bmat[:, 2, 2 * local] = gy[:, local]
        bmat[:, 2, 2 * local + 1] = gx[:, local]
    dofs = np.empty((len(tri), 6), dtype=np.int64)
    dofs[:, 0::2], dofs[:, 1::2] = 2 * tri, 2 * tri + 1
    mass = areas[:, None, None] * np.asarray(
        ((2.0, 1.0, 1.0), (1.0, 2.0, 1.0), (1.0, 1.0, 2.0))
    ) / 12.0
    grad = areas[:, None, None] * (
        gx[:, :, None] * gx[:, None, :]
        + gy[:, :, None] * gy[:, None, :]
    )
    return gx, gy, bmat, dofs, mass, grad


def make_solvers(
    *,
    nodes_np: np.ndarray,
    tri_np: np.ndarray,
    areas_np: np.ndarray,
    bmat_np: np.ndarray,
    dofs_np: np.ndarray,
    mass_np: np.ndarray,
    grad_np: np.ndarray,
    free_u_np: np.ndarray,
    base_nodes_np: np.ndarray,
    internal_nodes_np: np.ndarray,
    external_force_np: np.ndarray,
    gc_cell_np: np.ndarray,
    alpha_t_cell_np: np.ndarray,
    fatigue_n_cell_np: np.ndarray,
    diffusivity_cell_np: np.ndarray,
    c_environment: float,
    diffusion_dt: float,
    dtype,
):
    """Build JIT element kernels and three persistent cuDSS systems."""
    tri, dofs = jnp.asarray(tri_np), jnp.asarray(dofs_np)
    areas, bmat, mass, grad, gx, gy = (
        jnp.asarray(value, dtype=dtype)
        for value in (
            areas_np,
            bmat_np,
            mass_np,
            grad_np,
            bmat_np[:, 0, 0::2],
            bmat_np[:, 1, 1::2],
        )
    )
    gc_cell, alpha_t, fatigue_n, base_diffusivity = (
        jnp.asarray(value, dtype=dtype)
        for value in (
            gc_cell_np,
            alpha_t_cell_np,
            fatigue_n_cell_np,
            diffusivity_cell_np,
        )
    )
    nnode, ndof = len(nodes_np), 2 * len(nodes_np)
    tiny = jnp.asarray(1.0e-30, dtype=dtype)
    qp_shape = jnp.asarray(
        (
            (2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0),
            (1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0),
            (1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0),
        ),
        dtype=dtype,
    )
    external_force = jnp.asarray(external_force_np, dtype=dtype)
    zero_u = jnp.zeros(ndof, dtype=dtype)
    zero_d = jnp.zeros(nnode, dtype=dtype)
    c_boundary = jnp.zeros(nnode, dtype=dtype).at[internal_nodes_np].set(
        c_environment
    )
    base_mask = np.zeros(nnode, dtype=bool)
    base_mask[base_nodes_np] = True
    base_mask_gpu = jnp.asarray(base_mask)
    concentration_mask = np.zeros(nnode, dtype=bool)
    concentration_mask[internal_nodes_np] = True
    mechanics_direct = FixedTopologyCudssSolver(
        "mechanics",
        dofs_np,
        static_fixed=free_u_np == 0.0,
        diagonal_regularization=1.0e-12,
    )
    phase_direct = FixedTopologyCudssSolver(
        "phase",
        tri_np,
        static_fixed=base_mask,
    )
    concentration_direct = FixedTopologyCudssSolver(
        "concentration",
        tri_np,
        static_fixed=concentration_mask,
    )

    def scatter(ids, values, size):
        return jnp.zeros(size, dtype=dtype).at[ids.reshape(-1)].add(
            values.reshape(-1)
        )

    def strain_state(u):
        strain = jnp.einsum("eij,ej->ei", bmat, u[dofs])
        trace = strain[:, 0] + strain[:, 1]
        mean = trace / 2.0
        radius = jnp.sqrt(
            ((strain[:, 0] - strain[:, 1]) / 2.0) ** 2
            + (strain[:, 2] / 2.0) ** 2
        )
        eps_max = jnp.maximum(mean + radius, 0.0)
        return strain, trace, eps_max

    def smooth_positive(value):
        epsilon = jnp.asarray(1.0e-10, dtype=dtype)
        return 0.5 * (value + jnp.sqrt(value * value + epsilon * epsilon))

    def split_stress_energy(strain, trace):
        """Released Felino stress-eigenvalue tensile projection."""
        exx, eyy, gamma = strain[:, 0], strain[:, 1], strain[:, 2]
        sxx = 2.0 * MU * exx + LAMBDA * trace
        syy = 2.0 * MU * eyy + LAMBDA * trace
        sxy = MU * gamma
        centre = 0.5 * (sxx + syy)
        halfdiff = 0.5 * (sxx - syy)
        epsilon = jnp.asarray(1.0e-10, dtype=dtype)
        radius = jnp.sqrt(halfdiff**2 + sxy**2 + epsilon**2)
        high, low = centre + radius, centre - radius
        positive_high = smooth_positive(high)
        positive_low = smooth_positive(low)
        coefficient = (positive_high - positive_low) / (2.0 * radius)
        plus_xx = positive_low + coefficient * (sxx - low)
        plus_yy = positive_low + coefficient * (syy - low)
        plus_xy = coefficient * sxy
        plus = jnp.stack((plus_xx, plus_yy, plus_xy), axis=1)
        total = jnp.stack((sxx, syy, sxy), axis=1)
        history = 0.5 * (
            plus_xx * exx + plus_yy * eyy + plus_xy * gamma
        )
        return plus, total - plus, jnp.maximum(history, 0.0)

    def degradation_average(damage):
        d_qp = jnp.einsum(
            "qa,ea->eq", qp_shape, jnp.clip(damage[tri], 0.0, 1.0)
        )
        return jnp.mean(
            (1.0 - d_qp) ** 2 * (1.0 - KAPPA) + KAPPA, axis=1
        )

    @jax.jit
    def mechanical_terms(u, damage):
        strain, trace, _ = strain_state(u)
        degradation = degradation_average(damage)

        def stress_from_strain(value):
            value_trace = value[:, 0] + value[:, 1]
            plus, minus, _ = split_stress_energy(value, value_trace)
            return degradation[:, None] * plus + minus

        stress = stress_from_strain(strain)
        directions = jnp.eye(3, dtype=dtype)
        columns = jax.vmap(
            lambda direction: jax.jvp(
                stress_from_strain,
                (strain,),
                (jnp.broadcast_to(direction, strain.shape),),
            )[1]
        )(directions)
        tangent = jnp.transpose(columns, (1, 2, 0))
        local_force = areas[:, None] * jnp.einsum(
            "eji,ej->ei", bmat, stress
        )
        local_stiffness = areas[:, None, None] * jnp.einsum(
            "eia,eij,ejb->eab", bmat, tangent, bmat
        )
        internal = scatter(dofs, local_force, ndof)
        return internal, local_stiffness

    def solve_u(damage, previous):
        solution = previous
        linear_residual = jnp.asarray(0.0, dtype=dtype)
        nonlinear_residual = jnp.asarray(np.inf, dtype=dtype)
        free_u = jnp.asarray(free_u_np, dtype=dtype)
        for iteration in range(NEWTON_MAXITER + 1):
            internal, tangent = mechanical_terms(solution, damage)
            force_residual = internal - external_force
            numerator = jnp.linalg.norm(free_u * force_residual)
            denominator = jnp.maximum(
                jnp.maximum(
                    jnp.linalg.norm(free_u * external_force),
                    jnp.linalg.norm(internal),
                ),
                tiny,
            )
            nonlinear_residual = numerator / denominator
            if float(jax.device_get(nonlinear_residual)) <= NEWTON_TOL:
                return solution, nonlinear_residual, linear_residual, iteration
            increment, linear_residual = mechanics_direct.solve(
                tangent,
                zero_u,
                -force_residual,
            )
            solution = solution + increment
        raise RuntimeError(
            "mechanics Newton failed after "
            f"{NEWTON_MAXITER} updates; residual="
            f"{float(jax.device_get(nonlinear_residual)):.3e}"
        )

    @jax.jit
    def qois(u, damage):
        strain, trace, eps_max = strain_state(u)
        plus, minus, psi = split_stress_energy(strain, trace)
        degradation = degradation_average(damage)
        stress = degradation[:, None] * plus + minus
        szz0 = LAMBDA * trace
        szz_plus = smooth_positive(szz0)
        szz = degradation * szz_plus + szz0 - szz_plus
        sigma_h = (stress[:, 0] + stress[:, 1] + szz) / 3.0
        psi_eff = (
            2.0
            * E
            * eps_max**2
            * ((1.0 + LOAD_RATIO) / 2.0) ** 2
            * ((1.0 - LOAD_RATIO) / 2.0) ** fatigue_n
        )
        crack_area = jnp.sum(jnp.einsum("eij,ej->ei", mass, damage[tri]))
        return psi, psi_eff, crack_area, sigma_h

    @jax.jit
    def fatigue_factor(alpha_bar):
        asymptote = 2.0 * alpha_t / jnp.maximum(alpha_bar + alpha_t, tiny)
        return jnp.where(alpha_bar > alpha_t, asymptote**2, 1.0)

    @jax.jit
    def hydrogen_factor(concentration):
        cell_c = jnp.mean(concentration[tri], axis=1)
        cell_wppm = cell_c * 1.0e6 * M_H / RHO_M
        return jnp.clip(
            0.155 + 0.845 * jnp.exp(-24.1 * cell_wppm**2), 0.0, 1.0
        )

    @jax.jit
    def assemble_d(history, factor):
        local_gc = gc_cell * factor
        history_drive = (1.0 - KAPPA) * history
        matrix = (
            local_gc[:, None, None] * L0 * grad
            + (local_gc / L0 + 2.0 * history_drive)[:, None, None] * mass
        )
        rhs = scatter(
            tri,
            jnp.broadcast_to(
                (2.0 * history_drive * areas / 3.0)[:, None], tri.shape
            ),
            nnode,
        )
        return matrix, rhs

    def solve_d(history, factor, lower_bound):
        matrix, rhs = assemble_d(history, factor)
        lower = jnp.clip(lower_bound, 0.0, 1.0).at[base_nodes_np].set(0.0)
        solution, residual = phase_direct.solve(
            matrix,
            zero_d,
            rhs,
        )
        projected = jnp.clip(solution, lower, 1.0).at[base_nodes_np].set(0.0)
        return projected, residual

    nodal_weight = scatter(
        tri,
        jnp.broadcast_to((areas / 3.0)[:, None], tri.shape),
        nnode,
    )
    beta = jnp.asarray(V_H / (R_GAS_NMM * TEMPERATURE), dtype=dtype)
    inverse_dt = jnp.asarray(1.0 / diffusion_dt, dtype=dtype)

    @jax.jit
    def assemble_c(old_concentration, sigma_cell, damage):
        sigma_node = scatter(
            tri,
            jnp.broadcast_to(
                (sigma_cell * areas / 3.0)[:, None], tri.shape
            ),
            nnode,
        ) / jnp.maximum(nodal_weight, tiny)
        grad_sigma_x = jnp.sum(sigma_node[tri] * gx, axis=1)
        grad_sigma_y = jnp.sum(sigma_node[tri] * gy, axis=1)
        damage_cell = jnp.mean(damage[tri], axis=1)
        switch = 0.5 * (
            1.0 + jnp.tanh((damage_cell - PHI_THRESHOLD) / PHI_WIDTH)
        )
        diffusivity = base_diffusivity * (
            1.0 + CRACK_DIFFUSIVITY_FACTOR * switch
        )
        diffusion = diffusivity[:, None, None] * grad
        advection_row = -(
            gx * (diffusivity * beta * grad_sigma_x)[:, None]
            + gy * (diffusivity * beta * grad_sigma_y)[:, None]
        ) * areas[:, None] / 3.0
        matrix = inverse_dt * mass + diffusion + advection_row[:, :, None]
        local_rhs = inverse_dt * jnp.einsum(
            "eij,ej->ei", mass, old_concentration[tri]
        )
        return matrix, scatter(tri, local_rhs, nnode)

    def solve_c(old_concentration, sigma_cell, damage):
        matrix, rhs = assemble_c(old_concentration, sigma_cell, damage)
        concentration, residual = concentration_direct.solve(
            matrix, c_boundary, rhs
        )
        return jnp.maximum(concentration, 0.0), residual

    return {
        "solve_u": solve_u,
        "qois": qois,
        "fatigue_factor": fatigue_factor,
        "hydrogen_factor": hydrogen_factor,
        "solve_d": solve_d,
        "solve_c": solve_c,
        "direct": (
            mechanics_direct,
            phase_direct,
            concentration_direct,
        ),
    }


def mesh_data(uniform_refine: int):
    nodes, tri, areas, groups, boundaries, nodal_fields = load_pipeline()
    for _ in range(uniform_refine):
        nodes, tri, areas, groups, boundaries, nodal_fields = refine_once(
            nodes, tri, groups, boundaries, nodal_fields
        )
    fields = {
        name: values[tri].mean(axis=1)
        for name, values in nodal_fields.items()
    }
    outer_nodes = outer_surface_nodes(tri, boundaries)
    summary = {
        "nodes": len(nodes),
        "triangles": len(tri),
        "uniform_refine": uniform_refine,
        "internal_nodes": len(groups["internal"]),
        "internal_edges": len(boundaries["internal"]),
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
    return (
        nodes,
        tri,
        areas,
        groups,
        boundaries,
        fields,
        outer_nodes,
        summary,
    )


def save_npz(
    path: Path,
    *,
    nodes: np.ndarray,
    tri: np.ndarray,
    u,
    damage,
    history,
    alpha_bar,
    concentration,
    sigma_h,
    eta_haz: np.ndarray,
    cycle: int,
) -> None:
    np.savez_compressed(
        path,
        nodes=nodes,
        triangles=tri,
        displacement=np.asarray(jax.device_get(u)).reshape((-1, 2)),
        damage=np.asarray(jax.device_get(damage)),
        history=np.asarray(jax.device_get(history)),
        alpha_bar=np.asarray(jax.device_get(alpha_bar)),
        concentration=np.asarray(jax.device_get(concentration)),
        hydrostatic_stress=np.asarray(jax.device_get(sigma_h)),
        eta_haz=eta_haz,
        cycle=np.asarray(cycle),
    )


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    args = parser().parse_args(argv)
    validate_arguments(args)
    (
        nodes,
        tri,
        areas,
        groups,
        boundaries,
        fields,
        outer_nodes,
        mesh_summary,
    ) = mesh_data(args.uniform_refine)
    if args.mesh_check:
        print(json.dumps(mesh_summary, indent=2), flush=True)
        return

    load_runtime()
    jax.config.update("jax_enable_x64", True)
    devices = jax.devices()
    if not any(device.platform == "gpu" for device in devices):
        raise RuntimeError(f"NVIDIA GPU and FP64 required; devices={devices}")
    dtype = jnp.float64
    gx, gy, bmat, dofs, mass, grad = preprocess(nodes, tri, areas)
    nnode, ndof = len(nodes), 2 * len(nodes)
    fixed_nodes = np.unique(
        np.concatenate((groups["left"], groups["right"]))
    )
    fixed_dofs = np.column_stack(
        (2 * fixed_nodes, 2 * fixed_nodes + 1)
    ).reshape(-1)
    free_u = np.ones(ndof, dtype=np.float64)
    free_u[fixed_dofs] = 0.0
    external_force = pressure_force(
        nodes, boundaries["internal"], args.pressure_mpa
    )
    c_wppm = 0.243 * np.sqrt(args.pressure_mpa / 10.0)
    c_environment = c_wppm * C_ENV
    if args.time_mode == "source":
        diffusion_dt = 1.0
        diffusivity = fields["D_eff"] * 1.0e7
        acceleration = 1.0e7
    else:
        diffusion_dt = args.delta_n / 1.0e-5
        diffusivity = fields["D_eff"]
        acceleration = 1.0

    solvers = make_solvers(
        nodes_np=nodes,
        tri_np=tri,
        areas_np=areas,
        bmat_np=bmat,
        dofs_np=dofs,
        mass_np=mass,
        grad_np=grad,
        free_u_np=free_u,
        base_nodes_np=groups["base"],
        internal_nodes_np=groups["internal"],
        external_force_np=external_force,
        gc_cell_np=fields["Gc"],
        alpha_t_cell_np=fields["alpha_T"],
        fatigue_n_cell_np=fields["n"],
        diffusivity_cell_np=diffusivity,
        c_environment=c_environment,
        diffusion_dt=diffusion_dt,
        dtype=dtype,
    )
    solve_u = solvers["solve_u"]
    qois = solvers["qois"]
    fatigue_factor = solvers["fatigue_factor"]
    hydrogen_factor = solvers["hydrogen_factor"]
    solve_d = solvers["solve_d"]
    solve_c = solvers["solve_c"]
    direct_solvers = solvers["direct"]

    outdir = args.outdir.expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    snapshots = outdir / "snapshots"
    snapshot_cycles = set(map(int, args.snapshot_cycles))
    if args.restart_state is None:
        u = jnp.zeros(ndof, dtype=dtype)
        damage = jnp.zeros(nnode, dtype=dtype)
        history = jnp.zeros(len(tri), dtype=dtype)
        alpha_bar = jnp.zeros(len(tri), dtype=dtype)
        concentration = jnp.zeros(nnode, dtype=dtype)
    else:
        restart_path = args.restart_state.expanduser().resolve()
        with np.load(restart_path) as restart:
            if (
                restart["nodes"].shape != nodes.shape
                or not np.allclose(
                    restart["nodes"], nodes, rtol=0.0, atol=1.0e-12
                )
                or restart["triangles"].shape != tri.shape
                or not np.array_equal(restart["triangles"], tri)
            ):
                raise ValueError("restart state does not match requested mesh")
            u = jnp.asarray(restart["displacement"], dtype=dtype).reshape(-1)
            damage = jnp.asarray(restart["damage"], dtype=dtype)
            history = jnp.asarray(restart["history"], dtype=dtype)
            alpha_bar = jnp.asarray(restart["alpha_bar"], dtype=dtype)
            concentration = jnp.asarray(
                restart["concentration"], dtype=dtype
            )

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
        "stagger_converged",
        "stagger_residual",
        "u_residual",
        "u_linear_residual",
        "u_newton_iterations",
        "d_residual",
        "c_residual",
        "block_wall_s",
        "cumulative_wall_s",
    )
    rows: list[dict[str, float | int]] = []
    start = time.perf_counter()
    sigma_h = jnp.zeros(len(tri), dtype=dtype)
    crack_area = jnp.asarray(0.0, dtype=dtype)
    initial_damage = float(jax.device_get(jnp.max(damage)))
    initial_hydrogen = float(jax.device_get(jnp.max(concentration)))

    print(
        "Yang 2026 JMRT pipeline — full-GPU P1/JAX/cuDSS\n"
        f"JAX {jax.__version__}; Python {platform.python_version()}; "
        f"device={devices[0]}; mesh={nnode} nodes/{len(tri)} triangles",
        flush=True,
    )
    try:
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

                u, ur, ulr, ui = solve_u(damage, u)
                psi, psi_eff, _, sigma_h = qois(u, damage)
                alpha_trial = alpha_start + args.delta_n * psi_eff
                fatigue_trial = fatigue_factor(alpha_trial)
                concentration_trial, cr = solve_c(
                    concentration_start, sigma_h, damage
                )
                hydrogen_trial = hydrogen_factor(concentration_trial)
                factor = fatigue_trial * hydrogen_trial
                history_trial = jnp.maximum(history_start, psi)
                stagger_residual = float("inf")
                converged = False

                for stagger in range(1, args.max_stagger + 1):
                    old_damage = damage
                    old_alpha = alpha_trial
                    old_concentration = concentration_trial
                    old_history = history_trial
                    damage_trial, dr = solve_d(
                        history_trial, factor, damage_start
                    )
                    damage_solved = jnp.clip(
                        jnp.maximum(damage_start, damage_trial), 0.0, 1.0
                    )
                    damage = old_damage + args.stagger_relaxation * (
                        damage_solved - old_damage
                    )
                    damage = damage.at[groups["base"]].set(0.0)
                    u, ur, ulr, ui = solve_u(damage, u)
                    psi, psi_eff, crack_area, sigma_h = qois(u, damage)
                    alpha_solved = alpha_start + args.delta_n * psi_eff
                    alpha_trial = old_alpha + args.stagger_relaxation * (
                        alpha_solved - old_alpha
                    )
                    fatigue_trial = fatigue_factor(alpha_trial)
                    concentration_solved, cr = solve_c(
                        concentration_start, sigma_h, damage
                    )
                    concentration_trial = (
                        old_concentration
                        + args.stagger_relaxation
                        * (concentration_solved - old_concentration)
                    )
                    hydrogen_trial = hydrogen_factor(concentration_trial)
                    factor = fatigue_trial * hydrogen_trial
                    history_solved = jnp.maximum(history_start, psi)
                    history_trial = old_history + args.stagger_relaxation * (
                        history_solved - old_history
                    )
                    residuals = jnp.stack(
                        (
                            jnp.linalg.norm(damage - old_damage)
                            / jnp.maximum(jnp.linalg.norm(damage), 1.0e-30),
                            jnp.linalg.norm(alpha_trial - old_alpha)
                            / jnp.maximum(
                                jnp.linalg.norm(alpha_trial), 1.0e-30
                            ),
                            jnp.linalg.norm(
                                concentration_trial - old_concentration
                            )
                            / jnp.maximum(
                                jnp.linalg.norm(concentration_trial), 1.0e-30
                            ),
                            jnp.linalg.norm(history_trial - old_history)
                            / jnp.maximum(
                                jnp.linalg.norm(history_trial), 1.0e-30
                            ),
                        )
                    )
                    stagger_residual = float(
                        jax.device_get(jnp.max(residuals))
                    )
                    if stagger_residual < args.stagger_tol:
                        converged = True
                        break

                if not converged and not args.accept_unconverged:
                    raise RuntimeError(
                        f"stagger solve failed at block {block}: "
                        f"{stagger_residual:.3e} >= {args.stagger_tol:.3e} "
                        f"after {args.max_stagger} iterations"
                    )
                alpha_bar = alpha_trial
                concentration = concentration_trial
                history = history_trial
                urf, ulrf, drf, crf = (
                    float(jax.device_get(value))
                    for value in (ur, ulr, dr, cr)
                )
                if (
                    not all(
                        np.isfinite(value)
                        for value in (stagger_residual, urf, ulrf, drf, crf)
                    )
                    or max(urf, ulrf, crf) > args.linear_residual_tol
                    or drf > args.phase_field_residual_tol
                ):
                    raise RuntimeError(
                        f"residual failure at block {block}: "
                        f"stagger={stagger_residual:.3e}, u={urf:.3e}, "
                        f"u_linear={ulrf:.3e}, d={drf:.3e}, c={crf:.3e}"
                    )

                damage_host = np.asarray(jax.device_get(damage))
                alpha_host = np.asarray(jax.device_get(alpha_bar))
                fatigue_host = np.asarray(jax.device_get(fatigue_trial))
                hydrogen_host = np.asarray(jax.device_get(hydrogen_trial))
                concentration_host = np.asarray(
                    jax.device_get(concentration)
                )
                sigma_host = np.asarray(jax.device_get(sigma_h))
                cumulative = time.perf_counter() - start
                cycle = args.start_cycle + block * args.delta_n
                row = {
                    "block": block,
                    "cycle": cycle,
                    "max_damage": float(damage_host.max()),
                    "crack_area_mm2": float(jax.device_get(crack_area)),
                    "penetration_mm": penetration_mm(
                        nodes, damage_host, outer_nodes
                    ),
                    "max_alpha_bar_mpa": float(alpha_host.max()),
                    "min_fatigue_factor": float(fatigue_host.min()),
                    "min_hydrogen_factor": float(hydrogen_host.min()),
                    "max_concentration_mol_mm3": float(
                        concentration_host.max()
                    ),
                    "min_concentration_mol_mm3": float(
                        concentration_host.min()
                    ),
                    "max_hydrostatic_stress_mpa": float(sigma_host.max()),
                    "min_hydrostatic_stress_mpa": float(sigma_host.min()),
                    "stagger_iterations": stagger,
                    "stagger_converged": int(converged),
                    "stagger_residual": stagger_residual,
                    "u_residual": urf,
                    "u_linear_residual": ulrf,
                    "u_newton_iterations": ui,
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
                    save_npz(
                        snapshots / f"cycle{cycle:07d}.npz",
                        nodes=nodes,
                        tri=tri,
                        u=u,
                        damage=damage,
                        history=history,
                        alpha_bar=alpha_bar,
                        concentration=concentration,
                        sigma_h=sigma_h,
                        eta_haz=fields["eta_paper"],
                        cycle=cycle,
                    )
                if (
                    args.checkpoint_every
                    and block % args.checkpoint_every == 0
                ):
                    save_npz(
                        outdir / "restart_checkpoint.npz",
                        nodes=nodes,
                        tri=tri,
                        u=u,
                        damage=damage,
                        history=history,
                        alpha_bar=alpha_bar,
                        concentration=concentration,
                        sigma_h=sigma_h,
                        eta_haz=fields["eta_paper"],
                        cycle=cycle,
                    )
                eta = cumulative / block * (args.blocks - block)
                print(
                    f"block={block:3d}/{args.blocks} cycle={cycle:7d} "
                    f"maxd={row['max_damage']:.5f} "
                    f"pen={row['penetration_mm']:.3f}mm "
                    f"stagger={stagger}:{stagger_residual:.2e} "
                    f"u/d/c={urf:.2e}/{drf:.2e}/{crf:.2e} "
                    f"wall={row['block_wall_s']:.2f}s ETA={eta:.1f}s",
                    flush=True,
                )
                if (
                    args.stop_at_max_d > 0.0
                    and row["max_damage"] >= args.stop_at_max_d
                ):
                    break

        final_cycle = args.start_cycle + len(rows) * args.delta_n
        if args.save_state:
            save_npz(
                outdir / "state.npz",
                nodes=nodes,
                tri=tri,
                u=u,
                damage=damage,
                history=history,
                alpha_bar=alpha_bar,
                concentration=concentration,
                sigma_h=sigma_h,
                eta_haz=fields["eta_paper"],
                cycle=final_cycle,
            )
        summary = {
            "version": VERSION,
            "backend": "full GPU",
            "architecture": (
                "FP64 JAX element assembly; CuPy DLPack; fixed-topology "
                "cuDSS mechanics, phase, and nonsymmetric concentration plans"
            ),
            "device": str(devices[0]),
            "mesh": mesh_summary,
            "pressure_mpa": args.pressure_mpa,
            "environment_wppm": c_wppm,
            "blocks_completed": len(rows),
            "cycles_completed": final_cycle,
            "time_mode": args.time_mode,
            "diffusion_dt_s": diffusion_dt,
            "diffusivity_acceleration": acceleration,
            "checkpoint_every_blocks": args.checkpoint_every,
            "stagger_relaxation": args.stagger_relaxation,
            "accept_unconverged": args.accept_unconverged,
            "unconverged_blocks": sum(
                not bool(row["stagger_converged"]) for row in rows
            ),
            "fatigue_multiplier": None,
            "irreversibility": (
                "history field plus post-solve projection "
                "d=max(d_step_start,d_trial); BM nodes always d=0"
            ),
            "physics": {
                "plane_strain": True,
                "E_mpa": E,
                "nu": NU,
                "split": "Felino stress-eigenvalue tensile projection",
                "phase_length_mm": L0,
                "load_ratio": LOAD_RATIO,
                "temperature_k": TEMPERATURE,
                "hydrogen_law": "JMRT pipeline",
                "fatigue_energy": "released mean-load formula",
                "pressure": "consistent vector internal-edge force",
            },
            "initial_damage": initial_damage,
            "initial_hydrogen": initial_hydrogen,
            "final": rows[-1],
            "wall_s": time.perf_counter() - start,
            "cudss": [solver.stats() for solver in direct_solvers],
            "output_files": (
                ["results.csv", "summary.json"]
                + (["state.npz"] if args.save_state else [])
            ),
        }
        (outdir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        print(f"completed: {outdir}", flush=True)
    finally:
        for direct_solver in direct_solvers:
            direct_solver.close()


if __name__ == "__main__":
    main()
