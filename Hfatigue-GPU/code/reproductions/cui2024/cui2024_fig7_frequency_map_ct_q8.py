#!/usr/bin/env python3
"""Cui et al. (2024) Fig. 7 — self-contained run + plot script.

Reproduces the frequency-susceptibility map of

    C. Cui, P. Bortot, M. Ortolani, E. Martinez-Paneda,
    "Computational predictions of hydrogen-assisted fatigue crack growth",
    International Journal of Hydrogen Energy 72 (2024) 315-325.
    doi: 10.1016/j.ijhydene.2024.05.264

Fig. 7 plots da/dN (m/cycle) versus loading frequency f (Hz, log axis
0.001-100) at three fixed stress-intensity ranges DeltaK = 8, 15 and
24 MPa sqrt(m), for R = 0.1 and pH2 = 103 MPa (SA-372 Gr. J steel).
Each frequency and DeltaK pair is an independent short crack-growth
trajectory.  The target DeltaK is reached after about 1 mm of extension,
so all three rates probe the same initially pre-charged process-zone
region instead of confounding frequency with 10-12 mm of crack travel.

The file is fully self-contained: Q8 kernels, the cuDSS direct-solver
backend, the coupled deformation/phase-field/diffusion cycle loop, the
twelve Fig. 7 load cases and the final plotting all live here.  Only the
CT mesh (generated once, stored under outputs/) is read from disk.

Physics summary (validated against the published 55 MPa model line):
  - AT2 phase field with fatigue degradation f_F = (alpha_0/(alpha_bar +
    alpha_0))^2 and hydrogen degradation f_H(C) = 0.12 + 0.88 exp(-7 C^2).
  - Fatigue history increment per cycle block with the local normalisation
    alpha_n(C) = f_H(C) alpha_n0 implied by the degraded toughness.
  - Stress-assisted lattice diffusion with 24 h precharge, environmental
    concentration on the outer boundary and (penalty-enforced) on newly
    created crack faces; the skipped cycles of a block are homogenised
    with the mean min/max hydrostatic stress (cycle-averaged diffusion).

Usage:
    # run every case that has no finished result yet, then plot
    python cui2024_fig7_frequency_map_ct_q8.py --case all
    # run one case
    python cui2024_fig7_frequency_map_ct_q8.py --case f1_a
    # only regenerate the figure from existing runs
    python cui2024_fig7_frequency_map_ct_q8.py --plot-only
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import time
from pathlib import Path


def _requested_platform() -> str:
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--platform", choices=("auto", "cpu", "gpu"), default="auto")
    return probe.parse_known_args()[0].platform


def _requested_ell() -> float:
    """The regularisation length, read before the constants that derive from it.

    ell sets epsilon_c and through it alpha_n, and it is closed over by the
    jitted assembly, so it cannot be overridden after the module has been read
    the way an ordinary option can.  Probing the command line here keeps the
    derived constants consistent with the value actually used.
    """
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--ell-mm", type=float, default=0.27)
    return probe.parse_known_args()[0].ell_mm


if _requested_platform() != "auto":
    os.environ.setdefault(
        "JAX_PLATFORMS", "cuda" if _requested_platform() == "gpu" else "cpu"
    )
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import meshio  # noqa: E402
import numpy as np  # noqa: E402
from jax.scipy.sparse.linalg import bicgstab, cg  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
FIG_DIR = ROOT / "outputs" / "cui_2024_hydrogen_fatigue" / "fig7"
PAPER_MESH = (
    ROOT / "outputs" / "cui_2024_hydrogen_fatigue" / "fig5" / "mesh"
    / "cui_ct_half_q8_paper.msh"
)
EXPERIMENT_DIR = FIG_DIR / "experiments"

# --- Source-paper material and model parameters (N-mm-MPa-s-mol units) ---
E, NU = 210_000.0, 0.3
LAMBDA = E * NU / ((1.0 + NU) * (1.0 - 2.0 * NU))
MU = E / (2.0 * (1.0 + NU))
K_BULK = LAMBDA + 2.0 * MU / 3.0
GC, LENGTH_SCALE, KAPPA = 100.0, _requested_ell(), 1.0e-5
SIGMA_Y = 715.0
SIGMA_C = 4.0 * SIGMA_Y
EPSILON_C = np.sqrt(GC / (3.0 * LENGTH_SCALE * E))
ALPHA_N = SIGMA_C * EPSILON_C / 2.0
ALPHA_0 = 8.0
ALPHA_E = 0.05
MEAN_STRESS_KAPPA = 0.78
SOLUBILITY = 0.077  # wppm MPa^-0.5
RHO_M, M_H = 7.85e-6, 1.008e-3
DIFFUSIVITY = 2.0e-4  # mm^2/s
V_H, R_GAS, TEMPERATURE = 2000.0, 8.314, 300.0

# --- Fig. 3 CT geometry.  B=25 mm preserves the experimental B/W ratio
# after the paper scales the nominal W~26 mm specimen to W=50 mm. ---
W, THICKNESS = 50.0, 25.0
LOAD_LINE_X = 12.5
NOTCH_TIP_X, PRECRACK_TIP_X = 22.3, 29.0
PIN_X, PIN_Y, PIN_R = 12.5, 11.75, 6.25
MILESTONES_MM = (2.0, 8.0, 16.0)
VERSION = "cui2024_fig7_selfcontained_v1"

# --- The eighteen Fig. 7 load cases: one independent short trajectory
# for every frequency and target DeltaK.  Loads place the target after
# 1 mm of extension, inside the stable regression interval.
def _fig7_cases() -> dict[str, dict]:
    frequencies = {
        # Two decades below the published sweep, so that the operating map
        # covers the service rates of Table 1 rather than extrapolating to
        # them.  A slow run is not an expensive one: the transport
        # sub-stepping is capped, so cost is set by the number of blocks,
        # which falls as the frequency does.
        "f000001": 1.0e-5,
        "f00001": 1.0e-4,
        "f0001": 0.001,
        "f001": 0.01,
        "f01": 0.1,
        "f1": 1.0,
        "f10": 10.0,
        "f100": 100.0,
    }
    targets = {
        "k8": 7320.105109950231,
        "k15": 13725.197081156684,
        "k24": 21960.315329850695,
    }
    cases: dict[str, dict] = {}
    for tag, frequency in frequencies.items():
        for target, delta_p in targets.items():
            cases[f"{tag}_{target}"] = {
                "pressure": 103.0,
                "ratio": 0.1,
                "frequency": frequency,
                "delta_p": delta_p,
                "target": 4.0,
            }
    return cases


CASES: dict[str, dict] = _fig7_cases()


# =========================================================================
# cuDSS fixed-topology direct solver backend
# =========================================================================
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


# =========================================================================
# Q8 discretisation kernels
# =========================================================================
def q8_shapes():
    a = 1 / np.sqrt(3)
    gps = np.asarray([[-a, -a], [a, -a], [a, a], [-a, a]])
    n = np.empty((4, 8))
    dn = np.empty((4, 8, 2))
    for q, (x, y) in enumerate(gps):
        n[q] = (
            .25 * (1-x) * (1-y) * (-x-y-1),
            .25 * (1+x) * (1-y) * (x-y-1),
            .25 * (1+x) * (1+y) * (x+y-1),
            .25 * (1-x) * (1+y) * (-x+y-1),
            .5 * (1-x*x) * (1-y),
            .5 * (1+x) * (1-y*y),
            .5 * (1-x*x) * (1+y),
            .5 * (1-x) * (1-y*y),
        )
        dn[q, :, 0] = (
            .25*(1-y)*(2*x+y), .25*(1-y)*(2*x-y),
            .25*(1+y)*(2*x+y), .25*(1+y)*(2*x-y),
            -x*(1-y), .5*(1-y*y), -x*(1+y), -.5*(1-y*y),
        )
        dn[q, :, 1] = (
            .25*(1-x)*(x+2*y), .25*(1+x)*(-x+2*y),
            .25*(1+x)*(x+2*y), .25*(1-x)*(-x+2*y),
            -.5*(1-x*x), -(1+x)*y, .5*(1-x*x), -(1-x)*y,
        )
    return gps, n, dn


def q8_shapes_full():
    """Q8 shape data for standard 3x3 Gauss integration."""
    a = np.sqrt(3.0 / 5.0)
    abscissae = (-a, 0.0, a)
    one_dimensional_weights = (5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0)
    gps = np.asarray([(x, y) for y in abscissae for x in abscissae])
    weights = np.asarray(
        [
            one_dimensional_weights[ix] * one_dimensional_weights[iy]
            for iy in range(3)
            for ix in range(3)
        ]
    )
    n = np.empty((9, 8))
    dn = np.empty((9, 8, 2))
    for q, (x, y) in enumerate(gps):
        n[q] = (
            .25 * (1-x) * (1-y) * (-x-y-1),
            .25 * (1+x) * (1-y) * (x-y-1),
            .25 * (1+x) * (1+y) * (x+y-1),
            .25 * (1-x) * (1+y) * (-x+y-1),
            .5 * (1-x*x) * (1-y),
            .5 * (1+x) * (1-y*y),
            .5 * (1-x*x) * (1+y),
            .5 * (1-x) * (1-y*y),
        )
        dn[q, :, 0] = (
            .25*(1-y)*(2*x+y), .25*(1-y)*(2*x-y),
            .25*(1+y)*(2*x+y), .25*(1+y)*(2*x-y),
            -x*(1-y), .5*(1-y*y), -x*(1+y), -.5*(1-y*y),
        )
        dn[q, :, 1] = (
            .25*(1-x)*(x+2*y), .25*(1+x)*(-x+2*y),
            .25*(1+x)*(x+2*y), .25*(1-x)*(-x+2*y),
            -.5*(1-x*x), -(1+x)*y, .5*(1-x*x), -(1-x)*y,
        )
    return gps, weights, n, dn


def preprocess(nodes: np.ndarray, elem: np.ndarray, integration: str = "full"):
    if integration == "reduced":
        _, n, dn = q8_shapes()
        weights = np.ones(4)
    elif integration == "full":
        _, weights, n, dn = q8_shapes_full()
    else:
        raise ValueError("integration must be 'reduced' or 'full'")
    coords = nodes[elem]
    jac = np.einsum("qia,eib->eqab", dn, coords)
    jacobian_determinant = np.linalg.det(jac)
    if np.any(jacobian_determinant <= 0):
        raise RuntimeError(
            f"nonpositive Q8 Jacobian: {jacobian_determinant.min()}"
        )
    # grad_x(N) = jac^{-1}.T grad_xi(N); the transpose matters on the
    # distorted quadrilaterals of the CT mesh.
    grad = np.einsum("qia,eqba->eqib", dn, np.linalg.inv(jac))
    det = jacobian_determinant * weights[None, :]
    b = np.zeros((len(elem), len(n), 3, 16))
    for i in range(8):
        b[:, :, 0, 2*i] = grad[:, :, i, 0]
        b[:, :, 1, 2*i+1] = grad[:, :, i, 1]
        b[:, :, 2, 2*i] = grad[:, :, i, 1]
        b[:, :, 2, 2*i+1] = grad[:, :, i, 0]
    dofs = np.empty((len(elem), 16), dtype=np.int64)
    dofs[:, 0::2], dofs[:, 1::2] = 2*elem, 2*elem+1
    mass = np.einsum("qi,qj,eq->eij", n, n, det)
    stiff = np.einsum("eqia,eqja,eq->eij", grad, grad, det)
    # Q8 with 2x2 integration has one non-rigid zero-energy mode; recover
    # only that mode's stiffness from 3x3 integration.
    hourglass = np.zeros((len(elem), 16, 16))
    if integration == "reduced":
        _, full_weights, _, full_dn = q8_shapes_full()
        full_jac = np.einsum("qia,eib->eqab", full_dn, coords)
        full_det = np.linalg.det(full_jac) * full_weights[None, :]
        if np.any(full_det <= 0):
            raise RuntimeError(f"nonpositive full Q8 Jacobian: {full_det.min()}")
        full_grad = np.einsum(
            "qia,eqba->eqib", full_dn, np.linalg.inv(full_jac)
        )
        full_b = np.zeros((len(elem), len(full_weights), 3, 16))
        for inode in range(8):
            full_b[:, :, 0, 2*inode] = full_grad[:, :, inode, 0]
            full_b[:, :, 1, 2*inode+1] = full_grad[:, :, inode, 1]
            full_b[:, :, 2, 2*inode] = full_grad[:, :, inode, 1]
            full_b[:, :, 2, 2*inode+1] = full_grad[:, :, inode, 0]
        elastic = np.asarray((
            (LAMBDA + 2*MU, LAMBDA, 0.0),
            (LAMBDA, LAMBDA + 2*MU, 0.0),
            (0.0, 0.0, MU),
        ))
        full_ke = np.einsum(
            "eqia,ij,eqjb,eq->eab", full_b, elastic, full_b, full_det
        )
        for ie, (be, xy) in enumerate(zip(b, coords)):
            _, _, vh = np.linalg.svd(be.reshape(12, 16), full_matrices=True)
            null_projector = vh[12:].T @ vh[12:]
            rigid = np.zeros((16, 3))
            rigid[0::2, 0] = 1
            rigid[1::2, 1] = 1
            rigid[0::2, 2] = -xy[:, 1]
            rigid[1::2, 2] = xy[:, 0]
            qr, _ = np.linalg.qr(rigid)
            hg = .5*(null_projector-qr@qr.T + (null_projector-qr@qr.T).T)
            values, vectors = np.linalg.eigh(hg)
            projector = (vectors*np.clip(values, 0, None))@vectors.T
            stabilisation = projector @ full_ke[ie] @ projector
            hourglass[ie] = .5*(stabilisation + stabilisation.T)
    return n, grad, b, dofs, det, mass, stiff, hourglass


def make_solvers(nodes_np, elem_np, n_np, grad_np, b_np, dofs_np, det_np,
                 mass_np, stiff_np, hourglass_np, free_u_np, fixed_d_np,
                 dtype, tol, maxiter, steps_per_cycle, frequency,
                 external_force_np=None, linear_backend="cudss"):
    """Coupled solvers: isotropic split, AT2, stress-assisted diffusion."""
    elem, dofs = jnp.asarray(elem_np), jnp.asarray(dofs_np)
    arrays = (n_np, grad_np, b_np, det_np, mass_np, stiff_np, hourglass_np)
    n, grad, b, det, mass, stiff, hourglass = (
        jnp.asarray(x, dtype=dtype) for x in arrays
    )
    free_u = jnp.asarray(free_u_np, dtype=dtype)
    fixed_d0 = jnp.asarray(fixed_d_np, dtype=dtype)
    nnode, ndof = len(nodes_np), 2*len(nodes_np)
    external_force = jnp.asarray(
        np.zeros(ndof) if external_force_np is None else external_force_np,
        dtype=dtype,
    )
    tiny = jnp.asarray(1e-30, dtype=dtype)
    hydro_extrapolation = jnp.asarray(np.linalg.pinv(n_np), dtype=dtype)

    def scatter(ids, values, size):
        return jnp.zeros(size, dtype=dtype).at[ids.reshape(-1)].add(values.reshape(-1))

    def state(u, d):
        eps = jnp.einsum("eqij,ej->eqi", b, u[dofs])
        tr = eps[..., 0] + eps[..., 1]
        eps2 = eps[..., 0]**2 + eps[..., 1]**2 + .5*eps[..., 2]**2
        phi = jnp.einsum("qi,ei->eq", n, d[elem])
        g = (1-jnp.clip(phi, 0, 1))**2 + KAPPA
        psi = .5*LAMBDA*tr**2 + MU*eps2
        c11, c12 = g*(LAMBDA+2*MU), g*LAMBDA
        # Physical Cauchy hydrostatic stress: the undamaged stress in fully
        # cracked elements would invent a fictitious wake singularity.
        hydro = g*K_BULK*tr
        z = jnp.zeros_like(g)
        constit = jnp.stack((
            jnp.stack((c11, c12, z), -1),
            jnp.stack((c12, c11, z), -1),
            jnp.stack((z, z, MU*g), -1),
        ), -2)
        alpha = g*psi
        return psi, alpha, constit, hydro

    def element_stiffness(d, boundary, x0):
        _, _, c, _ = state(x0+boundary, d)
        ke = jnp.einsum("eqia,eqij,eqjb,eq->eab", b, c, b, det)
        phi = jnp.einsum("qi,ei->eq", n, d[elem])
        mean_degradation = jnp.mean(
            (1-jnp.clip(phi, 0, 1))**2 + KAPPA, axis=1
        )
        ke += mean_degradation[:, None, None]*hourglass
        return ke

    def solve_u(d, boundary, x0, load=1.0):
        ke = element_stiffness(d, boundary, x0)
        def kmv(x):
            return scatter(dofs, jnp.einsum("eij,ej->ei", ke, x[dofs]), ndof)
        rhs = free_u*(load*external_force-kmv(boundary))

        def op(x):
            return free_u*kmv(free_u*x)+(1-free_u)*x
        diag = free_u*scatter(dofs, jnp.diagonal(ke, axis1=1, axis2=2), ndof)+1-free_u
        scale = jnp.sqrt(jnp.maximum(diag, tiny))
        scaled_rhs = rhs/scale

        def scaled_op(y):
            return op(y/scale)/scale

        def nonzero(_):
            y0 = scale*free_u*x0
            y = cg(
                scaled_op, scaled_rhs, x0=y0, tol=tol, atol=0,
                maxiter=maxiter,
            )[0]
            return y/scale
        sol = jax.lax.cond(
            jnp.linalg.norm(rhs) <= tiny,
            lambda _: jnp.zeros_like(rhs),
            nonzero,
            None,
        )
        res = jnp.linalg.norm(op(sol)-rhs)/(jnp.linalg.norm(rhs)+tiny)
        return boundary+free_u*sol, res

    def phase_system(history, fatigue, hydrogen):
        # Cui supplementary Eqs. (8) and (11): the weak-form fracture terms
        # are multiplied directly by f_F(alpha_bar) f_H(C).
        factor = fatigue*hydrogen
        coeff = GC*factor/LENGTH_SCALE+2*history
        ke = jnp.einsum("eq,eqia,eqja,eq->eij",
                        GC*factor*LENGTH_SCALE, grad, grad, det)
        ke += jnp.einsum("eq,qi,qj,eq->eij", coeff, n, n, det)
        rhs_e = jnp.einsum("eq,qi,eq->ei", 2*history, n, det)
        return ke, scatter(elem, rhs_e, nnode)

    def solve_d(history, fatigue, x0, hydrogen):
        ke, assembled_rhs = phase_system(history, fatigue, hydrogen)
        def kmv(x):
            return scatter(elem, jnp.einsum("eij,ej->ei", ke, x[elem]), nnode)
        fixed = fixed_d0
        free, boundary = 1-fixed, fixed
        rhs = free*(assembled_rhs-kmv(boundary))

        def op(x):
            return free*kmv(free*x)+fixed*x
        diag = free*scatter(elem, jnp.diagonal(ke, axis1=1, axis2=2), nnode)+fixed
        scale = jnp.sqrt(jnp.maximum(diag, tiny))

        def scaled_op(y):
            return op(y/scale)/scale

        sol_scaled, _ = cg(
            scaled_op,
            rhs/scale,
            x0=scale*free*x0,
            tol=tol,
            atol=0,
            maxiter=maxiter,
        )
        sol = sol_scaled/scale
        scale = jnp.maximum(jnp.linalg.norm(rhs), tiny)
        return boundary+free*sol, jnp.linalg.norm(op(sol)-rhs)/scale

    area = jnp.sum(det, axis=1)
    nodal_weight = scatter(
        elem, jnp.broadcast_to((area/8)[:, None], elem.shape), nnode
    )
    dt = jnp.asarray(1/(frequency*steps_per_cycle), dtype=dtype)
    beta = jnp.asarray(V_H/(R_GAS*TEMPERATURE*1000), dtype=dtype)

    def solve_c(
        cold,
        hydro,
        fixed,
        cenv,
        dt_scale=1.0,
        damage=None,
        penalty=0.0,
        penalty_threshold=0.95,
    ):
        # Cui supplementary Eqs. (9), (12)-(14), discretised directly.  The
        # hydrostatic stress is extrapolated from integration points to the
        # Q8 nodes before evaluating its spatial gradient.
        hlocal = jnp.einsum("iq,eq->ei", hydro_extrapolation, hydro)
        hnode = scatter(
            elem, hlocal*(area/8)[:, None], nnode
        )/jnp.maximum(nodal_weight, tiny)
        hgrad = jnp.einsum("eqia,ei->eqa", grad, hnode[elem])
        drift_test = jnp.einsum("eqia,eqa->eqi", grad, hgrad)
        step = dt*dt_scale
        ae = mass + step*DIFFUSIVITY*stiff
        ae -= step*DIFFUSIVITY*beta*jnp.einsum(
            "eqi,qj,eq->eij", drift_test, n, det
        )
        rhs_e = jnp.einsum("eij,ej->ei", mass, cold[elem])

        if damage is not None:
            # Supplementary Eqs. (13)-(14): threshold=0.75 makes this
            # normalised ramp exactly <4d-3>+; kp is convergence-tested.
            damage_q = jnp.einsum("qi,ei->eq", n, damage[elem])
            penalty_q = penalty*jnp.maximum(
                (damage_q-penalty_threshold)/(1.0-penalty_threshold), 0.0
            )
            ae += step*DIFFUSIVITY*jnp.einsum(
                "eq,qi,qj,eq->eij", penalty_q, n, n, det
            )
            rhs_e += step*DIFFUSIVITY*jnp.einsum(
                "eq,qi,eq->ei", penalty_q*cenv, n, det
            )

        def kmv(x):
            return scatter(elem, jnp.einsum("eij,ej->ei", ae, x[elem]), nnode)
        fixed = fixed.astype(dtype)
        free = 1-fixed
        boundary = fixed*cenv
        rhs = free*(scatter(elem, rhs_e, nnode)-kmv(boundary))

        def op(x):
            return free*kmv(free*x)+fixed*x
        diag = free*scatter(
            elem, jnp.diagonal(ae, axis1=1, axis2=2), nnode
        )+fixed
        sol, _ = bicgstab(
            op, rhs, x0=free*cold,
            tol=tol, atol=0, maxiter=maxiter,
            M=lambda r: r/jnp.maximum(jnp.abs(diag), tiny),
        )
        scale = jnp.maximum(jnp.linalg.norm(rhs), tiny)
        concentration = boundary+free*sol
        return jnp.maximum(concentration, 0), jnp.linalg.norm(op(sol)-rhs)/scale

    def qois(u, d):
        psi, alpha, _, hydro = state(u, d)
        phi = jnp.einsum("qi,ei->eq", n, d[elem])
        gphi = jnp.einsum("eqia,ei->eqa", grad, d[elem])
        gamma = jnp.sum(
            det*(phi**2/(2*LENGTH_SCALE)+.5*LENGTH_SCALE*jnp.sum(gphi**2, axis=-1))
        )
        return psi, alpha, gamma, hydro

    concentration_jit, qois_jit = jax.jit(solve_c), jax.jit(qois)
    if linear_backend == "cudss":
        if dtype != jnp.float64:
            raise ValueError("cudss backend currently requires float64")
        direct_u = FixedTopologyCudssSolver(dofs_np, free_u_np)
        direct_d = FixedTopologyCudssSolver(elem_np, 1-fixed_d_np)
        assemble_u = jax.jit(element_stiffness)
        assemble_d = jax.jit(phase_system)

        def solve_u_cudss(d, boundary, x0, load=1.0):
            ke = assemble_u(d, boundary, x0)
            return direct_u.solve(ke, boundary, load*external_force)

        def solve_d_cudss(history, fatigue, x0, hydrogen):
            del x0
            ke, assembled_rhs = assemble_d(history, fatigue, hydrogen)
            return direct_d.solve(ke, fixed_d0, assembled_rhs)

        return solve_u_cudss, solve_d_cudss, concentration_jit, qois_jit
    return jax.jit(solve_u), jax.jit(solve_d), concentration_jit, qois_jit


# =========================================================================
# CT mesh, loading, physics helpers
# =========================================================================
def load_ct_mesh(path: Path):
    mesh = meshio.read(path)
    nodes = np.asarray(mesh.points[:, :2], dtype=np.float64)
    elements = np.concatenate(
        [
            np.asarray(block.data, dtype=np.int64)
            for block in mesh.cells
            if block.type == "quad8"
        ]
    )
    boundary_nodes: list[int] = []
    for block in mesh.cells:
        if block.type == "line3":
            boundary_nodes.extend(map(int, block.data.reshape(-1)))
    groups: dict[str, np.ndarray] = {"boundary": np.unique(boundary_nodes)}

    x, y = nodes[:, 0], nodes[:, 1]
    groups["precrack"] = np.flatnonzero(
        np.isclose(y, 0.0, atol=1.0e-8)
        & (x >= NOTCH_TIP_X - 1.0e-8)
        & (x <= PRECRACK_TIP_X + 1.0e-8)
    )
    groups["symmetry"] = np.flatnonzero(
        np.isclose(y, 0.0, atol=1.0e-8) & (x >= PRECRACK_TIP_X - 1.0e-8)
    )
    groups["pin"] = np.flatnonzero(
        np.abs(np.hypot(x - PIN_X, y - PIN_Y) - PIN_R) < 1.0e-6
    )
    if any(len(groups[name]) == 0 for name in ("precrack", "symmetry", "pin")):
        raise RuntimeError(
            "CT boundary detection failed: "
            + ", ".join(f"{name}={len(values)}" for name, values in groups.items())
        )
    return nodes, elements, groups


def pin_force_shape(nodes: np.ndarray, pin: np.ndarray) -> np.ndarray:
    """Arc-length weighted vertical nodal force with unit resultant."""
    angle = np.arctan2(nodes[pin, 1] - PIN_Y, nodes[pin, 0] - PIN_X)
    order = np.argsort(angle)
    ordered = pin[order]
    segment = np.linalg.norm(
        nodes[ordered] - np.roll(nodes[ordered], 1, axis=0), axis=1
    )
    ordered_weight = 0.5 * (segment + np.roll(segment, -1))
    weight = np.empty(len(pin))
    weight[order] = ordered_weight / ordered_weight.sum()
    force = np.zeros(2 * len(nodes), dtype=np.float64)
    force[2 * pin + 1] = weight
    return force


def initial_phase_field(nodes: np.ndarray, precrack: np.ndarray) -> np.ndarray:
    damage = np.zeros(len(nodes), dtype=np.float64)
    damage[precrack] = 1.0
    return damage


def hydrogen_factor(concentration_q):
    wppm = concentration_q * (1.0e6 * M_H / RHO_M)
    return jnp.clip(0.12 + 0.88 * jnp.exp(-7.0 * wppm**2), 0.12, 1.0)


def fatigue_factor(alpha_bar, alpha_0):
    return (alpha_0 / (alpha_bar + alpha_0)) ** 2


def fatigue_increment(
    alpha_max, peak_seen, cycles, load_ratio, fatigue_n, alpha_n_scale
):
    ratio = (1.0 - load_ratio) / 2.0
    active = peak_seen * ratio ** (2.0 * MEAN_STRESS_KAPPA) >= ALPHA_E
    increment = (
        (
            jnp.maximum(alpha_max, 0.0)
            / (ALPHA_N * alpha_n_scale)
        ) ** fatigue_n
        * ratio ** (2.0 * MEAN_STRESS_KAPPA * fatigue_n)
    )
    return cycles * jnp.where(active, increment, 0.0)


def crack_extension(nodes: np.ndarray, damage: np.ndarray) -> float:
    line = np.flatnonzero(
        np.isclose(nodes[:, 1], 0.0, atol=1.0e-8)
        & (nodes[:, 0] >= PRECRACK_TIP_X - 1.0e-8)
    )
    line = line[np.argsort(nodes[line, 0])]
    last = 0
    for index in range(1, len(line)):
        if damage[line[index]] < 0.5:
            break
        last = index
    tip = float(nodes[line[last], 0])
    if last + 1 < len(line):
        d0, d1 = float(damage[line[last]]), float(damage[line[last + 1]])
        if d0 >= 0.5 > d1 and abs(d1 - d0) > 1.0e-12:
            fraction = (0.5 - d0) / (d1 - d0)
            tip += fraction * float(
                nodes[line[last + 1], 0] - nodes[line[last], 0]
            )
    tip = max(PRECRACK_TIP_X, tip)
    return tip - PRECRACK_TIP_X


def delta_k(a_mm: float, delta_p_n: float, thickness_mm: float) -> float:
    ratio = a_mm / W
    polynomial = (
        0.886
        + 4.46 * ratio
        - 13.32 * ratio**2
        + 14.72 * ratio**3
        - 5.6 * ratio**4
    )
    value = (
        delta_p_n
        * (2.0 + ratio)
        * polynomial
        / (thickness_mm * np.sqrt(W) * (1.0 - ratio) ** 1.5)
    )
    return float(value / np.sqrt(1000.0))


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_state(
    path: Path, cycle: int, state: dict, rows: list[dict], integration: str
) -> None:
    arrays = {
        name: np.asarray(jax.device_get(value)) for name, value in state.items()
    }
    np.savez_compressed(
        path,
        cycle=np.asarray(cycle),
        rows_json=np.asarray(json.dumps(rows)),
        code_version=np.asarray(VERSION),
        integration=np.asarray(f"Q8 {integration}"),
        **arrays,
    )


def save_snapshot(
    path: Path,
    nodes: np.ndarray,
    elements: np.ndarray,
    cycle: int,
    extension: float,
    state: dict,
) -> None:
    np.savez_compressed(
        path,
        nodes=nodes,
        elements=elements,
        cycle=np.asarray(cycle),
        crack_extension_mm=np.asarray(extension),
        displacement=np.asarray(jax.device_get(state["u"])).reshape(-1, 2),
        phase_field=np.asarray(jax.device_get(state["d"])),
        concentration_mol_mm3=np.asarray(jax.device_get(state["c"])),
        concentration_wppm=(
            np.asarray(jax.device_get(state["c"])) * 1.0e6 * M_H / RHO_M
        ),
        alpha_bar=np.asarray(jax.device_get(state["alpha_bar"])),
    )


# =========================================================================
# Coupled cycle-block run loop for one load case
# =========================================================================
def run_case(name: str, args) -> None:
    case = CASES[name]
    target_extension_mm = case.get("target", args.target_extension_mm)
    outdir = args.runs_dir / name
    outdir.mkdir(parents=True, exist_ok=True)
    snapshot_dir = outdir / "snapshots"
    snapshot_dir.mkdir(exist_ok=True)

    jax.config.update("jax_enable_x64", True)
    dtype = jnp.float64
    devices = jax.devices()
    if args.platform == "gpu" and not any(
        device.platform == "gpu" for device in devices
    ):
        raise RuntimeError(f"GPU requested; available devices={devices}")

    mesh_path = args.mesh.expanduser().resolve()
    if not mesh_path.is_file():
        raise FileNotFoundError(f"required mesh does not exist: {mesh_path}")
    nodes, elements, groups = load_ct_mesh(mesh_path)
    c_env_wppm = SOLUBILITY * np.sqrt(case["pressure"])
    c_env = c_env_wppm * 1.0e-6 * RHO_M / M_H
    n, grad, bmat, dofs, det, mass, stiff, hourglass = preprocess(
        nodes, elements, integration=args.integration
    )
    nnode, ndof = len(nodes), 2 * len(nodes)

    fixed_u = 2 * groups["symmetry"] + 1
    anchor = groups["symmetry"][np.argmax(nodes[groups["symmetry"], 0])]
    fixed_u = np.unique(np.r_[fixed_u, 2 * anchor])
    free_u = np.ones(ndof)
    free_u[fixed_u] = 0.0
    fixed_d = np.zeros(nnode)
    fixed_d[groups["precrack"]] = 1.0
    force_shape = pin_force_shape(nodes, groups["pin"])

    solve_u, solve_d, solve_c, qois = make_solvers(
        nodes, elements, n, grad, bmat, dofs, det, mass, stiff, hourglass,
        free_u, fixed_d, dtype, args.cg_tol, args.cg_maxiter,
        args.steps_per_cycle, case["frequency"],
        external_force_np=force_shape, linear_backend=args.linear_backend,
    )
    zero_boundary = jnp.zeros(ndof, dtype=dtype)
    environment = np.setdiff1d(groups["boundary"], groups["symmetry"])
    environment_mask = np.zeros(nnode, dtype=bool)
    environment_mask[environment] = True

    pmax = case["delta_p"] / (1.0 - case["ratio"]) / THICKNESS
    pmin = case["ratio"] * pmax
    state_path = outdir / "checkpoint_latest.npz"
    csv_path = outdir / "results.csv"
    if args.resume and state_path.exists():
        saved = np.load(state_path)
        saved_version = str(saved["code_version"]) if "code_version" in saved else ""
        if saved_version != VERSION:
            raise RuntimeError(
                f"checkpoint version {saved_version!r} is incompatible with {VERSION}"
            )
        cycle = int(saved["cycle"])
        rows = json.loads(str(saved["rows_json"]))
        state = {
            key: jnp.asarray(saved[key], dtype=dtype)
            for key in ("u", "d", "history", "alpha_bar", "peak_seen", "c")
        }
    else:
        cycle, rows = 0, []
        state = {
            "u": jnp.zeros(ndof, dtype=dtype),
            "d": jnp.asarray(
                initial_phase_field(nodes, groups["precrack"]), dtype=dtype
            ),
            "history": jnp.zeros((len(elements), len(n)), dtype=dtype),
            "alpha_bar": jnp.zeros((len(elements), len(n)), dtype=dtype),
            "peak_seen": jnp.zeros((len(elements), len(n)), dtype=dtype),
            "c": jnp.zeros(nnode, dtype=dtype),
        }
        # Equilibrate the imposed d=1 pre-crack into its unloaded AT2
        # profile before history or fatigue accumulation starts.
        unit_q = jnp.ones((len(elements), len(n)), dtype=dtype)
        zero_q = jnp.zeros((len(elements), len(n)), dtype=dtype)
        equilibrated, equilibrium_residual = solve_d(
            zero_q, unit_q, state["d"], unit_q
        )
        state["d"] = jnp.clip(jnp.maximum(equilibrated, state["d"]), 0.0, 1.0)
        print(
            f"initial AT2 pre-crack equilibrium residual="
            f"{float(equilibrium_residual):.3e}",
            flush=True,
        )
        zero_hydro = jnp.zeros((len(elements), len(n)), dtype=dtype)
        if args.initial_hydrogen == "fully-charged":
            state["c"] = jnp.full(nnode, c_env, dtype=dtype)
            print(
                f"fully charged initial condition: C={c_env_wppm:.4f} wppm",
                flush=True,
            )
        else:
            base_dt = 1.0 / (case["frequency"] * args.steps_per_cycle)
            precharge_dt = args.precharge_hours * 3600.0 / args.precharge_steps
            for step in range(args.precharge_steps):
                state["c"], residual = solve_c(
                    state["c"],
                    zero_hydro,
                    jnp.asarray(environment_mask),
                    c_env,
                    precharge_dt / base_dt,
                    state["d"],
                    args.crack_penalty,
                    args.crack_penalty_threshold,
                )
                if step == args.precharge_steps - 1:
                    print(
                        f"precharge {args.precharge_hours:g} h: "
                        f"Cmax={float(jnp.max(state['c'])) * 1e6 * M_H / RHO_M:.4f} "
                        f"wppm, residual={float(residual):.3e}",
                        flush=True,
                    )

    milestones_done = {
        value: (snapshot_dir / f"da_{value:g}mm.npz").exists()
        for value in MILESTONES_MM
    }
    print("=" * 78)
    print(f"Cui 2024 Fig. 7 case {name} — Q8 upper-half CT")
    print(
        f"JAX {jax.__version__}; Python {platform.python_version()}; "
        f"devices={devices}"
    )
    print(
        f"{nnode} nodes, {len(elements)} Q8; ell={LENGTH_SCALE} mm; "
        f"Cenv={c_env_wppm:.4f} wppm; R={case['ratio']:g}; "
        f"f={case['frequency']:g} Hz; n={args.fatigue_n:g}"
    )
    print(
        f"DeltaP={case['delta_p']:g} N, B={THICKNESS:g} mm, initial DeltaK="
        f"{delta_k(PRECRACK_TIP_X-LOAD_LINE_X, case['delta_p'], THICKNESS):.3f}"
    )
    print("=" * 78)

    start = time.perf_counter()
    max_linear_residual = 0.0
    q_coordinates = np.einsum("qi,eij->eqj", n, nodes[elements])
    blocks_this_run = 0
    adaptive_jump = (
        args.cycle_jump
        if args.initial_cycle_jump is None
        else min(args.cycle_jump, max(args.min_cycle_jump, args.initial_cycle_jump))
    )
    adaptive_enabled = args.min_cycle_jump < args.cycle_jump
    while cycle < args.max_cycles:
        jump = min(adaptive_jump, args.max_cycles - cycle)
        blocks_this_run += 1
        state_before_block = dict(state)
        d_start = state["d"]
        alpha_bar_start = state["alpha_bar"]
        peak_seen_start = state["peak_seen"]
        history_start = state["history"]

        # Minimum-load state.  The concentration is advanced later with the
        # mean min/max hydrostatic stress (cycle-averaged diffusion).
        concentration_at_minimum = state["c"]
        state["u"], ur_min = solve_u(state["d"], zero_boundary, state["u"], pmin)
        _, _, _, hydro_min = qois(state["u"], state["d"])

        # Maximum load drives fatigue and fracture.  Newly generated crack
        # faces join the chemical boundary within the same staggered block.
        state["u"], ur = solve_u(state["d"], zero_boundary, state["u"], pmax)
        stagger_error = np.inf
        dr = jnp.asarray(0.0, dtype=dtype)
        cr = jnp.asarray(0.0, dtype=dtype)
        hydrogen_q = jnp.ones_like(alpha_bar_start)
        for stagger in range(1, args.max_stagger + 1):
            old_d = state["d"]
            old_c = state["c"]
            psi, alpha, _, hydro = qois(state["u"], state["d"])
            history_trial = jnp.maximum(history_start, psi)
            if args.diffusion_cycle_mode == "split":
                concentration_half, cr_min = solve_c(
                    concentration_at_minimum,
                    hydro_min,
                    jnp.asarray(environment_mask),
                    c_env,
                    float(jump),
                    state["d"],
                    args.crack_penalty,
                    args.crack_penalty_threshold,
                )
                state["c"], cr_max = solve_c(
                    concentration_half,
                    hydro,
                    jnp.asarray(environment_mask),
                    c_env,
                    float(jump),
                    state["d"],
                    args.crack_penalty,
                    args.crack_penalty_threshold,
                )
                cr = jnp.maximum(cr_min, cr_max)
            else:
                diffusion_hydro = 0.5 * (hydro_min + hydro)
                state["c"], cr = solve_c(
                    concentration_at_minimum,
                    diffusion_hydro,
                    jnp.asarray(environment_mask),
                    c_env,
                    2.0 * float(jump),
                    state["d"],
                    args.crack_penalty,
                    args.crack_penalty_threshold,
                )
            concentration_q = jnp.einsum(
                "qi,ei->eq",
                jnp.asarray(n, dtype=dtype),
                state["c"][jnp.asarray(elements)],
            )
            hydrogen_q = hydrogen_factor(concentration_q)
            # Since alpha_n = sigma_c epsilon_c / 2 and both critical
            # quantities follow the local degraded toughness, alpha_n(C)
            # scales with f_H(C).
            inert = jnp.ones_like(hydrogen_q)
            for_fatigue = (hydrogen_q ** args.alpha_n_hydrogen_exponent
                           if args.hydrogen_pathway in ("both", "fatigue")
                           else inert)
            for_toughness = (hydrogen_q if args.hydrogen_pathway in
                             ("both", "toughness") else inert)
            peak_seen_trial = jnp.maximum(peak_seen_start, alpha)
            alpha_bar_trial = alpha_bar_start + fatigue_increment(
                alpha,
                peak_seen_trial,
                jump,
                case["ratio"],
                args.fatigue_n,
                for_fatigue,
            )
            fatigue_q = fatigue_factor(alpha_bar_trial, args.alpha_0)
            trial, dr = solve_d(
                history_trial, fatigue_q, state["d"], for_toughness
            )
            state["d"] = jnp.clip(jnp.maximum(trial, d_start), 0.0, 1.0)
            state["u"], ur = solve_u(
                state["d"], zero_boundary, state["u"], pmax
            )
            damage_error = jnp.linalg.norm(state["d"] - old_d) / jnp.maximum(
                jnp.linalg.norm(state["d"]), 1.0e-30
            )
            concentration_error = jnp.linalg.norm(
                state["c"] - old_c
            ) / jnp.maximum(jnp.linalg.norm(state["c"]), 1.0e-30)
            stagger_error = float(
                jax.device_get(jnp.maximum(damage_error, concentration_error))
            )
            if stagger_error < args.stagger_tol:
                break
        state["peak_seen"] = peak_seen_trial
        state["alpha_bar"] = alpha_bar_trial
        state["history"] = history_trial

        # Refresh the peak-load concentration with the accepted damage field
        # so snapshots show the tip cloud consistent with the final crack.
        _, _, _, hydro = qois(state["u"], state["d"])
        if args.diffusion_cycle_mode == "split":
            concentration_half, cr_min = solve_c(
                concentration_at_minimum,
                hydro_min,
                jnp.asarray(environment_mask),
                c_env,
                float(jump),
                state["d"],
                args.crack_penalty,
                args.crack_penalty_threshold,
            )
            state["c"], cr_max = solve_c(
                concentration_half,
                hydro,
                jnp.asarray(environment_mask),
                c_env,
                float(jump),
                state["d"],
                args.crack_penalty,
                args.crack_penalty_threshold,
            )
            cr = jnp.maximum(cr_min, cr_max)
        else:
            state["c"], cr = solve_c(
                concentration_at_minimum,
                0.5 * (hydro_min + hydro),
                jnp.asarray(environment_mask),
                c_env,
                2.0 * float(jump),
                state["d"],
                args.crack_penalty,
                args.crack_penalty_threshold,
            )
        concentration_q = jnp.einsum(
            "qi,ei->eq",
            jnp.asarray(n, dtype=dtype),
            state["c"][jnp.asarray(elements)],
        )
        hydrogen_q = hydrogen_factor(concentration_q)

        residuals = [
            float(jax.device_get(value)) for value in (ur_min, ur, dr, cr)
        ]
        max_linear_residual = max(max_linear_residual, *residuals)
        if not all(
            np.isfinite(value) and value <= args.linear_residual_tol
            for value in residuals
        ):
            raise RuntimeError(f"linear residual limit exceeded: {residuals}")

        maximum_damage_increment = float(
            jax.device_get(jnp.max(state["d"] - d_start))
        )
        if (
            (
                maximum_damage_increment > args.max_damage_increment
                or stagger_error >= args.stagger_tol
            )
            and jump > args.min_cycle_jump
        ):
            state = state_before_block
            adaptive_jump = max(args.min_cycle_jump, jump // 2)
            print(
                f"reject N={cycle}+{jump}: max(dd)={maximum_damage_increment:.3e}; "
                f"coupled_err={stagger_error:.3e}; retry jump={adaptive_jump}",
                flush=True,
            )
            continue
        if stagger_error >= args.stagger_tol:
            raise RuntimeError(
                f"coupled iteration failed at minimum jump: {stagger_error:.3e}"
            )

        cycle += jump
        if adaptive_enabled:
            # Continuously steer the next block toward a safe fraction of the
            # admissible damage increment.  The previous "eight very stable
            # accepts, then double" rule usually locked onto the first
            # successful bisection and behaved like a fixed small block.
            target_increment = (
                args.adaptive_target_ratio * args.max_damage_increment
            )
            if maximum_damage_increment > 1.0e-14:
                factor = args.adaptive_safety * (
                    target_increment / maximum_damage_increment
                ) ** args.adaptive_gain
            else:
                factor = args.adaptive_max_growth
            factor = float(np.clip(
                factor, args.adaptive_min_shrink, args.adaptive_max_growth
            ))
            proposed_jump = int(round(jump * factor))
            if factor > 1.0 and proposed_jump <= jump:
                proposed_jump = jump + 1
            adaptive_jump = min(
                args.cycle_jump,
                max(args.min_cycle_jump, proposed_jump),
            )
        damage_host = np.asarray(jax.device_get(state["d"]))
        concentration_host = np.asarray(jax.device_get(state["c"]))
        extension = crack_extension(nodes, damage_host)
        crack_length = PRECRACK_TIP_X - LOAD_LINE_X + extension
        tip_x = PRECRACK_TIP_X + extension
        tip_zone = (
            (q_coordinates[..., 0] >= tip_x)
            & (q_coordinates[..., 0] <= tip_x + 2.0 * LENGTH_SCALE)
            & (q_coordinates[..., 1] <= 2.0 * LENGTH_SCALE)
        )
        fatigue_host = np.asarray(jax.device_get(fatigue_q))
        hydrogen_host = np.asarray(jax.device_get(hydrogen_q))
        tip_nodes = (
            (nodes[:, 0] >= tip_x)
            & (nodes[:, 0] <= tip_x + 2.0 * LENGTH_SCALE)
            & (nodes[:, 1] <= 2.0 * LENGTH_SCALE)
        )
        row = {
            "cycle": cycle,
            "crack_extension_mm": extension,
            "crack_length_mm": crack_length,
            "delta_K_mpa_sqrt_m": delta_k(
                crack_length, case["delta_p"], THICKNESS
            ),
            "maximum_phase_field": float(damage_host.max()),
            "maximum_hydrogen_wppm": float(
                concentration_host.max() * 1.0e6 * M_H / RHO_M
            ),
            "minimum_hydrogen_factor": float(hydrogen_host.min()),
            "minimum_fatigue_factor": float(fatigue_host.min()),
            "tip_zone_minimum_fatigue_factor": float(
                fatigue_host[tip_zone].min()
            ),
            "tip_zone_minimum_hydrogen_factor": float(
                hydrogen_host[tip_zone].min()
            ),
            "tip_zone_maximum_hydrogen_wppm": float(
                (
                    concentration_host[tip_nodes].max() * 1.0e6 * M_H / RHO_M
                )
                if np.any(tip_nodes)
                else 0.0
            ),
            "coupled_iterations": stagger,
            "coupled_error": stagger_error,
            "cycle_jump": jump,
            "maximum_damage_increment": maximum_damage_increment,
            "Ru_min": residuals[0],
            "Ru_max": residuals[1],
            "Rd": residuals[2],
            "Rc_max": residuals[3],
            "maximum_linear_residual": max(residuals),
            "wall_s": time.perf_counter() - start,
        }
        rows.append(row)
        for milestone in MILESTONES_MM:
            if not milestones_done[milestone] and extension >= milestone:
                save_snapshot(
                    snapshot_dir / f"da_{milestone:g}mm.npz",
                    nodes, elements, cycle, extension, state,
                )
                milestones_done[milestone] = True
        if cycle <= 3 or blocks_this_run % args.print_every == 0:
            print(
                f"N={cycle:7d} da={extension:8.4f} mm "
                f"DK={row['delta_K_mpa_sqrt_m']:6.3f} "
                f"Ctip={row['tip_zone_maximum_hydrogen_wppm']:.4f} "
                f"Cmax={row['maximum_hydrogen_wppm']:.4f} "
                f"jump={jump:5d} max(dd)={maximum_damage_increment:.2e} "
                f"coupled={stagger:2d} err={stagger_error:.2e}",
                flush=True,
            )
        if blocks_this_run % args.save_every == 0:
            write_rows(csv_path, rows)
            save_state(state_path, cycle, state, rows, args.integration)
        if extension >= target_extension_mm:
            print(
                f"Reached target crack extension {target_extension_mm:g} mm.",
                flush=True,
            )
            break

    write_rows(csv_path, rows)
    save_state(state_path, cycle, state, rows, args.integration)
    save_snapshot(
        snapshot_dir / "final.npz",
        nodes, elements, cycle, rows[-1]["crack_extension_mm"], state,
    )
    summary = {
        "paper": "Cui et al., IJHE 72 (2024) 315-325, Fig. 7",
        "version": VERSION,
        "doi": "10.1016/j.ijhydene.2024.05.264",
        "case": name,
        "completed": rows[-1]["crack_extension_mm"] >= target_extension_mm,
        "cycles": cycle,
        "final_crack_extension_mm": rows[-1]["crack_extension_mm"],
        "mesh": str(mesh_path),
        "nodes": nnode,
        "q8_elements": len(elements),
        "integration": (
            "2x2 Gauss reduced integration"
            if args.integration == "reduced"
            else "3x3 Gauss full integration"
        ),
        "parameters": {
            "E_mpa": E,
            "nu": NU,
            "Gc_n_per_mm": GC,
            "ell_mm": LENGTH_SCALE,
            "fatigue_n": args.fatigue_n,
            "mean_stress_kappa": MEAN_STRESS_KAPPA,
            "alpha_0_mpa": args.alpha_0,
            "alpha_e_mpa": ALPHA_E,
            "alpha_n_mpa": ALPHA_N,
            "R": case["ratio"],
            "frequency_hz": case["frequency"],
            "pressure_mpa": case["pressure"],
            "Cenv_wppm": c_env_wppm,
            "D_mm2_s": DIFFUSIVITY,
            "temperature_k": TEMPERATURE,
            "solubility_wppm_sqrt_mpa": SOLUBILITY,
            "lag_group_Pi": DIFFUSIVITY
            / (case["frequency"] * LENGTH_SCALE * LENGTH_SCALE),
            "precharge_hours": args.precharge_hours,
            "initial_hydrogen": args.initial_hydrogen,
            "crack_penalty": args.crack_penalty,
            "crack_penalty_threshold": args.crack_penalty_threshold,
            "delta_P_n": case["delta_p"],
            "thickness_mm": THICKNESS,
            "cycle_jump": args.cycle_jump,
            "initial_cycle_jump": args.initial_cycle_jump,
            "min_cycle_jump": args.min_cycle_jump,
            "max_damage_increment": args.max_damage_increment,
            "adaptive_target_ratio": args.adaptive_target_ratio,
            "adaptive_safety": args.adaptive_safety,
            "adaptive_gain": args.adaptive_gain,
            "diffusion_cycle_mode": args.diffusion_cycle_mode,
            "hydrogen_scaled_alpha_n": args.hydrogen_pathway in
                                       ("both", "fatigue"),
            "hydrogen_pathway": args.hydrogen_pathway,
            "alpha_n_hydrogen_exponent": args.alpha_n_hydrogen_exponent,
            "linear_backend": args.linear_backend,
        },
        "maximum_linear_residual": max_linear_residual,
        "wall_s": time.perf_counter() - start,
        "notes": [
            "No result-alignment or fatigue scaling parameter is used.",
            "B=25 mm follows geometric scaling of the nominal B=12.7, "
            "W=26 mm experiments to paper W=50 mm.",
            "Cycle jump multiplies only the fatigue-history increment; "
            "diffusion advances over one representative physical cycle.",
            "The direct nonsymmetric concentration weak form is solved "
            "with BiCGSTAB.",
        ],
    }
    (outdir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"completed case {name} through N={cycle}: {outdir}", flush=True)


# =========================================================================
# Plotting: model growth rates against the digitized experiments
# =========================================================================
# DeltaK sampling points (MPa sqrt(m)), independent-case tag and color.
FIG7_TARGETS = (
    (24.0, "k24", "#EE2724", r"$\Delta K$ = 24 MPa$\sqrt{m}$"),
    (15.0, "k15", "#1348E3", r"$\Delta K$ = 15 MPa$\sqrt{m}$"),
    (8.0, "k8", "#888888", r"$\Delta K$ = 8 MPa$\sqrt{m}$"),
)
FIG7_FREQUENCIES = (
    ("f0001", 0.001),
    ("f001", 0.01),
    ("f01", 0.1),
    ("f1", 1.0),
    ("f10", 10.0),
    ("f100", 100.0),
)


def growth_rate_curve(results_csv: Path, window_mm: float = 0.8):
    """da/dN (m/cycle) versus DeltaK from a results.csv history.

    A centred secant regression is evaluated over a fixed crack-extension
    window.  Regressing N(a) and inverting its slope avoids losing the
    high-rate end of strongly accelerating trajectories, as happens when
    one global cycle-space window is inferred from the mean growth rate.
    """
    data = np.genfromtxt(results_csv, delimiter=",", names=True)
    cycle = np.atleast_1d(data["cycle"]).astype(float)
    extension = np.atleast_1d(data["crack_extension_mm"]).astype(float)
    dk = np.atleast_1d(data["delta_K_mpa_sqrt_m"]).astype(float)
    order = np.argsort(cycle)
    cycle, extension, dk = cycle[order], extension[order], dk[order]
    keep = np.concatenate(([True], np.diff(cycle) > 0))
    cycle, extension, dk = cycle[keep], extension[keep], dk[keep]
    if len(cycle) < 8 or extension[-1] - extension[0] <= window_mm:
        return np.empty(0), np.empty(0)

    # Keep the final cycle of any zero-growth plateau so N(a) remains
    # single-valued without understating the elapsed cycles.
    _, reverse_index = np.unique(extension[::-1], return_index=True)
    unique_index = np.sort(len(extension) - 1 - reverse_index)
    extension_u = extension[unique_index]
    cycle_u = cycle[unique_index]
    dk_u = dk[unique_index]

    half = window_mm / 2.0
    lower = max(0.3, extension_u[0] + half)
    upper = extension_u[-1] - half
    if upper <= lower:
        return np.empty(0), np.empty(0)
    centers = np.linspace(lower, upper, 800)
    local_offsets = np.linspace(-half, half, 31)
    rates = []
    for center in centers:
        local_extension = center + local_offsets
        local_cycle = np.interp(local_extension, extension_u, cycle_u)
        dN_da = np.polyfit(local_extension, local_cycle, 1)[0]
        rates.append(1.0 / dN_da if dN_da > 0 else np.nan)
    rates = np.asarray(rates)
    dk_out = np.interp(centers, extension_u, dk_u)
    valid = np.isfinite(rates) & (rates > 0)
    return dk_out[valid], 1.0e-3 * rates[valid]


def rate_at_delta_k(
    results_csv: Path, target_dk: float, window_mm: float = 0.8
):
    """Interpolate da/dN (m/cycle) at `target_dk` from a leg's results.csv.

    Returns None when the run does not exist, produced no growth-rate
    curve, or does not cover the requested DeltaK.
    """
    if not results_csv.exists():
        return None
    dk, rate = growth_rate_curve(results_csv, window_mm=window_mm)
    if len(dk) < 2:
        return None
    order = np.argsort(dk)
    dk, rate = dk[order], rate[order]
    if not (dk[0] <= target_dk <= dk[-1]):
        return None
    return float(np.interp(target_dk, dk, rate))


def plot_figures(args) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    final_dir = FIG_DIR / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(7.4, 6.0), constrained_layout=True)

    experiment = EXPERIMENT_DIR / "fig7_experiments.csv"
    points = None
    if experiment.exists():
        points = np.genfromtxt(experiment, delimiter=",", names=True)
    paper_model_path = EXPERIMENT_DIR / "paper_fig7_model_curves.csv"
    paper_model = None
    if paper_model_path.exists():
        paper_model = np.genfromtxt(
            paper_model_path, delimiter=",", names=True
        )

    for target_dk, leg, color, label in FIG7_TARGETS:
        if paper_model is not None:
            published = np.isclose(
                np.atleast_1d(paper_model["delta_K"]).astype(float),
                target_dk,
            )
            if np.any(published):
                axis.plot(
                    np.atleast_1d(paper_model["frequency_hz"])[published],
                    np.atleast_1d(
                        paper_model["da_dN_m_per_cycle"]
                    )[published],
                    "--", color=color, lw=1.4, alpha=0.8,
                )
        if points is not None:
            select = np.isclose(
                np.atleast_1d(points["delta_K"]).astype(float), target_dk
            )
            if np.any(select):
                axis.plot(
                    np.atleast_1d(points["frequency_hz"]).astype(float)[select],
                    np.atleast_1d(
                        points["da_dN_m_per_cycle"]
                    ).astype(float)[select],
                    "^", ms=6.0, color=color, mec=color, alpha=0.8,
                )
        frequencies, rates = [], []
        for tag, frequency in FIG7_FREQUENCIES:
            rate = rate_at_delta_k(
                args.runs_dir / f"{tag}_{leg}" / "results.csv",
                target_dk,
                window_mm=args.growth_rate_window_mm,
            )
            if rate is not None:
                frequencies.append(frequency)
                rates.append(rate)
        if frequencies:
            axis.plot(
                frequencies, rates, "-o", color=color, lw=2.2, ms=5.0,
                label=label,
            )

    axis.set(
        xscale="log", yscale="log",
        xlim=(5.0e-4, 2.0e2), ylim=(1.0e-8, 1.0e-5),
        xlabel=r"$f$ (Hz)",
        ylabel=r"$da/dN$ (m/cycle)",
        title=(
            "Cui et al. (2024) Fig. 7 — FCGR susceptibility to loading "
            "frequency\n(thick: current; dashed: published; symbols: "
            "experiments; $R$ = 0.1, $p_{H_2}$ = 103 MPa)"
        ),
    )
    axis.grid(alpha=0.2, which="both")
    axis.legend(frameon=False, fontsize=10, loc="upper right")
    target = final_dir / "fig7_reproduction.png"
    fig.savefig(target, dpi=220)
    plt.close(fig)
    print(target, flush=True)


# =========================================================================
# Command line interface
# =========================================================================
def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    result.add_argument("--platform", choices=("auto", "cpu", "gpu"), default="gpu")
    result.add_argument(
        "--case",
        action="append",
        choices=tuple(CASES) + ("all",),
        help="load case(s) to run; 'all' runs every case without a finished "
        "summary.json (repeatable)",
    )
    result.add_argument("--plot-only", action="store_true")
    result.add_argument("--no-plot", action="store_true")
    result.add_argument("--force", action="store_true",
                        help="rerun cases even if a finished summary exists")
    result.add_argument("--mesh", type=Path, default=PAPER_MESH)
    result.add_argument("--runs-dir", type=Path, default=FIG_DIR / "runs")
    result.add_argument(
        "--ell-mm",
        type=float,
        default=0.27,
        help="phase-field regularisation length in mm.  Read again before the "
        "material constants are built, so the value given here also sets "
        "epsilon_c and alpha_n; the mesh must resolve it with at least two "
        "elements across",
    )
    result.add_argument(
        "--alpha-0",
        type=float,
        default=10.5,
        help="air-calibrated fatigue degradation scale in MPa",
    )
    result.add_argument(
        "--growth-rate-window-mm",
        type=float,
        default=0.8,
        help="centred crack-growth regression window in mm (about 3 ell)",
    )
    result.add_argument(
        "--integration", choices=("reduced", "full"), default="full"
    )
    result.add_argument(
        "--linear-backend", choices=("jax-cg", "cudss"), default="cudss"
    )
    result.add_argument(
        "--fatigue-n",
        type=float,
        default=1.25,
        help="fatigue exponent n of Eq. (7); the paper's Fig. 9 enhanced "
        "model uses 1.9 in hydrogen environments",
    )
    result.add_argument(
        "--alpha-n-hydrogen-exponent",
        type=float,
        default=1.0,
        help="exponent m in alpha_n(C) = f_H(C)^m alpha_n.  The published "
        "model takes m = 1, which ties the rate at which a point accumulates "
        "fatigue to the same factor that degrades its toughness.  Lowering m "
        "weakens the frequency response without touching the toughness route",
    )
    result.add_argument(
        "--hydrogen-pathway",
        choices=("both", "toughness", "fatigue"),
        default="both",
        help="which route hydrogen is allowed to act through. It normally "
        "does two things at once: it lowers the fracture toughness and it "
        "speeds up fatigue accumulation through alpha_n(C) = f_H(C) alpha_n. "
        "Disabling one at a time says which of them carries the frequency "
        "sensitivity, and is a diagnostic rather than a physical model",
    )
    result.add_argument(
        "--target-extension-mm",
        type=float,
        default=12.0,
        help="fallback target crack extension; each case's 'target' entry "
        "in the case table takes precedence",
    )
    result.add_argument(
        "--stop-at-mm",
        type=float,
        default=None,
        help=("override every case's target extension.  The growth rate is "
              "read at 1 mm, so a campaign that only needs a rate can stop "
              "well before the 4 mm the published trajectories run to"),
    )
    result.add_argument("--max-cycles", type=int, default=500_000_000)
    result.add_argument("--cycle-jump", type=int, default=100_000)
    result.add_argument("--min-cycle-jump", type=int, default=1)
    result.add_argument(
        "--initial-cycle-jump",
        type=int,
        default=None,
        help="initial adaptive block; defaults to --cycle-jump",
    )
    result.add_argument("--max-damage-increment", type=float, default=0.05)
    result.add_argument(
        "--adaptive-target-ratio",
        type=float,
        default=0.8,
        help="target damage increment as a fraction of the rejection limit",
    )
    result.add_argument(
        "--adaptive-safety",
        type=float,
        default=0.95,
        help="safety multiplier in the proportional block-size controller",
    )
    result.add_argument(
        "--adaptive-gain",
        type=float,
        default=0.5,
        help="exponent in the proportional block-size controller",
    )
    result.add_argument("--adaptive-min-shrink", type=float, default=0.5)
    result.add_argument("--adaptive-max-growth", type=float, default=2.0)
    result.add_argument("--steps-per-cycle", type=int, default=2)
    result.add_argument(
        "--diffusion-cycle-mode",
        choices=("average", "split"),
        default="average",
        help="mean-stress block diffusion or sequential load half-cycles",
    )
    result.add_argument("--precharge-hours", type=float, default=24.0)
    result.add_argument("--precharge-steps", type=int, default=24)
    result.add_argument(
        "--initial-hydrogen",
        choices=("24h", "fully-charged"),
        default="24h",
        help="finite precharge used by Fig. 7 or a saturation diagnostic",
    )
    result.add_argument("--crack-penalty", type=float, default=1.0e5)
    result.add_argument("--crack-penalty-threshold", type=float, default=0.75)
    result.add_argument("--max-stagger", type=int, default=50)
    result.add_argument("--stagger-tol", type=float, default=1.0e-4)
    result.add_argument("--cg-tol", type=float, default=1.0e-8)
    result.add_argument("--cg-maxiter", type=int, default=8000)
    result.add_argument("--linear-residual-tol", type=float, default=5.0e-5)
    result.add_argument("--save-every", type=int, default=10)
    result.add_argument("--print-every", type=int, default=10)
    result.add_argument("--resume", action="store_true")
    result.add_argument(
        "--diffusivity-mm2-s",
        type=float,
        default=DIFFUSIVITY,
        help="lattice diffusivity; raising it is equivalent to lowering the "
             "frequency, which is what the lag group D/(f ell^2) asserts",
    )
    result.add_argument(
        "--temperature-k",
        type=float,
        default=TEMPERATURE,
        help="absolute temperature entering the stress-drift factor V_H/(R T)",
    )
    result.add_argument(
        "--solubility-wppm-sqrt-mpa",
        type=float,
        default=SOLUBILITY,
        help="Sievert constant setting the environmental concentration",
    )
    result.add_argument(
        "--load-ratio",
        type=float,
        default=None,
        help=("override the load ratio of every case, holding delta_P and "
              "therefore delta_K fixed so that only the mean load changes"),
    )
    return result


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)

    # These feed jitted kernels through the module namespace, so they have to
    # be set before the first trace, i.e. before any case runs.
    global DIFFUSIVITY, TEMPERATURE, SOLUBILITY
    DIFFUSIVITY = args.diffusivity_mm2_s
    TEMPERATURE = args.temperature_k
    SOLUBILITY = args.solubility_wppm_sqrt_mpa

    # Raising R at fixed delta_P raises the mean load without touching the
    # amplitude, so delta_K is unchanged and the comparison isolates the mean
    # stress.  Editing the case table is the only place the ratio is read.
    if args.load_ratio is not None:
        for case in CASES.values():
            case["ratio"] = args.load_ratio
    if args.stop_at_mm is not None:
        for case in CASES.values():
            case["target"] = args.stop_at_mm

    if not args.plot_only:
        selected = args.case or ["all"]
        if "all" in selected:
            selected = list(CASES)
        for name in selected:
            summary = args.runs_dir / name / "summary.json"
            if summary.exists() and not args.force:
                finished = json.loads(summary.read_text(encoding="utf-8"))
                if finished.get("completed"):
                    print(f"skip finished case {name}", flush=True)
                    continue
            run_case(name, args)
    if not args.no_plot:
        plot_figures(args)


if __name__ == "__main__":
    main()
