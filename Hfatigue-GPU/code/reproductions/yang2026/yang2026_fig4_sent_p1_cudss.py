#!/usr/bin/env python3
"""Yang et al. (2026) Fig. 4 P1 SENT solver with fixed-CSR NVIDIA cuDSS.

The two production branches implement the paper's energy decompositions:
Miehe strain-spectral Eq. (5) and Amor volumetric-deviatoric Eq. (6).
The inconsistent stress-projection branch from the released Felino deck is
retained only as ``felino_stress`` for diagnostics. The bridge replaces only
unconstrained sparse-direct solves; the fatigue history follows the released
mean-load equation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

def load_runtime() -> None:
    """Import numerical dependencies only after CLI parsing."""
    global jax, jnp, meshio, np, osqp, sp, spla
    if "jax" in globals():
        return
    try:
        import jax as jax_module
        import jax.numpy as jax_numpy
        import meshio as meshio_module
        import numpy as numpy_module
        import osqp as osqp_module
        import scipy.sparse as scipy_sparse
        import scipy.sparse.linalg as scipy_sparse_linalg
    except ImportError as exc:
        raise RuntimeError(
            "Yang cuDSS solver requires JAX, meshio, NumPy, OSQP, and SciPy"
        ) from exc
    jax, jnp, meshio, np, osqp, sp, spla = (
        jax_module, jax_numpy, meshio_module, numpy_module, osqp_module,
        scipy_sparse, scipy_sparse_linalg,
    )

E, NU = 210000.0, 0.3
LAMBDA = E * NU / ((1 + NU) * (1 - 2 * NU))
MU = E / (2 * (1 + NU))
K_BULK = LAMBDA + 2 * MU / 3
GC, L0, ALPHA_T, KAPPA = 2.7, 0.016, 30.0, 1.0e-6
U_MAX, LOAD_RATIO, FATIGUE_N = 0.0005, 0.5, 0.5
RHO_M, M_H, A_M = 7.85e-6, 1.008e-3, 55.845e-3
C_ENV = 1e-6 * RHO_M / M_H
D_L, V_H, R_GAS_NMM, TEMPERATURE = 1.0e-1, 2000.0, 8.314e3, 300.0
HYDROGEN_CHI, DELTA_GB = 0.89, 3.0e4
CRACK_DIFFUSIVITY_FACTOR, PHI_THRESHOLD, PHI_WIDTH = 1.0e5, 0.95, 0.02
DEFAULT_DELTA_N, DEFAULT_BLOCKS = 100, 300
VERSION = "yang2026_sent_cudss_v5_paper_energy_splits"


def root() -> Path:
    """Repository root resolved from code/reproductions/yang2026."""
    return Path(__file__).resolve().parents[3]


def default_mesh() -> Path:
    return (root() / "open_source_code" / "phase-field-hydrogen-fatigue"
            / "phase-field-hydrogen-fatigue-main" / "01_IJF_2026"
            / "examples" / "SENT10.inp")


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mesh", type=Path, default=default_mesh())
    p.add_argument(
        "--split",
        choices=("spectral", "felino_stress", "amor"),
        default="spectral",
        help=(
            "spectral: paper Eq. (5) Miehe strain-spectral split; "
            "felino_stress: released Felino stress projection; "
            "amor: paper Eq. (6) volumetric-deviatoric split"
        ),
    )
    p.add_argument(
        "--figure",
        choices=("fig4", "fig5", "fig6", "fig7", "fig8", "fig10", "fig12",
                 "fig14", "fig15", "fig16", "fig17"),
        default="fig4",
        help="paper figure being reproduced; fig6/7/8/10/12 enable hydrogen",
    )
    p.add_argument("--loading", choices=("displacement", "pin"),
                   default="displacement",
                   help="pin: hold the BOTTOM hole and load the TOP hole")
    p.add_argument("--pin-load-n", type=float, default=0.0,
                   help="maximum pin load for --loading pin [N]")
    p.add_argument("--hydrogen-law",
                   choices=("langmuir", "pressure_vessel"),
                   default="langmuir",
                   help="pressure_vessel: Cui's f_C = 0.12 + 0.88 exp(-7c^2)")
    p.add_argument("--initial-crack-damage", action="store_true",
                   help="impose d = 1 on the Crack set, as the CT case does")
    p.add_argument("--phase-field-residual-tol", type=float, default=1.0e-3,
                   help="acceptable relative residual of the bound-constrained "
                        "phase-field solve, which is looser than the linear "
                        "tolerance because it is a quadratic program")
    p.add_argument("--stop-growth-mm", type=float, default=1.0,
                   help="crack growth beyond which an equilibrium failure "
                        "under pin loading is treated as specimen failure")
    p.add_argument("--precharge-hours", type=float, default=0.0,
                   help="unloaded diffusion time before cycling; the CT "
                        "specimens of Fig. 14 are pre-charged for 24 h")
    p.add_argument("--hydrogen", choices=("auto", "on", "off"), default="auto",
                   help="override the hydrogen coupling implied by --figure")
    p.add_argument("--length-scale", type=float, default=L0,
                   help="phase-field length scale; SENT uses 0.016 mm")
    p.add_argument("--alpha-t", type=float, default=ALPHA_T,
                   help="fatigue threshold alpha_T [MPa]")
    p.add_argument("--umax", type=float, default=U_MAX,
                   help="prescribed displacement amplitude on TOP [mm]")
    p.add_argument("--load-ratio", type=float, default=LOAD_RATIO)
    p.add_argument("--fatigue-n", type=float, default=FATIGUE_N,
                   help="fatigue degradation exponent n")
    p.add_argument("--blocks", type=int, default=DEFAULT_BLOCKS)
    p.add_argument("--delta-n", type=int, default=DEFAULT_DELTA_N,
                   help="fatigue cycles represented by one global solve")
    p.add_argument("--platform", choices=("auto", "cpu", "gpu"), default="auto")
    p.add_argument("--dtype", choices=("float64", "float32"), default="float64")
    p.add_argument("--outdir", type=Path)
    p.add_argument("--max-stagger", type=int, default=12,
                   help="MOOSE main-app fixed_point_max_its")
    p.add_argument("--stagger-tol", type=float, default=1e-6)
    p.add_argument(
        "--stagger-relaxation",
        type=float,
        default=1.0,
        help="fixed-point relaxation; MOOSE's released input uses the default 1",
    )
    p.add_argument("--cg-tol", type=float, default=1e-7)
    p.add_argument("--cg-maxiter", type=int, default=5000)
    p.add_argument("--linear-residual-tol", type=float, default=1e-4)
    p.add_argument("--fatigue-scale", type=float, default=1.0,
                   help="must be 1; retained for old command compatibility")
    p.add_argument(
        "--fatigue-energy",
        choices=("mean_load", "cla"),
        default="mean_load",
        help=(
            "mean_load: paper Eq. (16) surrogate as released; "
            "cla: paper Eq. (11) split tensile energy with the R-correction"
        ),
    )
    p.add_argument(
        "--multiply-by-degradation",
        action="store_true",
        help="apply the g(d) prefactor written in paper Eq. (11)",
    )
    p.add_argument("--precharge-ratio", type=float, default=0.1,
                   help="initial C/C_env for Fig.6; use 0 for no precharging")
    p.add_argument("--diffusion-time-mode", choices=("source", "physical"),
                   default="source",
                   help="source: 1 s/block as released input; physical: deltaN/f")
    p.add_argument("--frequency-hz", type=float, default=1.0)
    p.add_argument("--diffusivity-mm2-s", type=float, default=D_L,
                   help="lattice diffusivity; released input uses 0.1")
    p.add_argument("--diffusion-max-step-s", type=float, default=10.0,
                   help="maximum backward-Euler diffusion substep")
    p.add_argument("--gc", type=float, default=GC, help="Gc [N/mm]")
    p.add_argument("--chi", type=float, default=HYDROGEN_CHI,
                   help="hydrogen damage coefficient zeta")
    p.add_argument("--environment-wppm", type=float, default=1.0,
                   help="hydrogen concentration prescribed on the notch")
    p.add_argument("--delta-gb-j-mol", type=float, default=DELTA_GB,
                   help="hydrogen segregation free energy [J/mol]")
    p.add_argument("--fatigue-delay-cycles", type=int, default=0,
                   help="must be 0; retained for old command compatibility")
    p.add_argument("--fatigue-initial-boost", type=float, default=0.0,
                   help="must be 0; retained for old command compatibility")
    p.add_argument("--fatigue-boost-decay-cycles", type=float, default=2000.0,
                   help="exponential decay scale for the initial fatigue boost")
    p.add_argument("--save-state", action="store_true")
    p.add_argument(
        "--snapshot-cycles",
        type=int,
        nargs="*",
        help="cycle numbers at which to dump the full field state",
    )
    p.add_argument(
        "--snapshot-every-blocks",
        type=int,
        help="also dump the full field state every this many blocks",
    )
    return p


def load_npz_mesh(path: Path):
    """Load a mesh generated for a geometry that the release does not ship."""
    data = np.load(path)
    nodes = np.asarray(data["nodes"][:, :2], dtype=np.float64)
    tri = np.asarray(data["triangles"], dtype=np.int64)
    groups = {
        name: np.asarray(data[f"set_{name}"], dtype=np.int64)
        for name in ("TOP", "BOTTOM", "Crack")
    }
    xy = nodes[tri]
    twice_area = ((xy[:, 1, 0] - xy[:, 0, 0]) * (xy[:, 2, 1] - xy[:, 0, 1])
                  - (xy[:, 2, 0] - xy[:, 0, 0]) * (xy[:, 1, 1] - xy[:, 0, 1]))
    flip = twice_area < 0
    tri[flip] = tri[flip][:, [0, 2, 1]]
    areas = np.abs(twice_area) / 2
    if np.any(areas <= 0):
        raise RuntimeError("non-positive triangle area")
    return nodes, tri, areas, groups


def load_mesh(path: Path):
    if path.suffix == ".npz":
        return load_npz_mesh(path)
    mesh = meshio.read(str(path))
    nodes = np.asarray(mesh.points[:, :2], dtype=np.float64)
    cells = [np.asarray(c.data, dtype=np.int64)
             for c in mesh.cells if c.type == "triangle"]
    if not cells:
        raise RuntimeError("mesh contains no P1 triangles")
    tri = np.concatenate(cells)
    groups = {name: np.asarray(mesh.point_sets[name], dtype=np.int64)
              for name in ("TOP", "BOTTOM", "Crack") if name in mesh.point_sets}
    if set(groups) != {"TOP", "BOTTOM", "Crack"}:
        raise RuntimeError("SENT10.inp must contain TOP, BOTTOM, and Crack node sets")
    if (len(nodes), len(tri)) != (2991, 5768):
        raise RuntimeError(f"expected original 2991/5768 mesh, got {len(nodes)}/{len(tri)}")
    xy = nodes[tri]
    twice_area = ((xy[:, 1, 0] - xy[:, 0, 0]) * (xy[:, 2, 1] - xy[:, 0, 1])
                  - (xy[:, 2, 0] - xy[:, 0, 0]) * (xy[:, 1, 1] - xy[:, 0, 1]))
    flip = twice_area < 0
    tri[flip] = tri[flip][:, [0, 2, 1]]
    areas = np.abs(twice_area) / 2
    if np.any(areas <= 0):
        raise RuntimeError("non-positive triangle area")
    return nodes, tri, areas, groups



def preprocess(nodes, tri, areas):
    x, y = nodes[:, 0], nodes[:, 1]
    i, j, k = tri.T
    gx = np.stack((y[j] - y[k], y[k] - y[i], y[i] - y[j]), axis=1)
    gy = np.stack((x[k] - x[j], x[i] - x[k], x[j] - x[i]), axis=1)
    gx, gy = gx / (2 * areas[:, None]), gy / (2 * areas[:, None])
    bmat = np.zeros((len(tri), 3, 6))
    for a in range(3):
        bmat[:, 0, 2 * a], bmat[:, 1, 2 * a + 1] = gx[:, a], gy[:, a]
        bmat[:, 2, 2 * a], bmat[:, 2, 2 * a + 1] = gy[:, a], gx[:, a]
    dofs = np.empty((len(tri), 6), dtype=np.int64)
    dofs[:, 0::2], dofs[:, 1::2] = 2 * tri, 2 * tri + 1
    mass = areas[:, None, None] * np.array(((2, 1, 1), (1, 2, 1), (1, 1, 2))) / 12
    grad = areas[:, None, None] * (
        gx[:, :, None] * gx[:, None, :] + gy[:, :, None] * gy[:, None, :])
    return bmat, dofs, mass, grad


def make_solvers(nodes_np, tri_np, areas_np, b_np, dofs_np, mass_np, grad_np,
                 free_u_np, split, dtype, tol, maxiter, diffusivity_lattice,
                 gc_value, chi_value, c_environment, delta_gb,
                 external_force_np=None, hydrogen_law="langmuir",
                 fatigue_energy="mean_load", multiply_by_degradation=False,
                 alpha_t_cell=None, fatigue_n_cell=None,
                 damage_zero_ids=None):
    tri, dofs = jnp.asarray(tri_np), jnp.asarray(dofs_np)
    areas, bmat, mass, grad = (jnp.asarray(a, dtype=dtype)
                               for a in (areas_np, b_np, mass_np, grad_np))
    cell_count = len(tri_np)
    gc_cell_np = np.broadcast_to(
        np.asarray(gc_value, dtype=np.float64), (cell_count,)
    ).copy()
    diffusivity_cell_np = np.broadcast_to(
        np.asarray(diffusivity_lattice, dtype=np.float64), (cell_count,)
    ).copy()
    alpha_t = jnp.asarray(
        np.broadcast_to(
            ALPHA_T if alpha_t_cell is None else np.asarray(alpha_t_cell),
            (cell_count,),
        ),
        dtype=dtype,
    )
    fatigue_n = jnp.asarray(
        np.broadcast_to(
            FATIGUE_N if fatigue_n_cell is None else np.asarray(fatigue_n_cell),
            (cell_count,),
        ),
        dtype=dtype,
    )
    free_u = jnp.asarray(free_u_np, dtype=dtype)
    nnode, ndof = len(nodes_np), 2 * len(nodes_np)
    damage_zero_ids_np = np.asarray(
        [] if damage_zero_ids is None else damage_zero_ids, dtype=np.int64
    )
    damage_upper_np = np.ones(nnode, dtype=np.float64)
    damage_upper_np[damage_zero_ids_np] = 0.0
    tiny = jnp.asarray(1e-30, dtype=dtype)
    # libMesh TRI3 uses the three-point rule for the quadratic degradation
    # integrand. Displacement strain is constant, while d and g(d) vary at QPs.
    qp_shape = jnp.asarray(((2 / 3, 1 / 6, 1 / 6),
                            (1 / 6, 2 / 3, 1 / 6),
                            (1 / 6, 1 / 6, 2 / 3)), dtype=dtype)
    eye2 = jnp.eye(2, dtype=dtype)

    def scatter(ids, values, size):
        return jnp.zeros(size, dtype=dtype).at[ids.reshape(-1)].add(values.reshape(-1))

    def pcg(op, rhs, initial, precondition):
        r0 = rhs - op(initial)
        z0 = precondition(r0)
        state = (jnp.asarray(0, jnp.int32), initial, r0, z0, jnp.vdot(r0, z0))
        target = tol * jnp.maximum(jnp.linalg.norm(rhs), tiny)

        def cond(s):
            return (s[0] < maxiter) & (jnp.linalg.norm(s[2]) > target)

        def body(s):
            it, x, residual, direction, rz = s
            ad = op(direction)
            step = rz / jnp.where(jnp.abs(jnp.vdot(direction, ad)) > tiny,
                                  jnp.vdot(direction, ad), tiny)
            x, residual = x + step * direction, residual - step * ad
            z = precondition(residual)
            rz_new = jnp.vdot(residual, z)
            beta = rz_new / jnp.where(jnp.abs(rz) > tiny, rz, tiny)
            return it + 1, x, residual, z + beta * direction, rz_new

        final = jax.lax.while_loop(cond, body, state)
        return final[0], final[1]

    def strain_state(u):
        strain = jnp.einsum("eij,ej->ei", bmat, u[dofs])
        trace = strain[:, 0] + strain[:, 1]
        mean = trace / 2
        radius = jnp.sqrt(((strain[:, 0] - strain[:, 1]) / 2) ** 2
                          + (strain[:, 2] / 2) ** 2)
        p1, p2 = mean + radius, mean - radius
        eps_max = jnp.maximum(jnp.maximum(p1, p2), 0)
        return strain, trace, eps_max

    def smooth_positive(x):
        # Differentiable machine-scale regularisation of Macaulay brackets.
        eps = jnp.asarray(1e-10, dtype=dtype)
        return 0.5 * (x + jnp.sqrt(x * x + eps * eps))

    def split_stress_energy(strain, trace):
        exx, eyy, gamma = strain[:, 0], strain[:, 1], strain[:, 2]
        sxx = 2 * MU * exx + LAMBDA * trace
        syy = 2 * MU * eyy + LAMBDA * trace
        sxy = MU * gamma
        if split == "felino_stress":
            # Felino ADComputePFFStress projects the uncracked stress.
            centre = 0.5 * (sxx + syy)
            halfdiff = 0.5 * (sxx - syy)
            eps_reg = jnp.asarray(1e-10, dtype=dtype)
            radius = jnp.sqrt(
                halfdiff * halfdiff + sxy**2 + eps_reg * eps_reg)
            high, low = centre + radius, centre - radius
            ph, pl = smooth_positive(high), smooth_positive(low)
            coeff = (ph - pl) / (2 * radius)
            sp_xx = pl + coeff * (sxx - low)
            sp_yy = pl + coeff * (syy - low)
            sp_xy = coeff * sxy
            plus = jnp.stack((sp_xx, sp_yy, sp_xy), axis=1)
            total = jnp.stack((sxx, syy, sxy), axis=1)
            minus = total - plus
            history = 0.5 * (
                sp_xx * exx + sp_yy * eyy + sp_xy * gamma)
        elif split == "spectral":
            # Paper Eq. (5): Miehe's strain-spectral decomposition in plane
            # strain. This differs from the stress projection implemented by
            # the released Felino ADComputePFFStress snapshot.
            centre = 0.5 * (exx + eyy)
            halfdiff = 0.5 * (exx - eyy)
            eps_reg = jnp.asarray(1e-14, dtype=dtype)
            radius = jnp.sqrt(
                halfdiff * halfdiff + (0.5 * gamma) ** 2
                + eps_reg * eps_reg)
            high, low = centre + radius, centre - radius
            ph, pl = smooth_positive(high), smooth_positive(low)
            coeff = (ph - pl) / (2 * radius)
            ep_xx = pl + coeff * (exx - low)
            ep_yy = pl + coeff * (eyy - low)
            ep_xy = coeff * 0.5 * gamma
            tr_plus = smooth_positive(trace)
            plus = jnp.stack((
                LAMBDA * tr_plus + 2 * MU * ep_xx,
                LAMBDA * tr_plus + 2 * MU * ep_yy,
                2 * MU * ep_xy,
            ), axis=1)
            total = jnp.stack((sxx, syy, sxy), axis=1)
            minus = total - plus
            history = (
                0.5 * LAMBDA * tr_plus**2
                + MU * (ph**2 + pl**2)
            )
        else:
            # Exact three-dimensional Amor volumetric/deviatoric split under
            # plane strain (eps_zz=0), matching RankTwoTensor::deviatoric().
            tr_plus, tr_minus = smooth_positive(trace), trace - smooth_positive(trace)
            dev_xx, dev_yy, dev_zz = (
                exx - trace / 3, eyy - trace / 3, -trace / 3)
            plus = jnp.stack((K_BULK * tr_plus + 2 * MU * dev_xx,
                              K_BULK * tr_plus + 2 * MU * dev_yy,
                              MU * gamma), axis=1)
            minus = jnp.stack((K_BULK * tr_minus,
                               K_BULK * tr_minus,
                               jnp.zeros_like(trace)), axis=1)
            dev_contract = (
                dev_xx**2 + dev_yy**2 + dev_zz**2 + 0.5 * gamma**2)
            history = 0.5 * K_BULK * tr_plus**2 + MU * dev_contract
        return plus, minus, jnp.maximum(history, 0)

    def degradation_average(damage):
        d_qp = jnp.einsum("qa,ea->eq", qp_shape, jnp.clip(damage[tri], 0, 1))
        return jnp.mean((1 - d_qp) ** 2 * (1 - KAPPA) + KAPPA, axis=1)

    def mechanical_terms(u, damage):
        strain, trace, _ = strain_state(u)
        degradation = degradation_average(damage)

        def stress_from_strain(value):
            value_trace = value[:, 0] + value[:, 1]
            plus, minus, _ = split_stress_energy(value, value_trace)
            return degradation[:, None] * plus + minus

        stress = stress_from_strain(strain)
        directions = jnp.eye(3, dtype=dtype)
        tangent_columns = jax.vmap(
            lambda direction: jax.jvp(
                stress_from_strain, (strain,),
                (jnp.broadcast_to(direction, strain.shape),))[1])(directions)
        tangent = jnp.transpose(tangent_columns, (1, 2, 0))
        local_force = areas[:, None] * jnp.einsum("eji,ej->ei", bmat, stress)
        local_stiffness = areas[:, None, None] * jnp.einsum(
            "eia,eij,ejb->eab", bmat, tangent, bmat)
        return local_force, local_stiffness

    mechanical_terms_jit = jax.jit(mechanical_terms)
    mech_row = np.broadcast_to(
        dofs_np[:, :, None], (len(tri_np), 6, 6)).reshape(-1)
    mech_col = np.broadcast_to(
        dofs_np[:, None, :], (len(tri_np), 6, 6)).reshape(-1)
    free_ids = np.flatnonzero(free_u_np)
    fixed_ids = np.flatnonzero(1 - free_u_np)
    external_force = (
        np.zeros(ndof, dtype=np.float64) if external_force_np is None
        else np.asarray(external_force_np, dtype=np.float64))
    if external_force.shape != (ndof,):
        raise ValueError("external_force_np must have one value per displacement DOF")

    def solve_u(damage, boundary, x0):
        # Felino's main app uses Newton + SuperLU. Sparse direct Newton avoids
        # the severe Krylov slowdown as the crack makes the tangent ill-conditioned.
        previous_solution = np.asarray(
            jax.device_get(x0), dtype=np.float64).copy()
        solution = previous_solution.copy()
        boundary_np = np.asarray(jax.device_get(boundary), dtype=np.float64)
        solution[fixed_ids] = boundary_np[fixed_ids]
        residual = np.inf
        newton_iterations = 0

        for newton_iterations in range(41):
            local_force, local_stiffness = (
                np.asarray(jax.device_get(v), dtype=np.float64)
                for v in mechanical_terms_jit(
                    jnp.asarray(solution, dtype=dtype), damage))
            internal = np.bincount(
                dofs_np.reshape(-1), weights=local_force.reshape(-1),
                minlength=ndof)
            force_residual = internal - external_force
            residual = (np.linalg.norm(force_residual[free_ids])
                        / max(np.linalg.norm(external_force[free_ids]),
                              np.linalg.norm(internal), 1e-30))
            if residual <= 1e-8:
                break
            tangent = sp.coo_matrix(
                (local_stiffness.reshape(-1), (mech_row, mech_col)),
                shape=(ndof, ndof)).tocsr()
            tangent_free = tangent[free_ids][:, free_ids].tocsr()
            # Once the crack fully separates the specimen, the upper half has
            # an almost-rigid horizontal mode. MOOSE/SuperLU regularises it
            # numerically; make that pivot stabilisation explicit here.
            diagonal_scale = max(
                float(np.max(np.abs(tangent_free.diagonal()))), 1.0)
            # Update the diagonal in-place. Sparse matrix addition prunes exact
            # zero entries and creates thousands of nominally different CSR
            # fingerprints during spectral crack evolution, defeating cuDSS
            # symbolic-plan reuse even though the FE adjacency is unchanged.
            tangent_free.setdiag(
                tangent_free.diagonal() + 1e-12 * diagonal_scale)
            tangent_free.sort_indices()
            increment = spla.spsolve(
                tangent_free, -force_residual[free_ids])
            solution[free_ids] += increment
            solution[fixed_ids] = boundary_np[fixed_ids]

        return (jnp.asarray(solution, dtype=dtype),
                jnp.asarray(residual, dtype=dtype),
                jnp.asarray(newton_iterations, dtype=jnp.int32))

    def qois(u, damage):
        strain, trace, eps_max = strain_state(u)
        plus, minus, psi = split_stress_energy(strain, trace)
        degradation = degradation_average(damage)
        stress = degradation[:, None] * plus + minus
        szz0 = LAMBDA * trace
        szz_plus = smooth_positive(szz0)
        szz = degradation * szz_plus + szz0 - szz_plus
        sigma_h = (stress[:, 0] + stress[:, 1] + szz) / 3
        if fatigue_energy == "mean_load":
            # Exact ADComputeFatigueEnergy.C mean_load formula, paper Eq. (16).
            psi_eff = (2 * E * eps_max**2 * ((1 + LOAD_RATIO) / 2) ** 2
                       * ((1 - LOAD_RATIO) / 2) ** fatigue_n)
        else:
            # Constant-load accumulation of the split's own tensile energy,
            # i.e. paper Eq. (11) driving measure with the R-correction of
            # Kristensen et al. (2023), Eq. (24).
            psi_eff = psi * (1 - LOAD_RATIO**2)
        if multiply_by_degradation:
            # ADComputeFatigueEnergy multiply_by_D, i.e. the g(d) prefactor
            # written in paper Eq. (11).
            psi_eff = psi_eff * degradation
        crack_area = jnp.sum(jnp.einsum("eij,ej->ei", mass, damage[tri]))
        return psi, psi_eff, crack_area, sigma_h

    def fatigue_factor(alpha_bar):
        asymptote = 2 * alpha_t / jnp.maximum(alpha_bar + alpha_t, tiny)
        return jnp.where(alpha_bar > alpha_t, asymptote**2, 1)

    def solve_d(history, factor, lower):
        # The source uses PETSc vinewtonrsls with SuperLU. Match that strategy
        # directly: assemble the sparse QP on the host and solve each reduced
        # active set with sparse LU. Mechanics and QoIs remain on the GPU.
        history_np, factor_np, lower_np = (
            np.asarray(jax.device_get(v), dtype=np.float64)
            for v in (history, factor, lower))
        lower_np = np.minimum(lower_np, damage_upper_np)
        local_gc = gc_cell_np * factor_np
        history_drive = (1 - KAPPA) * history_np
        ke = (local_gc[:, None, None] * L0 * grad_np
              + (local_gc / L0 + 2 * history_drive)[:, None, None] * mass_np)
        row = np.broadcast_to(tri_np[:, :, None], ke.shape).reshape(-1)
        col = np.broadcast_to(tri_np[:, None, :], ke.shape).reshape(-1)
        matrix = sp.coo_matrix(
            (ke.reshape(-1), (row, col)), shape=(nnode, nnode)).tocsr()
        rhs_np = np.bincount(
            tri_np.reshape(-1),
            weights=np.repeat(2 * history_drive * areas_np / 3, 3),
            minlength=nnode)
        solution = np.clip(
            spla.spsolve(matrix, rhs_np), lower_np, damage_upper_np
        )
        rhs_norm = max(np.linalg.norm(rhs_np), 1e-30)
        qp = osqp.OSQP()
        qp.setup(P=sp.triu(matrix).tocsc(), q=-rhs_np,
                 A=sp.eye(nnode, format="csc"), l=lower_np,
                 u=damage_upper_np, verbose=False,
                 eps_abs=1e-10, eps_rel=1e-8, max_iter=100000,
                 polishing=True, adaptive_rho=True)
        qp.warm_start(x=solution)
        qp_result = qp.solve(raise_error=False)
        if qp_result.x is not None and np.all(np.isfinite(qp_result.x)):
            solution = np.clip(qp_result.x, lower_np, damage_upper_np)
        gradient = matrix @ solution - rhs_np
        # OSQP's dual variable gives the unambiguous bound KKT residual;
        # tolerance-based primal classification is unreliable when many
        # nodes lie only nanometres above the old-value lower bound.
        dual = (qp_result.y if qp_result.y is not None
                and np.all(np.isfinite(qp_result.y))
                else np.zeros_like(solution))
        residual = np.linalg.norm(gradient + dual) / rhs_norm

        return (jnp.asarray(solution, dtype=dtype),
                jnp.asarray(residual, dtype=dtype),
                jnp.asarray(qp_result.info.iter, dtype=jnp.int32))

    scalar_row = np.broadcast_to(
        tri_np[:, :, None], (len(tri_np), 3, 3)).reshape(-1)
    scalar_col = np.broadcast_to(
        tri_np[:, None, :], (len(tri_np), 3, 3)).reshape(-1)
    global_mass = sp.coo_matrix(
        (mass_np.reshape(-1), (scalar_row, scalar_col)),
        shape=(nnode, nnode)).tocsr()
    lumped_mass = np.asarray(global_mass.sum(axis=1)).ravel()
    gx_np, gy_np = b_np[:, 0, 0::2], b_np[:, 1, 1::2]

    def project_sigma(sigma_cell):
        values = np.asarray(jax.device_get(sigma_cell), dtype=np.float64)
        weighted = np.repeat(values * areas_np / 3, 3)
        numerator = np.bincount(
            tri_np.reshape(-1), weights=weighted, minlength=nnode)
        return numerator / np.maximum(lumped_mass, 1e-30)

    def hydrogen_factor(concentration):
        values = np.asarray(jax.device_get(concentration), dtype=np.float64)
        cell_c = np.mean(values[tri_np], axis=1)
        if hydrogen_law == "pressure_vessel":
            cell_wppm = cell_c * 1e6 * M_H / RHO_M
            return jnp.asarray(
                np.clip(0.12 + 0.88 * np.exp(-7 * cell_wppm**2), 0, 1),
                dtype=dtype)
        if hydrogen_law == "jmrt_pipeline":
            cell_wppm = cell_c * 1e6 * M_H / RHO_M
            return jnp.asarray(
                np.clip(0.155 + 0.845 * np.exp(-24.1 * cell_wppm**2), 0, 1),
                dtype=dtype,
            )
        if hydrogen_law != "langmuir":
            raise ValueError(f"unknown hydrogen degradation law: {hydrogen_law}")
        c_imp = cell_c * A_M / RHO_M
        theta = c_imp / (
            c_imp + np.exp(-delta_gb / (8.314 * TEMPERATURE)))
        return jnp.asarray(np.clip(1 - chi_value * theta, 0, 1), dtype=dtype)

    def solve_c(old_concentration, sigma_cell, damage, dt, boundary_ids,
                substeps=1):
        old_np = np.asarray(
            jax.device_get(old_concentration), dtype=np.float64)
        damage_np = np.asarray(jax.device_get(damage), dtype=np.float64)
        sigma_node = project_sigma(sigma_cell)
        grad_sigma_x = np.sum(sigma_node[tri_np] * gx_np, axis=1)
        grad_sigma_y = np.sum(sigma_node[tri_np] * gy_np, axis=1)
        d_cell = np.mean(damage_np[tri_np], axis=1)
        switch = 0.5 * (
            1 + np.tanh((d_cell - PHI_THRESHOLD) / PHI_WIDTH))
        diffusivity = diffusivity_cell_np * (
            1 + CRACK_DIFFUSIVITY_FACTOR * switch)
        diffusion = diffusivity[:, None, None] * grad_np
        beta = V_H / (R_GAS_NMM * TEMPERATURE)
        advection_row = -(
            gx_np * (diffusivity * beta * grad_sigma_x)[:, None]
            + gy_np * (diffusivity * beta * grad_sigma_y)[:, None]
        ) * areas_np[:, None] / 3
        local_operator = diffusion + advection_row[:, :, None]
        operator = sp.coo_matrix(
            (local_operator.reshape(-1), (scalar_row, scalar_col)),
            shape=(nnode, nnode)).tocsr()
        substeps = int(substeps)
        if substeps < 1:
            raise ValueError("diffusion substeps must be positive")
        sub_dt = dt / substeps
        matrix = global_mass / sub_dt + operator
        boundary_ids = np.asarray(boundary_ids, dtype=np.int64)
        free_ids_c = np.setdiff1d(
            np.arange(nnode, dtype=np.int64), boundary_ids)
        boundary_values = np.full(len(boundary_ids), c_environment)
        solution = old_np.copy()
        solution[boundary_ids] = boundary_values
        free_matrix = matrix[free_ids_c][:, free_ids_c].tocsc()
        boundary_term = (
            matrix[free_ids_c][:, boundary_ids] @ boundary_values)
        solve_free = spla.factorized(free_matrix)
        for _ in range(substeps):
            rhs = global_mass @ solution / sub_dt
            rhs_free = rhs[free_ids_c] - boundary_term
            solution[free_ids_c] = solve_free(rhs_free)
            solution = np.maximum(solution, 0)
        residual = np.linalg.norm(
            (matrix @ solution - rhs)[free_ids_c]
        ) / max(np.linalg.norm(rhs_free), 1e-30)
        return (jnp.asarray(solution, dtype=dtype),
                jnp.asarray(residual, dtype=dtype))

    return (solve_u, jax.jit(qois), jax.jit(fatigue_factor), solve_d,
            solve_c, hydrogen_factor)


def main(argv: list[str] | None = None) -> None:
    args = make_parser().parse_args(argv)
    # The geometry-dependent constants are module level because the jitted
    # closures read them; rebind them before any solver is traced.
    globals().update(
        L0=args.length_scale, ALPHA_T=args.alpha_t, U_MAX=args.umax,
        LOAD_RATIO=args.load_ratio, FATIGUE_N=args.fatigue_n)
    load_runtime()
    if args.blocks < 1:
        raise ValueError("--blocks must be positive")
    if args.delta_n < 1:
        raise ValueError("--delta-n must be positive")
    if (not np.isclose(args.fatigue_scale, 1.0)
            or args.fatigue_delay_cycles != 0
            or args.fatigue_initial_boost != 0):
        raise ValueError(
            "source-parameter runs require fatigue_scale=1, "
            "fatigue_delay_cycles=0, and fatigue_initial_boost=0")
    if not 0 <= args.precharge_ratio <= 1:
        raise ValueError("--precharge-ratio must be between zero and one")
    if args.frequency_hz <= 0:
        raise ValueError("--frequency-hz must be positive")
    if args.diffusion_max_step_s <= 0:
        raise ValueError("--diffusion-max-step-s must be positive")
    if not 0 < args.stagger_relaxation <= 1:
        raise ValueError("--stagger-relaxation must lie in (0, 1]")
    if args.gc <= 0 or args.environment_wppm <= 0 or args.delta_gb_j_mol <= 0:
        raise ValueError("--gc, --environment-wppm and --delta-gb-j-mol must be positive")
    if not 0 <= args.chi <= 1:
        raise ValueError("--chi must be between zero and one")
    hydrogen_enabled = (
        args.figure in ("fig6", "fig7", "fig8", "fig10", "fig12")
        if args.hydrogen == "auto"
        else args.hydrogen == "on"
    )
    c_environment = args.environment_wppm * C_ENV
    if hydrogen_enabled and args.split != "spectral":
        raise ValueError("Yang et al. hydrogen figures use the spectral split")
    jax.config.update("jax_enable_x64", args.dtype == "float64")
    dtype = jnp.float64 if args.dtype == "float64" else jnp.float32
    devices = jax.devices()
    if args.platform == "gpu" and not any(d.platform == "gpu" for d in devices):
        raise RuntimeError(f"GPU requested, available devices: {devices}")

    nodes, tri, areas, groups = load_mesh(args.mesh)
    base = (root() / "outputs" / "yang_2026_high_cycle_hydrogen_fatigue"
            / args.figure)
    print(f"mesh={len(nodes)} nodes/{len(tri)} P1 triangles; "
          f"Crack={len(groups['Crack'])} geometric-boundary nodes", flush=True)
    bmat, dofs, mass, grad = preprocess(nodes, tri, areas)
    nnode, ndof = len(nodes), 2 * len(nodes)
    external_force_np = None
    if args.loading == "displacement":
        fixed = np.unique(np.concatenate(
            (2 * groups["BOTTOM"], 2 * groups["BOTTOM"] + 1,
             2 * groups["TOP"] + 1)))
        free_u = np.ones(ndof)
        free_u[fixed] = 0
        boundary_np = np.zeros(ndof)
        boundary_np[2 * groups["TOP"] + 1] = U_MAX
    else:
        # Pin loading of a compact-tension specimen: the lower hole is held
        # and the upper hole carries the load, distributed over the half of
        # its boundary that a pin can push against.
        fixed = np.unique(np.concatenate(
            (2 * groups["BOTTOM"], 2 * groups["BOTTOM"] + 1)))
        free_u = np.ones(ndof)
        free_u[fixed] = 0
        boundary_np = np.zeros(ndof)
        loaded = groups["TOP"]
        centre = nodes[loaded].mean(axis=0)
        bearing = nodes[loaded, 1] >= centre[1]
        if not bearing.any():
            raise RuntimeError("no bearing nodes found on the loaded hole")
        weights = np.zeros(len(loaded))
        weights[bearing] = 1.0 / bearing.sum()
        external_force_np = np.zeros(ndof)
        external_force_np[2 * loaded + 1] = args.pin_load_n * weights
    boundary = jnp.asarray(boundary_np, dtype=dtype)
    solve_u, qois, fatigue_factor, solve_d, solve_c, hydrogen_factor = make_solvers(
        nodes, tri, areas, bmat, dofs, mass, grad, free_u, args.split,
        dtype, args.cg_tol, args.cg_maxiter, args.diffusivity_mm2_s,
        args.gc, args.chi, c_environment, args.delta_gb_j_mol,
        external_force_np=external_force_np,
        hydrogen_law=args.hydrogen_law,
        fatigue_energy=args.fatigue_energy,
        multiply_by_degradation=args.multiply_by_degradation)

    outdir = args.outdir or base / "baseline" / args.split
    outdir.mkdir(parents=True, exist_ok=True)
    for stale_name in ("summary.json", "state.npz"):
        (outdir / stale_name).unlink(missing_ok=True)
    fields = ("block", "cycle", "crack_area_mm2", "crack_tip_x_mm",
              "max_damage", "max_alpha_bar_mpa",
              "min_fatigue_factor", "min_hydrogen_factor",
              "max_concentration_mol_mm3", "mean_concentration_mol_mm3",
              "stagger_iterations", "stagger_residual",
              "u_cg_iterations", "d_cg_iterations", "u_linear_residual",
              "d_linear_residual", "c_linear_residual",
              "block_wall_s", "cumulative_wall_s")
    u = jnp.zeros(ndof, dtype=dtype)
    damage_np = np.zeros(nnode)
    if args.initial_crack_damage:
        damage_np[groups["Crack"]] = 1.0
    damage = jnp.asarray(damage_np, dtype=dtype)
    history = jnp.zeros(len(tri), dtype=dtype)
    alpha_bar = jnp.zeros(len(tri), dtype=dtype)
    concentration = jnp.full(
        nnode, args.precharge_ratio * c_environment if hydrogen_enabled else 0,
        dtype=dtype)
    diffusion_dt = (
        1.0 if args.diffusion_time_mode == "source"
        else args.delta_n / args.frequency_hz)
    diffusion_substeps = max(
        1, int(np.ceil(diffusion_dt / args.diffusion_max_step_s)))
    if hydrogen_enabled and args.precharge_hours > 0:
        # The CT specimens of Fig. 14 are charged for 24 h before cycling,
        # which leaves a diffusion profile rather than the uniform field a
        # precharge ratio would give.
        precharge_s = args.precharge_hours * 3600.0
        concentration, _ = solve_c(
            concentration, jnp.zeros(len(tri), dtype=dtype), damage,
            precharge_s, groups["Crack"],
            max(1, int(np.ceil(precharge_s / args.diffusion_max_step_s))))
        charged = np.asarray(jax.device_get(concentration))
        print(f"pre-charged {args.precharge_hours:g} h: "
              f"c_max={charged.max():.4e}, c_mean={charged.mean():.4e} "
              f"mol/mm^3", flush=True)
    far_edge = nodes[:, 0].max()
    ligament_end = np.flatnonzero(
        (nodes[:, 0] > far_edge - 4.0 * L0) & (np.abs(nodes[:, 1]) < 6.0 * L0))
    crack_plane = np.flatnonzero(np.abs(nodes[:, 1]) < 1.5 * L0)

    def crack_tip_x(field) -> float:
        # A phase-field crack has no sharp tip, so the tip is taken where the
        # damage on the crack plane last falls through one half, which is the
        # usual convention for reporting a(N).
        broken = crack_plane[field[crack_plane] > 0.5]
        return float(nodes[broken, 0].max()) if broken.size else float("nan")

    first_crack_tip = crack_tip_x(damage_np)
    if not np.isfinite(first_crack_tip):
        first_crack_tip = float(nodes[:, 0].min())

    def traversed(field) -> bool:
        if ligament_end.size == 0:
            return False
        values = np.asarray(jax.device_get(field))[ligament_end]
        return bool(values.max() > 0.95)

    snapshot_cycles = set(args.snapshot_cycles or ())
    snapshot_dir = outdir / "snapshots"
    snapshot_files: list[str] = []
    rows, max_linear, start = [], 0.0, time.perf_counter()
    print(f"Yang {args.figure.upper()} split={args.split} device={devices[0]} "
          f"blocks={args.blocks} deltaN={args.delta_n} hydrogen={hydrogen_enabled} "
          f"precharge={args.precharge_ratio:g} dt_H={diffusion_dt:g}s", flush=True)

    with (outdir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for block in range(1, args.blocks + 1):
            tic, damage_start, ucg, dcg = time.perf_counter(), damage, 0, 0
            alpha_start, concentration_start, history_start = (
                alpha_bar, concentration, history)
            fatigue_delta_n = max(
                0, min(args.delta_n,
                       block * args.delta_n - args.fatigue_delay_cycles))
            active_cycles_before = max(
                0, (block - 1) * args.delta_n - args.fatigue_delay_cycles)
            block_fatigue_scale = args.fatigue_scale * (
                1 + args.fatigue_initial_boost * np.exp(
                    -active_cycles_before / args.fatigue_boost_decay_cycles))
            u, ur, ui = solve_u(damage, boundary, u)
            psi, psi_eff, _, sigma_h = qois(u, damage)
            alpha_trial = (
                alpha_start + block_fatigue_scale * fatigue_delta_n * psi_eff)
            fatigue_trial = fatigue_factor(alpha_trial)
            if hydrogen_enabled:
                concentration_trial, cr = solve_c(
                    concentration_start, sigma_h, damage, diffusion_dt,
                    groups["Crack"], diffusion_substeps)
                hydrogen_trial = hydrogen_factor(concentration_trial)
            else:
                concentration_trial, cr = concentration_start, jnp.asarray(0, dtype=dtype)
                hydrogen_trial = jnp.ones(len(tri), dtype=dtype)
            factor = fatigue_trial * hydrogen_trial
            history_trial = jnp.maximum(history_start, psi)
            ucg += int(jax.device_get(ui))
            stagger_residual = float("inf")
            for stagger in range(1, args.max_stagger + 1):
                old, old_alpha, old_concentration = (
                    damage, alpha_trial, concentration_trial)
                damage_candidate, dr, di = solve_d(
                    history_trial, factor, damage_start)
                damage = jnp.maximum(
                    damage_start,
                    old + args.stagger_relaxation
                    * (damage_candidate - old))
                u, ur, ui = solve_u(damage, boundary, u)
                psi, psi_eff, area_dev, sigma_h = qois(u, damage)
                # MOOSE re-executes the fracture MultiApp in every global
                # fixed-point iteration. Its stateful old value stays at the
                # beginning of the time step, while current strain is updated.
                alpha_candidate = (
                    alpha_start
                    + block_fatigue_scale * fatigue_delta_n * psi_eff)
                alpha_trial = (
                    old_alpha + args.stagger_relaxation
                    * (alpha_candidate - old_alpha))
                fatigue_trial = fatigue_factor(alpha_trial)
                if hydrogen_enabled:
                    concentration_candidate, cr = solve_c(
                        concentration_start, sigma_h, damage, diffusion_dt,
                        groups["Crack"], diffusion_substeps)
                    concentration_trial = (
                        old_concentration + args.stagger_relaxation
                        * (concentration_candidate - old_concentration))
                    hydrogen_trial = hydrogen_factor(concentration_trial)
                factor = fatigue_trial * hydrogen_trial
                history_trial = jnp.maximum(history_start, psi)
                ucg += int(jax.device_get(ui))
                dcg += int(jax.device_get(di))
                jax.block_until_ready(damage)
                damage_residual = jnp.linalg.norm(damage - old) / jnp.maximum(
                    jnp.linalg.norm(damage), 1e-30)
                alpha_residual = jnp.linalg.norm(alpha_trial - old_alpha) / jnp.maximum(
                    jnp.linalg.norm(alpha_trial), 1e-30)
                concentration_residual = (
                    jnp.linalg.norm(concentration_trial - old_concentration)
                    / jnp.maximum(jnp.linalg.norm(concentration_trial), 1e-30))
                stagger_residual = float(jax.device_get(jnp.maximum(
                    jnp.maximum(damage_residual, alpha_residual),
                    concentration_residual)))
                if stagger_residual < args.stagger_tol:
                    break
            alpha_bar, concentration = alpha_trial, concentration_trial
            history = history_trial
            urf, drf, crf = (float(jax.device_get(v)) for v in (ur, dr, cr))
            max_linear = max(max_linear, urf, drf, crf)
            # Released MOOSE deck sets accept_on_max_fixed_point_iteration=true.
            damage_np_now = np.asarray(jax.device_get(damage))
            if (not np.isfinite(stagger_residual) or not np.isfinite(urf)
                    or not np.isfinite(drf) or not np.isfinite(crf)
                    or max(urf, crf) > args.linear_residual_tol
                    or drf > args.phase_field_residual_tol):
                # Once the crack has reached the far edge the ligament carries
                # no load and the equilibrium tangent is singular, so a
                # residual failure there is the end of the analysis rather
                # than a solver defect.
                if traversed(damage):
                    print(f"stopping at block {block}: crack has traversed "
                          f"the ligament (u residual {urf:.3e})", flush=True)
                    break
                # Under load control the specimen simply fails once the
                # ligament can no longer carry the applied load, so an
                # equilibrium failure after real crack growth is the end of
                # the test rather than a solver defect.
                grown = crack_tip_x(damage_np_now) - first_crack_tip
                if args.loading == "pin" and grown > args.stop_growth_mm:
                    print(f"stopping at block {block}: load-controlled "
                          f"failure after {grown:.2f} mm of growth "
                          f"(u residual {urf:.3e})", flush=True)
                    break
                raise RuntimeError(f"residual failure block {block}: "
                                   f"stagger={stagger_residual:.3e}, u={urf:.3e}, "
                                   f"d={drf:.3e}, c={crf:.3e}")
            elapsed, cumulative = time.perf_counter() - tic, time.perf_counter() - start
            dh, ah, ffh, hfh, ch = (np.asarray(jax.device_get(v))
                                     for v in (damage, alpha_bar, fatigue_trial,
                                               hydrogen_trial, concentration))
            row = dict(block=block, cycle=block * args.delta_n,
                       crack_area_mm2=float(jax.device_get(area_dev)),
                       crack_tip_x_mm=crack_tip_x(dh),
                       max_damage=float(dh.max()), max_alpha_bar_mpa=float(ah.max()),
                       min_fatigue_factor=float(ffh.min()),
                       min_hydrogen_factor=float(hfh.min()),
                       max_concentration_mol_mm3=float(ch.max()),
                       mean_concentration_mol_mm3=float(ch.mean()),
                       stagger_iterations=stagger,
                       stagger_residual=stagger_residual, u_cg_iterations=ucg,
                       d_cg_iterations=dcg, u_linear_residual=urf,
                       d_linear_residual=drf, c_linear_residual=crf,
                       block_wall_s=elapsed,
                       cumulative_wall_s=cumulative)
            writer.writerow(row)
            handle.flush()
            rows.append(row)
            periodic = (
                args.snapshot_every_blocks
                and block % args.snapshot_every_blocks == 0)
            if block * args.delta_n in snapshot_cycles or periodic:
                snapshot_dir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    snapshot_dir / f"cycle{block * args.delta_n:07d}.npz",
                    nodes=nodes,
                    triangles=tri,
                    displacement=np.asarray(
                        jax.device_get(u)).reshape((-1, 2)),
                    damage=np.asarray(jax.device_get(damage)),
                    alpha_bar=np.asarray(jax.device_get(alpha_bar)),
                    concentration=np.asarray(jax.device_get(concentration)),
                    hydrostatic_stress=np.asarray(jax.device_get(sigma_h)),
                    cycle=block * args.delta_n)
                snapshot_files.append(
                    f"snapshots/cycle{block * args.delta_n:07d}.npz")
            eta = cumulative / block * (args.blocks - block)
            print(f"block={block:3d}/{args.blocks} cycle={block*args.delta_n:6d} "
                  f"Acr={row['crack_area_mm2']:.7e} maxd={row['max_damage']:.5f} "
                  f"CG(u/d)={ucg}/{dcg} res={stagger_residual:.2e}/{urf:.2e}/{drf:.2e} "
                  f"Cmax={row['max_concentration_mol_mm3']:.3e} "
                  f"wall={elapsed:.2f}s ETA={eta:.1f}s", flush=True)

    total = time.perf_counter() - start
    output_files = ["results.csv", "summary.json", *snapshot_files]
    if args.save_state:
        output_files.append("state.npz")
        np.savez_compressed(outdir / "state.npz", nodes=nodes, triangles=tri,
                            displacement=np.asarray(jax.device_get(u)).reshape((-1, 2)),
                            damage=np.asarray(jax.device_get(damage)),
                            history=np.asarray(jax.device_get(history)),
                            alpha_bar=np.asarray(jax.device_get(alpha_bar)),
                            concentration=np.asarray(jax.device_get(concentration)))
    summary = {
        "version": VERSION,
        "paper": f"Yang et al., IJF 210 (2026) 109674, {args.figure.upper()}",
        "doi": "10.1016/j.ijfatigue.2026.109674",
        "device": str(devices[0]), "jax_version": jax.__version__,
        "python_version": platform.python_version(), "dtype": args.dtype,
        "split": args.split, "nodes": nnode, "triangles": len(tri),
        "geometric_crack_boundary_nodes": len(groups["Crack"]),
        "initial_phase_field": (
            "d = 1 imposed on the Crack set" if args.initial_crack_damage
            else "zero; Crack is the actual narrow-notch boundary"),
        "boundary_conditions": (
            f"BOTTOM hole ux=uy=0; {args.pin_load_n} N pin load on the TOP hole"
            if args.loading == "pin" else "TOP uy=0.0005 mm; BOTTOM ux=uy=0"),
        "parameters": {"E_mpa": E, "nu": NU, "Gc_n_per_mm": args.gc,
                       "length_scale_mm": L0, "alpha_T_mpa": ALPHA_T,
                       "umax_mm": U_MAX, "R": LOAD_RATIO, "n": FATIGUE_N,
                       "cycles_per_block": args.delta_n,
                       "fixed_point_relaxation": args.stagger_relaxation,
                       "fatigue_energy": args.fatigue_energy,
                       "multiply_by_degradation": args.multiply_by_degradation,
                       "fatigue_delay_cycles": args.fatigue_delay_cycles,
                       "fatigue_initial_boost": args.fatigue_initial_boost,
                       "fatigue_boost_decay_cycles":
                           args.fatigue_boost_decay_cycles,
                       "loading": args.loading,
                       "pin_load_n": args.pin_load_n,
                       "hydrogen": hydrogen_enabled,
                       "hydrogen_law": args.hydrogen_law,
                       "precharge_ratio": args.precharge_ratio,
                       "environment_wppm": args.environment_wppm,
                       "C_env_mol_mm3": c_environment,
                       "chi": args.chi,
                       "delta_gb_j_mol": args.delta_gb_j_mol,
                       "D_L_mm2_s": args.diffusivity_mm2_s,
                       "diffusion_time_mode": args.diffusion_time_mode,
                       "diffusion_dt_s": diffusion_dt},
        "fatigue_scale": args.fatigue_scale,
        "blocks_completed": len(rows), "cycles_completed": len(rows) * args.delta_n,
        "final_crack_area_mm2": rows[-1]["crack_area_mm2"],
        "final_max_damage": rows[-1]["max_damage"],
        "maximum_linear_residual": max_linear, "total_wall_s": total,
        "estimated_300_block_wall_s": total / len(rows) * DEFAULT_BLOCKS,
        "output_files": output_files,
        "implementation_notes": [
            "AT2, cell history, nodal irreversibility, and consistent P1 Acr integral.",
            "TRI3 three-point integration for quadratic degradation.",
            "Full split stress and consistent tangent; Newton mechanics with sparse SuperLU.",
            "Bound-constrained phase-field QP solved to KKT convergence with OSQP.",
            "Exact Felino mean_load expression with alpha_bar += scale*deltaN*Psi_eff.",
            "Fatigue energy and phase field are re-evaluated in each source-level "
            "global fixed-point iteration.",
            (
                "Released Felino stress-eigenvalue projection."
                if args.split == "felino_stress"
                else "Paper Eq. (5) Miehe strain-spectral decomposition."
                if args.split == "spectral"
                else "Paper Eq. (6) Amor volumetric-deviatoric decomposition."
            ),
            "No fitted cycle-axis shift or fatigue scale is applied."
        ]}
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"completed: {outdir}; wall={total:.2f}s", flush=True)


# Fixed-topology CSR bridge from SciPy matrices to NVIDIA nvmath cuDSS
# (inlined verbatim from the former yang2026_p1_cudss_bridge.py).  The Yang
# port retains its JAX element kernels and OSQP bound-constrained phase-field
# solve; the bridge replaces only unconstrained sparse-direct solves.
def _runtime():
    try:
        import cupy as cp
        import nvmath
        import numpy as np
        from cupyx.scipy.sparse import csr_matrix
    except ImportError as exc:
        raise RuntimeError(
            "Yang accelerated solvers require cupy-cuda12x and "
            "nvmath-python[cu12]"
        ) from exc
    return cp, nvmath, np, csr_matrix


def _fingerprint(matrix) -> tuple:
    csr = matrix.tocsr()
    digest = hashlib.sha256()
    digest.update(csr.indptr.tobytes())
    digest.update(csr.indices.tobytes())
    return csr.shape, digest.digest()


@dataclass
class _System:
    matrix: object
    rhs: object
    solver: object
    data: object
    solves: int = 0

    def close(self) -> None:
        self.solver.free()


class FixedTopologyCudssBridge:
    """Cache a bounded number of planned cuDSS CSR solver systems.

    Sparse arithmetic can remove exact zero entries as the crack evolves, so
    the mechanically identical FE topology can produce several CSR
    fingerprints.  Retaining every fingerprint leaks large cuDSS factorization
    workspaces over a long fatigue run.  A small LRU cache preserves reuse for
    recurring patterns while bounding GPU memory.
    """

    def __init__(self, max_systems: int = 4) -> None:
        if max_systems < 1:
            raise ValueError("max_systems must be positive")
        self.max_systems = max_systems
        self._systems: OrderedDict[tuple, _System] = OrderedDict()
        self.solve_calls = 0
        self.systems_created = 0
        self.systems_evicted = 0
        self.peak_cached_systems = 0

    def _evict_oldest(self, cp) -> None:
        _, system = self._systems.popitem(last=False)
        system.close()
        del system
        self.systems_evicted += 1
        # Return unreferenced CuPy blocks before cuDSS plans the next topology.
        cp.get_default_memory_pool().free_all_blocks()

    def solve(self, matrix, rhs):
        cp, nvmath, np, csr_matrix = _runtime()
        self.solve_calls += 1
        host = matrix.tocsr().astype(np.float64)
        host.sort_indices()
        key = _fingerprint(host)
        rhs_host = np.asarray(rhs, dtype=np.float64)
        system = self._systems.get(key)
        if system is None:
            if len(self._systems) >= self.max_systems:
                self._evict_oldest(cp)
            data = cp.asarray(host.data)
            gpu_matrix = csr_matrix(
                (data, cp.asarray(host.indices, dtype=cp.int32),
                 cp.asarray(host.indptr, dtype=cp.int32)),
                shape=host.shape,
            )
            gpu_rhs = cp.asarray(rhs_host)
            solver = nvmath.sparse.advanced.DirectSolver(gpu_matrix, gpu_rhs)
            solver.plan()
            system = _System(gpu_matrix, gpu_rhs, solver, data)
            self._systems[key] = system
            self.systems_created += 1
            self.peak_cached_systems = max(
                self.peak_cached_systems, len(self._systems))
        else:
            self._systems.move_to_end(key)
            if system.data.size != host.data.size:
                raise RuntimeError("CSR topology changed after cuDSS planning")
            system.data[...] = cp.asarray(host.data)
            system.rhs[...] = cp.asarray(rhs_host)
        system.solver.factorize()
        result = system.solver.solve()
        system.solves += 1
        cp.cuda.Stream.null.synchronize()
        residual = cp.linalg.norm(system.matrix @ result - system.rhs)
        scale = cp.linalg.norm(system.rhs) + 1.0e-30
        if not bool(cp.isfinite(residual / scale)):
            raise RuntimeError("cuDSS returned a non-finite residual")
        return cp.asnumpy(result)

    def factorized(self, matrix):
        """Match scipy.sparse.linalg.factorized while retaining CSR topology."""
        return lambda rhs: self.solve(matrix, rhs)

    def install(self, yang_core) -> None:
        """Replace only sparse-direct entry points used by the Yang core."""
        yang_core.spla.spsolve = self.solve
        yang_core.spla.factorized = self.factorized

    def close(self) -> None:
        for system in self._systems.values():
            system.close()
        self._systems.clear()

    def stats(self) -> dict[str, int]:
        return {
            "solve_calls": self.solve_calls,
            "systems_created": self.systems_created,
            "systems_evicted": self.systems_evicted,
            "peak_cached_systems": self.peak_cached_systems,
            "max_cached_systems": self.max_systems,
        }


def accelerated_entry() -> None:
    """Run the SENT solver with the historical cuDSS wrapper defaults."""
    bridge = FixedTopologyCudssBridge()
    argv = sys.argv[1:]
    repo = Path(__file__).resolve().parents[3]
    defaults = [
        "--platform", "gpu",
        "--dtype", "float64",
        "--mesh", str(
            repo / "open_source_code" / "phase-field-hydrogen-fatigue"
            / "phase-field-hydrogen-fatigue-main" / "01_IJF_2026"
            / "examples" / "SENT10.inp"
        ),
        "--outdir", str(
            repo / "outputs" / "yang_2026_high_cycle_hydrogen_fatigue"
            / "fig4" / "accelerated" / "spectral_cudss"
        ),
    ]
    run_argv = [*defaults, *argv]
    try:
        if "--help" not in argv and "-h" not in argv:
            load_runtime()
            bridge.install(sys.modules[__name__])
        main(run_argv)
        if "--help" not in argv and "-h" not in argv:
            outdir = make_parser().parse_args(run_argv).outdir
            summary_path = outdir / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["acceleration"] = {
                "unconstrained_sparse_direct_solver": "NVIDIA cuDSS",
                "bridge_cache_policy": "bounded LRU by CSR fingerprint",
                "bridge_stats": bridge.stats(),
            }
            summary_path.write_text(
                json.dumps(summary, indent=2), encoding="utf-8")
    finally:
        print(f"cuDSS bridge stats: {json.dumps(bridge.stats())}", flush=True)
        bridge.close()


if __name__ == "__main__":
    accelerated_entry()
