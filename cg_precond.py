"""Preconditioned Conjugate Gradient solvers implemented with PyKokkos."""

import os
import time
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pykokkos as pk
import scipy.sparse as sp
from netgen.geom2d import unit_square
from ngsolve import (
    BND,
    BilinearForm,
    CoefficientFunction,
    GridFunction,
    H1,
    IfPos,
    LinearForm,
    Mesh,
    VTKOutput,
    grad,
    dx,
    pi,
    sin,
)

from parse_args import parse_args


@pk.workunit
def copy_kernel(i: int, src: pk.View1D[pk.double], dst: pk.View1D[pk.double]):
    dst[i] = src[i]


@pk.workunit
def axpy_kernel(i: int, alpha: pk.double, x: pk.View1D[pk.double], y: pk.View1D[pk.double]):
    y[i] = alpha * x[i] + y[i]


@pk.workunit
def scale_kernel(i: int, alpha: pk.double, x: pk.View1D[pk.double]):
    x[i] = alpha * x[i]


@pk.workunit
def reduce_dot_kernel(i: int, acc: pk.Acc[pk.double], x: pk.View1D[pk.double], y: pk.View1D[pk.double]):
    acc += x[i] * y[i]


@pk.workunit
def spmv_kernel(
    i: int,
    y_view: pk.View1D[pk.double],
    x_view: pk.View1D[pk.double],
    A_data: pk.View1D[pk.double],
    A_indices: pk.View1D[pk.int32],
    A_indptr: pk.View1D[pk.int32],
):
    row_start: pk.int32 = A_indptr[i]
    row_end: pk.int32 = A_indptr[i + 1]
    tmp_sum: pk.double = 0.0

    for j in range(row_start, row_end):
        col_idx: pk.int32 = A_indices[j]
        val: pk.double = A_data[j]
        tmp_sum += val * x_view[col_idx]

    y_view[i] = tmp_sum


@pk.workunit
def zero_kernel(i: int, x: pk.View1D[pk.double]):
    x[i] = 0.0


@pk.workunit
def apply_jacobi(i: int, z_view: pk.View1D[pk.double], r_view: pk.View1D[pk.double], d_inv_view: pk.View1D[pk.double]):
    """Jacobi preconditioner: ``z = D^{-1} r``."""

    z_view[i] = d_inv_view[i] * r_view[i]


@pk.workunit
def apply_symmetric_gauss_seidel(
    i: int,
    N: int,
    z_view: pk.View1D[pk.double],
    r_view: pk.View1D[pk.double],
    A_data: pk.View1D[pk.double],
    A_indices: pk.View1D[pk.int32],
    A_indptr: pk.View1D[pk.int32],
):
    """Sequential symmetric Gauss-Seidel preconditioner executed by a single workunit."""

    if i != 0:
        return

    for row in range(N):
        z_view[row] = 0.0

    for row in range(N):
        row_start = A_indptr[row]
        row_end = A_indptr[row + 1]
        diag_val = 1.0
        sum_val = 0.0

        for j in range(row_start, row_end):
            col = A_indices[j]
            val = A_data[j]
            if col == row:
                diag_val = val
            elif col < row:
                sum_val += val * z_view[col]

        z_view[row] = (r_view[row] - sum_val) / diag_val if diag_val != 0.0 else r_view[row]

    for row in range(N - 1, -1, -1):
        row_start = A_indptr[row]
        row_end = A_indptr[row + 1]
        diag_val = 1.0
        sum_val = 0.0

        for j in range(row_start, row_end):
            col = A_indices[j]
            val = A_data[j]
            if col == row:
                diag_val = val
            elif col > row:
                sum_val += val * z_view[col]

        if diag_val != 0.0:
            z_view[row] -= sum_val / diag_val


class PyKokkosCGPrecond:
    """Preconditioned Conjugate Gradient solver built on PyKokkos views and kernels."""

    def __init__(self, A_csr: sp.csr_matrix, b_np: np.ndarray, tol: float = 1e-8, maxiter: int = 10000, precond: str = "jacobi"):
        self.n = len(b_np)
        self.tol = tol
        self.maxiter = maxiter
        self.b_norm = np.linalg.norm(b_np)
        self.precond = precond

        self.A_data = pk.View([A_csr.data.size], pk.double)
        self.A_indices = pk.View([A_csr.indices.size], pk.int32)
        self.A_indptr = pk.View([A_csr.indptr.size], pk.int32)

        self.A_data.data[:] = A_csr.data.astype(np.float64)
        self.A_indices.data[:] = A_csr.indices.astype(np.int32)
        self.A_indptr.data[:] = A_csr.indptr.astype(np.int32)

        self.x = pk.View([self.n], pk.double)
        self.r = pk.View([self.n], pk.double)
        self.p = pk.View([self.n], pk.double)
        self.Ap = pk.View([self.n], pk.double)
        self.z = pk.View([self.n], pk.double)

        if precond == "jacobi":
            diag = A_csr.diagonal()
            d_inv = np.zeros_like(diag)
            nz = diag != 0.0
            d_inv[nz] = 1.0 / diag[nz]
            self.d_inv = pk.View([self.n], pk.double)
            self.d_inv.data[:] = d_inv
        else:
            self.d_inv = None

        self.r.data[:] = b_np
        if precond == "none":
            pk.parallel_for(self.n, copy_kernel, src=self.r, dst=self.z)
            pk.parallel_for(self.n, copy_kernel, src=self.r, dst=self.p)
        else:
            self.apply_preconditioner()
            pk.parallel_for(self.n, copy_kernel, src=self.z, dst=self.p)

    def reset(self, b_np: np.ndarray) -> None:
        self.r.data[:] = b_np
        pk.parallel_for(self.n, zero_kernel, x=self.x)
        pk.parallel_for(self.n, zero_kernel, x=self.Ap)
        if self.precond == "none":
            pk.parallel_for(self.n, copy_kernel, src=self.r, dst=self.z)
            pk.parallel_for(self.n, copy_kernel, src=self.r, dst=self.p)
        else:
            self.apply_preconditioner()
            pk.parallel_for(self.n, copy_kernel, src=self.z, dst=self.p)

    def _dot_product(self, x: pk.View1D, y: pk.View1D) -> float:
        return pk.parallel_reduce(self.n, reduce_dot_kernel, x=x, y=y)

    def apply_preconditioner(self) -> None:
        if self.precond == "jacobi":
            pk.parallel_for(self.n, apply_jacobi, z_view=self.z, r_view=self.r, d_inv_view=self.d_inv)
        elif self.precond == "sgs":
            pk.parallel_for(
                1,
                apply_symmetric_gauss_seidel,
                N=self.n,
                z_view=self.z,
                r_view=self.r,
                A_data=self.A_data,
                A_indices=self.A_indices,
                A_indptr=self.A_indptr,
            )
        else:
            pk.parallel_for(self.n, copy_kernel, src=self.r, dst=self.z)

    def solve(self) -> tuple[np.ndarray, int, float, List[float]]:
        rsold = self._dot_product(self.r, self.z)
        res_hist: List[float] = []

        if abs(rsold) < 1e-30:
            final_res = float(np.sqrt(max(rsold, 0.0)))
            res_hist.append(final_res)
            return self.x.data[:], 0, final_res, res_hist

        for it in range(self.maxiter):
            pk.parallel_for(
                self.n,
                spmv_kernel,
                y_view=self.Ap,
                x_view=self.p,
                A_data=self.A_data,
                A_indices=self.A_indices,
                A_indptr=self.A_indptr,
            )

            pAp = self._dot_product(self.p, self.Ap)
            if abs(pAp) < 1e-30:
                break
            alpha = rsold / pAp

            pk.parallel_for(self.n, axpy_kernel, alpha=alpha, x=self.p, y=self.x)
            pk.parallel_for(self.n, axpy_kernel, alpha=-alpha, x=self.Ap, y=self.r)

            self.apply_preconditioner()

            rsnew = self._dot_product(self.r, self.z)
            res_hist.append(float(np.sqrt(max(rsnew, 0.0))))

            if np.sqrt(max(rsnew, 0.0)) < self.tol * self.b_norm:
                rsold = rsnew
                break

            if abs(rsold) < 1e-30:
                break

            beta = rsnew / rsold
            pk.parallel_for(self.n, scale_kernel, alpha=beta, x=self.p)
            pk.parallel_for(self.n, axpy_kernel, alpha=1.0, x=self.z, y=self.p)

            rsold = rsnew

        final_res = float(np.sqrt(max(rsold, 0.0)))
        niters = it + 1 if "it" in locals() else 0
        return self.x.data[:], niters, final_res, res_hist


def test_cg_solver_scaling_precond(nrepeat: int = 1) -> List[Dict[str, object]]:
    """Run CG for several mesh sizes and preconditioners, reporting performance tables."""

    mesh_sizes = [0.1, 0.01, 0.005]
    methods = [
        ("None", "none"),
        ("Jacobi", "jacobi"),
        ("Gauss_Seidel", "sgs"),
    ]

    print("PyKokkos CG (with/without Preconditioner) Scaling Analysis")
    print("=" * 70)

    all_results: List[Dict[str, object]] = []

    for maxh in mesh_sizes:
        print(f"\nMeshing with maxh = {maxh}")
        mesh = Mesh(unit_square.GenerateMesh(maxh=maxh))
        fes = H1(mesh, order=2, dirichlet="top|bottom|left|right")
        u, v = fes.TnT()

        a = BilinearForm(fes)
        a += grad(u) * grad(v) * dx
        a.Assemble()

        f = LinearForm(fes)
        f += 1.0 * v * dx
        f.Assemble()

        gfu = GridFunction(fes)
        gfu.Set(CoefficientFunction(IfPos(y - 1 + 1e-10, sin(pi * x), 0)), BND)

        freedofs = fes.FreeDofs()
        free = np.where(freedofs)[0]

        A_scipy_full = sp.csr_matrix(a.mat.CSR())

        b_np = f.vec.FV().NumPy().copy()
        gfu_np = gfu.vec.FV().NumPy()

        b_modified = b_np - A_scipy_full @ gfu_np

        A_csr = A_scipy_full[np.ix_(free, free)]
        b_free = b_modified[free]

        print("Number of elements:", mesh.ne)
        print("Number of vertices:", mesh.nv)
        print("Number of nodes elements:", mesh.nnodes)
        print("Number of faces elements:", mesh.nface)
        print("Number of nfacet elements:", mesh.nfacet)
        print("Number of nedge elements:", mesh.nedge)

        print(f"  Mesh size: {maxh}")
        dofs = len(free)
        print(f"  Problem size: {dofs} DOFs")
        print(f"  Matrix non-zeros: {A_csr.nnz}")

        print("  " + "Method          Time(s)    Iters    Rel.Res      Rate(it/s)")
        print("  " + "-" * 61)

        last_x: Optional[np.ndarray] = None
        for method_name, precond in methods:
            print(f"    Running {method_name} preconditioner...")
            times: List[float] = []
            iters_list: List[int] = []
            res_list: List[float] = []
            residual_histories: List[List[float]] = []

            solver = PyKokkosCGPrecond(A_csr, b_free, precond=precond)
            try:
                for _ in range(nrepeat):
                    solver.reset(b_free)
                    start_time = time.time()
                    x_sol, iterations, residual, res_hist = solver.solve()
                    solve_time = time.time() - start_time
                    times.append(solve_time)
                    iters_list.append(iterations)
                    res_list.append(residual)
                    residual_histories.append(res_hist)
                    last_x = x_sol
            finally:
                try:
                    del solver
                except Exception:
                    pass

            avg_time = sum(times) / len(times) if times else float("inf")
            avg_iters = float(sum(iters_list)) / len(iters_list) if iters_list else 0.0
            final_res = res_list[-1] if res_list else float("nan")
            iter_rate = avg_iters / avg_time if avg_time > 0 and np.isfinite(avg_time) else 0.0
            rel_res = final_res / np.linalg.norm(b_free) if np.linalg.norm(b_free) > 0 else float("nan")

            print(
                f"    {method_name:<14} {avg_time:8.4f} {int(round(avg_iters)):7d} "
                f"{rel_res:10.2e} {iter_rate:11.1f}"
            )

            all_results.append(
                {
                    "maxh": maxh,
                    "dofs": dofs,
                    "method": method_name,
                    "iterations": avg_iters,
                    "residual": final_res,
                    "rel_res": rel_res,
                    "total_time": avg_time,
                    "rate": iter_rate,
                    "residual_histories": residual_histories,
                }
            )

            if last_x is not None:
                gfu_full = GridFunction(fes)
                full_vec = gfu_full.vec
                full_vec[:] = 0.0
                full_np = full_vec.FV().NumPy()
                full_np[free] = last_x

                vtk = VTKOutput(
                    ma=mesh,
                    coefs=[gfu_full],
                    names=[f"u_{method_name}"],
                    filename=f"base_0.1_maxh_{maxh}",
                    subdivision=1,
                )
                vtk.Do()

        if "A_csr" in locals():
            try:
                del A_csr
            except Exception:
                pass

    print("\n" + "=" * 70)
    print("GLOBAL SUMMARY (all mesh sizes and methods)")
    print("=" * 70)
    print(
        f"{'maxh':>6} {'DOFs':>8} {'Method':>12} {'Time(s)':>10} "
        f"{'Iters':>8} {'Rel.Res':>12} {'Rate(it/s)':>12}"
    )
    print("-" * 70)
    for r in all_results:
        print(
            f"{r['maxh']:>6.3g} {r['dofs']:8d} {r['method']:>12} "
            f"{r['total_time']:10.4f} {int(round(r['iterations'])):8d} "
            f"{r['rel_res']:12.2e} {r['rate']:12.1f}"
        )

    return all_results


def plot_convergence(all_results: List[Dict[str, object]], output_path: str = "cg_precond_convergence.png") -> None:
    """Create a convergence plot that compares all recorded residual histories."""

    if not all_results:
        return

    methods_data: Dict[str, List[float]] = {}
    for entry in all_results:
        method = entry["method"]
        residual_histories = entry.get("residual_histories", [])
        if not residual_histories:
            continue
        methods_data.setdefault(method, residual_histories[0])

    if not methods_data:
        return

    plt.figure(figsize=(6, 4))
    for method, res_hist in methods_data.items():
        res_hist = np.array(res_hist, dtype=float)
        iters = np.arange(len(res_hist))
        y_vals = res_hist / res_hist[0] if res_hist.size > 0 and res_hist[0] > 0 else res_hist
        plt.semilogy(iters, y_vals, label=method)

    plt.xlabel("Iteration")
    plt.ylabel("Relative residual")
    plt.title("CG convergence with different preconditioners")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


if __name__ == "__main__":
    print("PyKokkos Preconditioned CG Solver Analysis")
    print("=" * 40)

    N, M, S, E, nrepeat, space, fill = parse_args()

    if space:
        os.environ.setdefault("PK_EXECUTION_SPACE", space)

    pk.initialize()

    n_warm = 128
    warm_x = pk.View([n_warm], pk.double)
    warm_y = pk.View([n_warm], pk.double)
    for _ in range(3):
        pk.parallel_for(n_warm, copy_kernel, src=warm_x, dst=warm_y)
        pk.fence()
        pk.parallel_for(n_warm, axpy_kernel, alpha=1.0, x=warm_x, y=warm_y)
        pk.fence()
        _ = pk.parallel_reduce(n_warm, reduce_dot_kernel, x=warm_x, y=warm_y)
        pk.fence()

    try:
        results = test_cg_solver_scaling_precond(nrepeat=nrepeat)
        plot_path = "cg_precond_convergence_lambda_0.1.png"
        # plot_convergence(results, output_path=plot_path)
        print("\nCG + preconditioner comparison complete!")
        print(f"Convergence plot saved to '{plot_path}'")
    finally:
        pk.finalize()
        print("PyKokkos finalized cleanly (preconditioned)")
