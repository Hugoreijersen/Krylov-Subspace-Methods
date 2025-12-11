"""PyKokkos-based restarted GMRES solvers with basic preconditioning."""

import os
import time
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pykokkos as pk
import scipy.sparse as sp
from netgen.occ import OCCGeometry, Rectangle, X, Y
from ngsolve import (
    BND,
    BilinearForm,
    CoefficientFunction,
    GridFunction,
    H1,
    IfPos,
    InnerProduct,
    LinearForm,
    Mesh,
    VectorH1,
    VTKOutput,
    dx,
    grad,
    pi,
    sin,
    x,
    y,
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
    row_start: pk.int32 = 0
    row_end: pk.int32 = 0
    tmp_sum: pk.double = 0.0
    j: pk.int32 = 0
    col_idx: pk.int32 = 0
    val: pk.double = 0.0

    row_start = A_indptr[i]
    row_end = A_indptr[i + 1]
    tmp_sum = 0.0

    for j in range(row_start, row_end):
        col_idx = A_indices[j]
        val = A_data[j]
        tmp_sum += val * x_view[col_idx]

    y_view[i] = tmp_sum


@pk.workunit
def zero_kernel(i: int, x: pk.View1D[pk.double]):
    x[i] = 0.0


@pk.workunit
def apply_jacobi(i, z_view, r_view, d_inv_view):
    """Jacobi preconditioner: z = D^{-1} r"""
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
    """Sequential symmetric Gauss–Seidel preconditioner implemented as a workunit.

    Only the i==0 "thread" performs the forward/backward sweeps.
    """

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


class PyKokkosGMRES:
    """Restarted GMRES supporting optional Jacobi or symmetric Gauss-Seidel preconditioners."""

    def __init__(
        self,
        A_csr: sp.csr_matrix,
        b_np: np.ndarray,
        tol: float = 1e-8,
        maxiter: int = 1000,
        restart: int = 50,
        precond: str = "gs",
    ):
        self.n = len(b_np)
        self.tol = tol
        self.maxiter = maxiter
        self.restart = restart
        self.precond = precond
        self.b_norm = np.linalg.norm(b_np)
        self.b = pk.View([self.n], pk.double)

        # CSR data on device (views)
        self.A_data = pk.View([A_csr.data.size], pk.double)
        self.A_indices = pk.View([A_csr.indices.size], pk.int32)
        self.A_indptr = pk.View([A_csr.indptr.size], pk.int32)

        self.A_data.data[:] = A_csr.data.astype(np.float64)
        self.A_indices.data[:] = A_csr.indices.astype(np.int32)
        self.A_indptr.data[:] = A_csr.indptr.astype(np.int32)

        # Solution and residual
        self.x = pk.View([self.n], pk.double)
        self.r = pk.View([self.n], pk.double)
        self.z = pk.View([self.n], pk.double)  # preconditioned residual

        # Jacobi preconditioner data
        if precond == "jacobi":
            diag = A_csr.diagonal()
            d_inv = np.zeros_like(diag)
            nz = diag != 0.0
            d_inv[nz] = 1.0 / diag[nz]
            self.d_inv = pk.View([self.n], pk.double)
            self.d_inv.data[:] = d_inv
        else:
            self.d_inv = None

        self.b.data[:] = b_np
        self.r.data[:] = b_np
        pk.parallel_for(self.n, zero_kernel, x=self.x)

    def reset(self, b_np: np.ndarray) -> None:
        """Reset RHS and initial guess (x = 0, r = b)."""
        self.b.data[:] = b_np
        self.r.data[:] = b_np
        pk.parallel_for(self.n, zero_kernel, x=self.x)
        pk.parallel_for(self.n, zero_kernel, x=self.z)

    def _dot(self, x: pk.View1D, y: pk.View1D) -> float:
        return pk.parallel_reduce(self.n, reduce_dot_kernel, x=x, y=y)

    def _spmv(self, x_view: pk.View1D, y_view: pk.View1D) -> None:
        pk.parallel_for(
            self.n,
            spmv_kernel,
            y_view=y_view,
            x_view=x_view,
            A_data=self.A_data,
            A_indices=self.A_indices,
            A_indptr=self.A_indptr,
        )

    def apply_preconditioner(self) -> None:
        """Compute ``z ≈ M^{-1} r`` using the chosen preconditioner."""
        if self.precond == "jacobi" and self.d_inv is not None:
            pk.parallel_for(self.n, apply_jacobi, z_view=self.z, r_view=self.r, d_inv_view=self.d_inv)

        elif self.precond == "gs":
            # Symmetric Gauss–Seidel via workunit (i == 0 does both sweeps)
            pk.parallel_for(
                self.n,
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

    def _apply_preconditioner_to_view(self, src_view: pk.View1D, dst_view: pk.View1D) -> None:
        """Compute ``dst_view ≈ M^{-1} src_view`` for an arbitrary view."""
        if self.precond == "jacobi" and self.d_inv is not None:
            pk.parallel_for(
                self.n,
                apply_jacobi,
                z_view=dst_view,
                r_view=src_view,
                d_inv_view=self.d_inv,
            )

        elif self.precond == "gs":
            # Avoid aliasing src and dst for GS: use a temporary RHS copy if needed
            if src_view is dst_view:
                tmp_rhs = pk.View([self.n], pk.double)
                pk.parallel_for(self.n, copy_kernel, src=src_view, dst=tmp_rhs)
                pk.parallel_for(
                    self.n,
                    apply_symmetric_gauss_seidel,
                    N=self.n,
                    z_view=dst_view,
                    r_view=tmp_rhs,
                    A_data=self.A_data,
                    A_indices=self.A_indices,
                    A_indptr=self.A_indptr,
                )
            else:
                pk.parallel_for(
                    self.n,
                    apply_symmetric_gauss_seidel,
                    N=self.n,
                    z_view=dst_view,
                    r_view=src_view,
                    A_data=self.A_data,
                    A_indices=self.A_indices,
                    A_indptr=self.A_indptr,
                )

        else:
            pk.parallel_for(self.n, copy_kernel, src=src_view, dst=dst_view)

    def _recompute_residual(self) -> float:
        self._spmv(self.x, self.z)
        pk.parallel_for(self.n, copy_kernel, src=self.b, dst=self.r)
        pk.parallel_for(self.n, axpy_kernel, alpha=-1.0, x=self.z, y=self.r)
        return np.sqrt(max(self._dot(self.r, self.r), 0.0))

    def solve(self):
        """Restarted GMRES with preconditioning (Jacobi or symmetric GS).

        res_hist stores one *true* residual per restart cycle (including initial).
        """
        res_hist: List[float] = []
        res_iters: List[int] = []
        iters = 0

        # Initial residual r = b - A x (x = 0)
        res0 = self._recompute_residual()
        res_hist.append(res0)
        res_iters.append(iters)
        if res0 == 0.0 or (self.b_norm > 0 and res0 / self.b_norm < self.tol):
            return self.x.data[:], 0, res0, res_hist, res_iters

        m = self.restart
        H = np.zeros((m + 1, m))
        cs = np.zeros(m)
        ss = np.zeros(m)
        V = [pk.View([self.n], pk.double) for _ in range(m + 1)]
        w = pk.View([self.n], pk.double)

        def normalize_into(src_view, dest_view):
            norm2 = self._dot(src_view, src_view)
            norm = np.sqrt(max(norm2, 0.0))
            if norm > 0:
                alpha = 1.0 / norm
                pk.parallel_for(self.n, copy_kernel, src=src_view, dst=dest_view)
                pk.parallel_for(self.n, scale_kernel, alpha=alpha, x=dest_view)
            return norm

        while iters < self.maxiter:
            # Preconditioned initial direction v0 = M^{-1} r / ||M^{-1} r||
            self._apply_preconditioner_to_view(self.r, self.z)
            beta = normalize_into(self.z, V[0])
            if beta == 0.0:
                break
            g = np.zeros(m + 1)
            g[0] = beta

            for j in range(m):
                # w = A * V[j]
                self._spmv(V[j], w)

                # Left preconditioner: w = M^{-1} * w
                self._apply_preconditioner_to_view(w, w)

                # Arnoldi: orthogonalize w against V[0..j]
                for i in range(j + 1):
                    Hij = self._dot(V[i], w)
                    H[i, j] = Hij
                    pk.parallel_for(self.n, axpy_kernel, alpha=-Hij, x=V[i], y=w)

                Hj1j = np.sqrt(max(self._dot(w, w), 0.0))
                H[j + 1, j] = Hj1j
                if Hj1j != 0.0:
                    alpha = 1.0 / Hj1j
                    pk.parallel_for(self.n, copy_kernel, src=w, dst=V[j + 1])
                    pk.parallel_for(self.n, scale_kernel, alpha=alpha, x=V[j + 1])

                # Apply previous Givens rotations
                for i in range(j):
                    tmp = cs[i] * H[i, j] + ss[i] * H[i + 1, j]
                    H[i + 1, j] = -ss[i] * H[i, j] + cs[i] * H[i + 1, j]
                    H[i, j] = tmp

                # New Givens rotation
                denom = np.hypot(H[j, j], H[j + 1, j])
                if denom == 0.0:
                    cs[j] = 1.0
                    ss[j] = 0.0
                else:
                    cs[j] = H[j, j] / denom
                    ss[j] = H[j + 1, j] / denom

                # Apply Givens to H and g
                H[j, j] = cs[j] * H[j, j] + ss[j] * H[j + 1, j]
                H[j + 1, j] = 0.0
                g_j = cs[j] * g[j] + ss[j] * g[j + 1]
                g[j + 1] = -ss[j] * g[j] + cs[j] * g[j + 1]
                g[j] = g_j

                iters += 1

                res_est = abs(g[j + 1])
                if self.b_norm > 0 and res_est / self.b_norm < self.tol:
                    # Solve least-squares H(0:j,0:j) y = g(0:j)
                    y = np.zeros(j + 1)
                    for k in range(j, -1, -1):
                        sum_hy = 0.0
                        for l in range(k + 1, j + 1):
                            sum_hy += H[k, l] * y[l]
                        if H[k, k] != 0.0:
                            y[k] = (g[k] - sum_hy) / H[k, k]
                        else:
                            y[k] = 0.0  # simple fallback to avoid NaN

                    # x += V[:,0:j] * y
                    for k in range(j + 1):
                        pk.parallel_for(self.n, axpy_kernel, alpha=y[k], x=V[k], y=self.x)

                    final_res = self._recompute_residual()
                    res_hist.append(final_res)
                    res_iters.append(iters)
                    return self.x.data[:], iters, final_res, res_hist, res_iters

                if iters >= self.maxiter:
                    break

            # Restart: solve full m×m LS, update x, recompute residual
            y = np.zeros(m)
            for k in range(m - 1, -1, -1):
                sum_hy = 0.0
                for l in range(k + 1, m):
                    sum_hy += H[k, l] * y[l]
                y[k] = (g[k] - sum_hy) / H[k, k] if H[k, k] != 0.0 else 0.0

            for k in range(m):
                pk.parallel_for(self.n, axpy_kernel, alpha=y[k], x=V[k], y=self.x)

            res_new = self._recompute_residual()
            res_hist.append(res_new)
            res_iters.append(iters)
            if self.b_norm > 0 and res_new / self.b_norm < self.tol:
                break

        final_res = self._recompute_residual()
        res_hist.append(final_res)
        res_iters.append(iters)
        return self.x.data[:], iters, final_res, res_hist, res_iters


# ---------- Stokes-like vector problem ----------
def rect_vector_system(maxh: float) -> Tuple[Mesh, VectorH1, sp.csr_matrix, np.ndarray, np.ndarray]:
    """Build a Stokes-like vector Laplace problem on a 2×0.41 channel."""

    rect = Rectangle(2.0, 0.41).Face()
    rect.edges.Min(X).name = "inlet"
    rect.edges.Max(X).name = "outlet"
    rect.edges.Min(Y).name = "wall"
    rect.edges.Max(Y).name = "wall"

    geo = OCCGeometry(rect, dim=2)
    mesh = Mesh(geo.GenerateMesh(maxh=maxh)).Curve(3)

    nu = 0.001
    k = 1
    V = VectorH1(mesh, order=k, dirichlet="wall|inlet")

    u, v = V.TnT()

    a = BilinearForm(V)
    a += (nu * InnerProduct(grad(u), grad(v)) + 1e-6 * InnerProduct(u, v)) * dx
    a.Assemble()

    f = LinearForm(V)
    f.Assemble()

    # GridFunction for BCs
    gfu = GridFunction(V)
    u_in = CoefficientFunction((1.5 * 4 * y * (0.41 - y) / (0.41 * 0.41), 0))
    gfu.Set(u_in, definedon=mesh.Boundaries("inlet"))

    # Extract free DOFs and build reduced CSR system
    freedofs = V.FreeDofs()
    free = np.where(freedofs)[0]

    A_full = sp.csr_matrix(a.mat.CSR())
    b_np = f.vec.FV().NumPy().copy()
    gfu_np = gfu.vec.FV().NumPy()

    # Incorporate Dirichlet BCs: b_modified = b - A * g_D
    b_modified = b_np - A_full @ gfu_np

    A_csr = A_full[np.ix_(free, free)]
    b_free = b_modified[free]

    return mesh, V, A_csr, b_free, free


def run_gmres_stokes(nrepeat: int = 1) -> List[Dict[str, object]]:
    """Run GMRES on the Stokes-like system for multiple meshes and preconditioners."""

    mesh_sizes = [0.0025]
    preconds = ["none", "jacobi", "gs"]
    restart_values = [20, 50, 75]

    results: List[Dict[str, object]] = []

    print("PyKokkos GMRES for Stokes-like vector Laplace system")
    print("=" * 70)

    for maxh in mesh_sizes:
        print(f"\nMeshing with maxh = {maxh}")
        mesh, fes, A_csr, b_free, free = rect_vector_system(maxh)

        print("Number of elements:", mesh.ne)
        print("Number of vertices:", mesh.nv)
        print(f"  Mesh size: {maxh}")
        dofs = len(free)
        print(f"  Problem size: {dofs} DOFs")
        print(f"  Matrix non-zeros: {A_csr.nnz}")

        print(
            "  {variant:>24} {time:>10} {iters:>8} {relres:>12} {rate:>12}".format(
                variant="Variant", time="Time(s)", iters="Iters", relres="Rel.Res", rate="Rate(it/s)"
            )
        )
        print("  " + "-" * 70)

        last_x_for_vtk = None

        for precond in preconds:
            for restart in restart_values:
                method_name = f"GMRES_{precond}_m{restart}"
                print(f"\tRunning {method_name}...")

                gmres_solver = PyKokkosGMRES(
                    A_csr,
                    b_free,
                    tol=1e-8,
                    maxiter=10000,
                    restart=restart,
                    precond=precond,
                )

                try:
                    iters_list = []
                    times = []
                    residuals = []
                    res_hists = []
                    res_iter_hists = []
                    last_x = None

                    for _ in range(nrepeat):
                        gmres_solver.reset(b_free)
                        start = time.time()
                        x_sol, iters, res, res_hist, res_iter_hist = gmres_solver.solve()
                        t_end = time.time() - start

                        iters_list.append(iters)
                        times.append(t_end)
                        residuals.append(res)
                        res_hists.append(res_hist)
                        res_iter_hists.append(res_iter_hist)
                        last_x = x_sol
                finally:
                    try:
                        del gmres_solver
                    except Exception:
                        pass

                avg_time = sum(times) / len(times) if times else float("inf")
                avg_iters = float(sum(iters_list)) / len(iters_list) if iters_list else 0.0
                final_res = residuals[-1] if residuals else float("nan")
                iter_rate = avg_iters / avg_time if avg_time > 0 and np.isfinite(avg_time) else 0.0
                b_norm = np.linalg.norm(b_free)
                rel_res = final_res / b_norm if b_norm > 0 else float("nan")

                print(
                    f"\t{method_name:>24} {avg_time:10.4f} {int(round(avg_iters)):8d} "
                    f"{rel_res:12.2e} {iter_rate:12.1f}"
                )

                results.append(
                    {
                        "maxh": maxh,
                        "dofs": dofs,
                        "precond": precond,
                        "restart": restart,
                        "iterations": avg_iters,
                        "residual": final_res,
                        "rel_res": rel_res,
                        "total_time": avg_time,
                        "rate": iter_rate,
                        # store first residual history for plotting
                        "res_hist": res_hists[0] if res_hists else None,
                        "res_hist_iters": res_iter_hists[0] if res_iter_hists else None,
                    }
                )

                # Remember last solution of this mesh to optionally write to VTK later
                if last_x is not None:
                    last_x_for_vtk = last_x

        # Export solution of last run on this mesh to VTK for visualization
        if last_x_for_vtk is not None:
            gfu = GridFunction(fes)
            full_vec = gfu.vec
            full_vec[:] = 0.0
            full_np = full_vec.FV().NumPy()
            full_np[free] = last_x_for_vtk

            vtk = VTKOutput(
                ma=mesh,
                coefs=[gfu.components[0], gfu.components[1]],
                names=["u_x", "u_y"],
                filename=f"rect_vector_gmres_maxh_{maxh}",
                subdivision=1,
            )
            vtk.Do()

    print("\n" + "=" * 70)
    print("GLOBAL SUMMARY (GMRES Stokes-like with various preconditioners)")
    print("=" * 70)
    print(
        f"{'maxh':>6} {'DOFs':>8} {'Prec':>8} {'m':>4} "
        f"{'Time(s)':>10} {'Iters':>8} {'Rel.Res':>12} {'Rate(it/s)':>12}"
    )
    print("-" * 80)
    for r in results:
        print(
            f"{r['maxh']:>6.3g} {r['dofs']:8d} {r['precond']:>8} {r['restart']:4d} "
            f"{r['total_time']:10.4f} {int(round(r['iterations'])):8d} "
            f"{r['rel_res']:12.2e} {r['rate']:12.1f}"
        ) 

    return results


def plot_gmres_convergence(results: List[Dict[str, object]]) -> None:
    """Plot GMRES convergence (relative residual vs iteration) per mesh size.

    For each mesh size and restart value, creates a PNG showing
    curves for the different preconditioners.
    """
    if not results:
        return

    # Group results by (maxh, restart)
    by_mesh: Dict[Tuple[float, int], List[Dict[str, object]]] = {}
    for r in results:
        key = (r["maxh"], r["restart"])
        by_mesh.setdefault(key, []).append(r)

    for (maxh, restart), entries in by_mesh.items():
        plt.figure(figsize=(6, 4))

        for r in entries:
            res_hist = r.get("res_hist")
            res_iters = r.get("res_hist_iters")
            if not res_hist or not res_iters:
                continue
            rh = np.array(res_hist, dtype=float)
            iters = np.array(res_iters, dtype=float)
            if rh.size == 0 or iters.size != rh.size:
                continue
            # normalize by initial residual for relative convergence
            if rh[0] > 0:
                vals = rh / rh[0]
            else:
                vals = rh
            label = f"{r['precond']} (m={restart})"
            plt.semilogy(iters, vals, label=label)

        plt.xlabel("Iteration")
        plt.ylabel("Relative residual")
        plt.title(f"GMRES convergence, maxh={maxh}, restart={restart}")
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.legend()
        plt.tight_layout()

        filename = f"gmres_stokes_maxh_{maxh}_m{restart}.png"
        plt.savefig(filename, dpi=200)
        plt.close()


if __name__ == "__main__":
    print("PyKokkos GMRES for Stokes-like vector Laplace system")
    print("=" * 40)

    N, M, S, E, nrepeat, space, fill = parse_args()

    if space:
        os.environ.setdefault("PK_EXECUTION_SPACE", space)

    pk.initialize()

    # Warm-up
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
        results = run_gmres_stokes(nrepeat=nrepeat)
        plot_gmres_convergence(results)
    finally:
        pk.finalize()
        print("PyKokkos finalized cleanly (GMRES)")
