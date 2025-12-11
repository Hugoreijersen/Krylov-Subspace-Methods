"""PyKokkos GMRES driver for NACA-inspired Stokes systems with block preconditioning."""

import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pykokkos as pk
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from netgen.occ import OCCGeometry
from ngsolve import (
    BilinearForm,
    CoefficientFunction,
    GridFunction,
    H1,
    InnerProduct,
    LinearForm,
    Mesh,
    VectorH1,
    VTKOutput,
    dx,
    div,
    grad,
)

from naca_geometry import occ_naca_profile
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
    row_start = A_indptr[i]
    row_end = A_indptr[i + 1]
    tmp_sum: pk.double = 0.0

    for j in range(row_start, row_end):
        col_idx = A_indices[j]
        val = A_data[j]
        tmp_sum += val * x_view[col_idx]

    y_view[i] = tmp_sum


@pk.workunit
def zero_kernel(i: int, x: pk.View1D[pk.double]):
    x[i] = 0.0


@pk.workunit
def apply_jacobi(i: int, z_view: pk.View1D[pk.double], r_view: pk.View1D[pk.double], d_inv_view: pk.View1D[pk.double]):
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
    """Restarted GMRES with optional Jacobi, GS, or block preconditioning."""

    def __init__(
        self,
        A_csr: sp.csr_matrix,
        b_np: np.ndarray,
        tol: float = 1e-8,
        maxiter: int = 1000,
        restart: int = 75,
        precond: str = "gs",
        block_precond: Optional[Dict[str, object]] = None,
    ):
        self.n = len(b_np)
        self.tol = tol
        self.maxiter = maxiter
        self.restart = restart
        self.precond = precond
        self.block_prec = block_precond
        self.b_norm = np.linalg.norm(b_np)
        self.b = pk.View([self.n], pk.double)

        self.A_data = pk.View([A_csr.data.size], pk.double)
        self.A_indices = pk.View([A_csr.indices.size], pk.int32)
        self.A_indptr = pk.View([A_csr.indptr.size], pk.int32)
        self.A_data.data[:] = A_csr.data.astype(np.float64)
        self.A_indices.data[:] = A_csr.indices.astype(np.int32)
        self.A_indptr.data[:] = A_csr.indptr.astype(np.int32)

        self.x = pk.View([self.n], pk.double)
        self.r = pk.View([self.n], pk.double)
        self.z = pk.View([self.n], pk.double)
        self._host_rhs = np.zeros(self.n, dtype=np.float64)
        self._host_sol = np.zeros(self.n, dtype=np.float64)

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
        if self.precond == "jacobi" and self.d_inv is not None:
            pk.parallel_for(self.n, apply_jacobi, z_view=self.z, r_view=self.r, d_inv_view=self.d_inv)
        elif self.precond == "gs":
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
        elif self.precond == "block":
            self._apply_block_preconditioner(self.r, self.z)
        else:
            pk.parallel_for(self.n, copy_kernel, src=self.r, dst=self.z)

    def _apply_preconditioner_to_view(self, src_view: pk.View1D, dst_view: pk.View1D) -> None:
        if self.precond == "jacobi" and self.d_inv is not None:
            pk.parallel_for(
                self.n,
                apply_jacobi,
                z_view=dst_view,
                r_view=src_view,
                d_inv_view=self.d_inv,
            )
        elif self.precond == "gs":
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
        elif self.precond == "block":
            self._apply_block_preconditioner(src_view, dst_view)
        else:
            pk.parallel_for(self.n, copy_kernel, src=src_view, dst=dst_view)

    def _apply_block_preconditioner(self, src_view: pk.View1D, dst_view: pk.View1D) -> None:
        if self.block_prec is None:
            pk.parallel_for(self.n, copy_kernel, src=src_view, dst=dst_view)
            return

        bp = self.block_prec
        vel_size = bp["vel_size"]

        np.copyto(self._host_rhs, src_view.data)
        rhs_u = self._host_rhs[:vel_size]
        rhs_p = self._host_rhs[vel_size:]

        z_u = bp["A_solver"].solve(rhs_u)
        rhs_p_corr = rhs_p - bp["Bt"].dot(z_u)
        z_p = bp["Mp_solver"].solve(rhs_p_corr)

        self._host_sol[:vel_size] = z_u
        self._host_sol[vel_size:] = z_p
        dst_view.data[:] = self._host_sol

    def _recompute_residual(self) -> float:
        self._spmv(self.x, self.z)
        pk.parallel_for(self.n, copy_kernel, src=self.b, dst=self.r)
        pk.parallel_for(self.n, axpy_kernel, alpha=-1.0, x=self.z, y=self.r)
        return float(np.sqrt(max(self._dot(self.r, self.r), 0.0)))

    def solve(self) -> Tuple[np.ndarray, int, float, List[float]]:
        res_hist: List[float] = []
        iters = 0

        res0 = self._recompute_residual()
        res_hist.append(res0)
        if res0 == 0.0 or (self.b_norm > 0 and res0 / self.b_norm < self.tol):
            return self.x.data[:], 0, res0, res_hist

        m = self.restart
        H = np.zeros((m + 1, m))
        cs = np.zeros(m)
        ss = np.zeros(m)
        V = [pk.View([self.n], pk.double) for _ in range(m + 1)]
        w = pk.View([self.n], pk.double)

        def normalize_into(src_view: pk.View1D, dest_view: pk.View1D) -> float:
            norm2 = self._dot(src_view, src_view)
            norm = np.sqrt(max(norm2, 0.0))
            if norm > 0.0:
                alpha = 1.0 / norm
                pk.parallel_for(self.n, copy_kernel, src=src_view, dst=dest_view)
                pk.parallel_for(self.n, scale_kernel, alpha=alpha, x=dest_view)
            return float(norm)

        while iters < self.maxiter:
            self._apply_preconditioner_to_view(self.r, self.z)
            beta = normalize_into(self.z, V[0])
            if beta == 0.0:
                break
            g = np.zeros(m + 1)
            g[0] = beta

            for j in range(m):
                self._spmv(V[j], w)
                self._apply_preconditioner_to_view(w, w)

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

                for i in range(j):
                    tmp = cs[i] * H[i, j] + ss[i] * H[i + 1, j]
                    H[i + 1, j] = -ss[i] * H[i, j] + cs[i] * H[i + 1, j]
                    H[i, j] = tmp

                denom = np.hypot(H[j, j], H[j + 1, j])
                if denom == 0.0:
                    cs[j] = 1.0
                    ss[j] = 0.0
                else:
                    cs[j] = H[j, j] / denom
                    ss[j] = H[j + 1, j] / denom

                H[j, j] = cs[j] * H[j, j] + ss[j] * H[j + 1, j]
                H[j + 1, j] = 0.0
                g_j = cs[j] * g[j] + ss[j] * g[j + 1]
                g[j + 1] = -ss[j] * g[j] + cs[j] * g[j + 1]
                g[j] = g_j

                res = abs(g[j + 1])
                res_hist.append(res)
                iters += 1

                if self.b_norm > 0 and res / self.b_norm < self.tol:
                    y = np.zeros(j + 1)
                    for k in range(j, -1, -1):
                        sum_hy = 0.0
                        for l in range(k + 1, j + 1):
                            sum_hy += H[k, l] * y[l]
                        y[k] = (g[k] - sum_hy) / H[k, k] if H[k, k] != 0.0 else 0.0

                    for k in range(j + 1):
                        pk.parallel_for(self.n, axpy_kernel, alpha=y[k], x=V[k], y=self.x)

                    final_res = self._recompute_residual()
                    res_hist.append(final_res)
                    return self.x.data[:], iters, final_res, res_hist

                if iters >= self.maxiter:
                    break

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
            if self.b_norm > 0 and res_new / self.b_norm < self.tol:
                break

        final_res = self._recompute_residual()
        res_hist.append(final_res)
        return self.x.data[:], iters, final_res, res_hist


def stokes_system_naca(maxh: float = 0.2) -> Tuple[Mesh, VectorH1, sp.csr_matrix, np.ndarray, np.ndarray, GridFunction, Optional[Dict[str, object]]]:
    naca_geo = OCCGeometry(occ_naca_profile(type="2412", height=4, angle=4, h=0.05), dim=2)
    mesh = Mesh(naca_geo.GenerateMesh(maxh=maxh, grading=0.9))
    mesh.Curve(3)

    nu = 1e-4

    V = VectorH1(mesh, order=3, dirichlet="inlet|wall")
    Q = H1(mesh, order=2)
    X = V * Q

    (u, p), (v, q) = X.TnT()

    a = BilinearForm(X, symmetric=True)
    a += (nu * InnerProduct(grad(u), grad(v)) - div(u) * q - div(v) * p) * dx
    a.Assemble()

    F = LinearForm(X)
    F.Assemble()

    gf = GridFunction(X)
    gfu, _ = gf.components
    gfu.Set(CoefficientFunction((5.0, 0.0)), definedon=mesh.Boundaries("inlet"))

    A_ng = a.mat
    res = F.vec - A_ng * gf.vec

    freedofs = X.FreeDofs()
    free = np.where(freedofs)[0]
    free.sort()

    A_full = sp.csr_matrix(A_ng.CSR())
    res_vec = res.CreateVector()
    res_vec.data = res
    b_full = res_vec.FV().NumPy().copy()

    A_csr = A_full[np.ix_(free, free)]
    b_free = b_full[free]

    vel_cut = V.ndof
    vel_mask = free < vel_cut
    n_vel = int(np.count_nonzero(vel_mask))
    n_total = len(free)
    n_p = n_total - n_vel

    block_prec: Optional[Dict[str, object]] = None
    if n_vel > 0 and n_p > 0:
        press_free_global = free[~vel_mask]
        press_local = press_free_global - vel_cut

        A_vv = A_csr[:n_vel, :n_vel]
        Bt = A_csr[n_vel:, :n_vel]

        qp, qq = Q.TnT()
        mass_form = BilinearForm(Q, symmetric=True)
        mass_form += qp * qq * dx
        mass_form.Assemble()
        M_full = sp.csr_matrix(mass_form.mat.CSR())
        Mp = M_full[np.ix_(press_local, press_local)]

        A_solver = spla.splu(A_vv.tocsc())
        Mp_solver = spla.splu(Mp.tocsc())

        block_prec = {
            "vel_size": n_vel,
            "A_solver": A_solver,
            "Bt": Bt.tocsr(),
            "Mp_solver": Mp_solver,
        }

    return mesh, X, A_csr, b_free, free, gf, block_prec


def run_gmres_naca(nrepeat: int = 1) -> List[Dict[str, object]]:
    mesh_sizes = [0.5, 0.1, 0.05]
    preconds = ["block"]
    restart_values = [20, 50, 75]

    results: List[Dict[str, object]] = []

    print("PyKokkos GMRES for NACA Stokes system")
    print("=" * 70)

    for maxh in mesh_sizes:
        print(f"\nMeshing with maxh = {maxh}")
        mesh, X, A_csr, b_free, free, gf_bc, block_prec = stokes_system_naca(maxh=maxh)

        print("Number of elements:", mesh.ne)
        print("Number of vertices:", mesh.nv)
        dofs = len(free)
        print(f"  Mesh size: {maxh}")
        print(f"  Problem size (free DOFs): {dofs}")
        print(f"  Matrix non-zeros: {A_csr.nnz}")

        print(
            "  {variant:>24} {time:>10} {iters:>8} {relres:>12} {rate:>12}".format(
                variant="Variant", time="Time(s)", iters="Iters", relres="Rel.Res", rate="Rate(it/s)"
            )
        )
        print("  " + "-" * 70)

        last_x_for_vtk: Optional[np.ndarray] = None

        for precond in preconds:
            for restart in restart_values:
                method_name = f"GMRES_{precond}_m{restart}"
                print(f"\tRunning {method_name}...")

                if precond == "block" and block_prec is None:
                    print("\t  Skipping block preconditioner (insufficient block structure).")
                    continue

                gmres_solver = PyKokkosGMRES(
                    A_csr,
                    b_free,
                    tol=1e-8,
                    maxiter=10000,
                    restart=restart,
                    precond=precond,
                    block_precond=block_prec if precond == "block" else None,
                )

                try:
                    iters_list: List[int] = []
                    times: List[float] = []
                    residuals: List[float] = []
                    last_x: Optional[np.ndarray] = None

                    for _ in range(nrepeat):
                        gmres_solver.reset(b_free)
                        start = time.time()
                        x_sol, iters, res, _ = gmres_solver.solve()
                        solve_time = time.time() - start

                        iters_list.append(iters)
                        times.append(solve_time)
                        residuals.append(res)
                        last_x = np.array(x_sol, copy=True)
                finally:
                    try:
                        del gmres_solver
                    except Exception:
                        pass

                avg_time = sum(times) / len(times) if times else float("inf")
                avg_iters = float(sum(iters_list)) / len(iters_list) if iters_list else 0.0
                final_res = residuals[-1] if residuals else float("nan")
                iter_rate = avg_iters / avg_time if avg_time > 0 and np.isfinite(avg_time) else 0.0
                rel_res = final_res / np.linalg.norm(b_free) if np.linalg.norm(b_free) > 0 else float("nan")

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
                    }
                )

                if last_x is not None:
                    last_x_for_vtk = last_x

        if last_x_for_vtk is not None:
            gfuX = GridFunction(X)
            gfuX.vec.data = gf_bc.vec
            full_np = gfuX.vec.FV().NumPy()
            full_np[free] += last_x_for_vtk

            vel = gfuX.components[0]
            pres = gfuX.components[1]

            vtk = VTKOutput(
                ma=mesh,
                coefs=[vel, pres],
                names=["u", "p"],
                filename=f"naca_gmres_maxh_{maxh}",
                subdivision=1,
            )
            vtk.Do()

    print("\n" + "=" * 70)
    print("GLOBAL SUMMARY (GMRES NACA Stokes with various preconditioners)")
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


if __name__ == "__main__":
    print("PyKokkos GMRES for NACA Stokes system")
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
        _ = run_gmres_naca(nrepeat=nrepeat)
    finally:
        pk.finalize()
        print("PyKokkos finalized cleanly (GMRES NACA)")
