"""2-rank MPI worker: real NavierStokesMPIInters cross-rank Delta_e ordering.

Launched via ``mpiexec -n 2 python _mpi_mpiinters_worker.py`` by
``test_wmles_sgs.py::test_mpiinters_delta_e_ordering_two_ranks``.

Unlike ``_mpi_delta_e_worker.py`` (which replicates the Sendrecv pattern in
isolation), this constructs an ACTUAL ``NavierStokesMPIInters`` on each rank
with an SGS model active, so the genuine one-shot Delta_e exchange in its
``__init__`` runs against real elements.  Each rank holds two cube hexes of
distinct edge length L (so delta_e = L/4 is distinct per element), giving a
per-fpt delta_e that VARIES along the interface in two blocks -- an ordering
bug would swap the blocks and be caught.

The two ranks declare their interface connectivity in corresponding order
(element 0 <-> element 0, element 1 <-> element 1), mirroring what PyFR's
mesh partitioner guarantees.  The test then checks the real exchange code
preserves that correspondence: each rank's received delta_e_r must equal the
neighbour's delta_e_l, block for block.
"""
import sys

import numpy as np

from pyfr.backends import get_backend
from pyfr.inifile import Inifile
from pyfr.mpiutil import get_comm_rank_root
from pyfr.readers.native import Connectivity
from pyfr.shapes import HexShape
from pyfr.solvers.navstokes.elements import NavierStokesElements
from pyfr.solvers.navstokes.inters import NavierStokesMPIInters


# Per-rank cube edge lengths -> delta_e = L/4, all distinct across the four
# elements so the per-fpt ordering is fully observable.
_RANK_L = {0: [2.0, 6.0], 1: [4.0, 8.0]}


def _make_cfg():
    cfg = Inifile()
    cfg.set('solver', 'order', '3')
    cfg.set('solver', 'sgs-model', 'vreman')
    cfg.set('solver', 'sgs-include-faces', 'yes')
    cfg.set('solver-elements-hex', 'soln-pts', 'gauss-legendre')
    cfg.set('solver-interfaces-quad', 'flux-pts', 'gauss-legendre')
    cfg.set('solver-interfaces', 'riemann-solver', 'rusanov')
    cfg.set('solver-interfaces', 'ldg-beta', '0.5')
    cfg.set('solver-interfaces', 'ldg-tau', '0.1')
    cfg.set('constants', 'gamma', '1.4')
    cfg.set('constants', 'mu', '1e-3')
    cfg.set('constants', 'Pr', '0.71')
    return cfg


def _build_eles(be, cfg, edges):
    ref = HexShape.std_ele(1)
    hexes = np.stack([(ref + 1.0)/2.0*Li for Li in edges], axis=1)  # (8, ne, 3)
    eles = NavierStokesElements(HexShape, hexes, cfg)
    eles.set_backend(be, '0', 0)
    be.commit()
    return eles


def _extract_extra(kern_factory):
    # delta_e_l / delta_e_r are captured in the comm_flux lambda's closure
    # (passed as **extra to be.kernel); pull the dict back out.
    for cell in (kern_factory.__closure__ or []):
        v = cell.cell_contents
        if isinstance(v, dict) and 'delta_e_l' in v:
            return v
    return None


def main():
    comm, rank, root = get_comm_rank_root()
    if comm.size != 2:
        if rank == 0:
            print(f'FAIL: expected 2 ranks, got {comm.size}', flush=True)
        return 1

    nbr = 1 - rank
    nfp = 16  # order-3 hex quad face: (3+1)^2

    cfg = _make_cfg()
    be = get_backend('openmp', cfg)
    eles = _build_eles(be, cfg, _RANK_L[rank])

    # Interface: both elements contribute face 0, declared in element order.
    lhs = Connectivity(np.array([0, 0]), np.array([0, 1]), {0: ('hex', 0)})

    # Real construction -> runs the one-shot Delta_e Sendrecv in __init__.
    mpiint = NavierStokesMPIInters(be, lhs, nbr, {'hex': eles}, cfg)

    extra = _extract_extra(mpiint.kernels['comm_flux'])
    ok = True
    if extra is None:
        print(f'[rank {rank}] FAIL: delta_e_l/delta_e_r not wired', flush=True)
        return 1

    dl = extra['delta_e_l'].get().ravel()
    dr = extra['delta_e_r'].get().ravel()

    my_de = [L/4.0 for L in _RANK_L[rank]]
    nbr_de = [L/4.0 for L in _RANK_L[nbr]]
    exp_l = np.concatenate([np.full(nfp, my_de[0]), np.full(nfp, my_de[1])])
    exp_r = np.concatenate([np.full(nfp, nbr_de[0]), np.full(nfp, nbr_de[1])])

    if not np.allclose(dl, exp_l):
        print(f'[rank {rank}] FAIL delta_e_l: {dl[::nfp]} want {exp_l[::nfp]}',
              flush=True)
        ok = False

    # The crux: received neighbour values, in the neighbour's block order.
    if not np.allclose(dr, exp_r):
        print(f'[rank {rank}] FAIL delta_e_r: {dr[::nfp]} want {exp_r[::nfp]}',
              flush=True)
        ok = False

    # Block structure preserved (constant within each nfp-block).
    for blk in (dr[:nfp], dr[nfp:]):
        if not np.allclose(blk, blk[0]):
            print(f'[rank {rank}] FAIL delta_e_r block structure', flush=True)
            ok = False

    comm.Barrier()
    if ok:
        print(f'[rank {rank}] PASS', flush=True)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
