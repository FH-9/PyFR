"""2-rank MPI worker for the one-shot Delta_e exchange test.

Launched via ``mpiexec -n 2 python _mpi_delta_e_worker.py`` by
``test_wmles_sgs.py::test_delta_e_mpi_exchange_two_ranks``.  It mirrors the
one-shot ``comm.Sendrecv`` performed in
``NavierStokesMPIInters.__init__`` (pyfr/solvers/navstokes/inters.py) when
face SGS is active: each rank sends its local per-fpt ``delta_e`` array to
the neighbour rank and, with a matching tag, receives the neighbour's
array into a ``np.empty_like`` buffer.

Each rank asserts the received data matches the neighbour's deterministic
array exactly and in order; any failure exits non-zero so ``mpiexec``
propagates a non-zero return code to the pytest wrapper.
"""
import sys

import numpy as np

from pyfr.mpiutil import get_comm_rank_root


# Per-fpt delta_e for a rank: n_eles element values, each repeated nfp
# times (the block layout produced by get_delta_e_for_inters_np).  The
# value encoding is distinct per (rank, element) so a misrouted message,
# a swapped block, or a reversed buffer is detectable.
def _rank_local_arr(rank, n_eles=4, nfp=3):
    ele_vals = 100.0*rank + np.arange(1, n_eles + 1)
    return np.repeat(ele_vals, nfp).astype(np.float64), n_eles, nfp


def _sendrecv(comm, nbr, local_arr, tag):
    # Faithful copy of the inters.py exchange pattern.
    remote_arr = np.empty_like(local_arr)
    comm.Sendrecv(local_arr, dest=nbr, sendtag=tag,
                  recvbuf=remote_arr, source=nbr, recvtag=tag)
    return remote_arr


def main():
    comm, rank, root = get_comm_rank_root()

    if comm.size != 2:
        if rank == 0:
            print(f'FAIL: expected 2 ranks, got {comm.size}', flush=True)
        return 1

    nbr = 1 - rank
    local_arr, n_eles, nfp = _rank_local_arr(rank)
    local_before = local_arr.copy()

    # --- Primary exchange (tag 0, as next_mpi_tag() yields first) ---------
    remote = _sendrecv(comm, nbr, local_arr, tag=0)
    expected, _, _ = _rank_local_arr(nbr)

    ok = True

    if remote.dtype != local_arr.dtype:
        print(f'[rank {rank}] FAIL dtype: {remote.dtype} != {local_arr.dtype}',
              flush=True)
        ok = False

    if remote.shape != expected.shape:
        print(f'[rank {rank}] FAIL shape: {remote.shape} != {expected.shape}',
              flush=True)
        ok = False
    elif not np.array_equal(remote, expected):
        print(f'[rank {rank}] FAIL values: got {remote} want {expected}',
              flush=True)
        ok = False

    # Per-element block structure must be preserved (constant within each
    # nfp-block) -> the neighbour's element ordering survived the transfer.
    blocks = remote.reshape(n_eles, nfp)
    if not np.all(blocks == blocks[:, :1]):
        print(f'[rank {rank}] FAIL block structure: {remote}', flush=True)
        ok = False

    # The send buffer must be untouched by Sendrecv (no aliasing).
    if not np.array_equal(local_arr, local_before):
        print(f'[rank {rank}] FAIL send buffer mutated', flush=True)
        ok = False

    # --- A second, independent exchange on a different tag -----------------
    # Mirrors multiple interfaces each consuming their own next_mpi_tag();
    # confirms tag pairing keeps payloads from crossing.
    local2 = (local_arr + 0.5).astype(np.float64)
    remote2 = _sendrecv(comm, nbr, local2, tag=1)
    expected2 = (expected + 0.5).astype(np.float64)
    if not np.array_equal(remote2, expected2):
        print(f'[rank {rank}] FAIL tagged exchange crosstalk', flush=True)
        ok = False

    # --- Zero-length edge case (a rank/interface with no fpts) -------------
    empty = np.empty(0, dtype=np.float64)
    remote_empty = _sendrecv(comm, nbr, empty, tag=2)
    if remote_empty.shape != (0,):
        print(f'[rank {rank}] FAIL empty exchange: {remote_empty.shape}',
              flush=True)
        ok = False

    comm.Barrier()
    if ok:
        print(f'[rank {rank}] PASS', flush=True)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
