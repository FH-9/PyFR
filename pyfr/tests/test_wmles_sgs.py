import os
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from mpi4py import MPI

from pyfr.backends import get_backend
from pyfr.inifile import Inifile
from pyfr.shapes import (BaseShape, HexShape, PriShape, PyrShape, QuadShape,
                         TetShape, TriShape)
from pyfr.readers.native import Connectivity
from pyfr.solvers.navstokes.elements import NavierStokesElements
from pyfr.solvers.navstokes.inters import (
    NavierStokesAlgWallBCInters, NavierStokesEqODEWallBCInters,
    NavierStokesNoSlpAdiaWallBCInters)
from pyfr.solvers.navstokes.wmles import (_check_containment, _invert_geomap,
                                          setup_matching_groups)


# Concrete shape classes paired with their analytic reference-element
# centroid (centre of mass in PyFR's [-1, 1] standard coordinates) and
# reference-element volume.
_SHAPES = [
    (QuadShape, [0.0, 0.0],              4.0),
    (HexShape,  [0.0, 0.0, 0.0],         8.0),
    (TriShape,  [-1/3, -1/3],            2.0),
    (TetShape,  [-1/2, -1/2, -1/2],      4/3),
    (PriShape,  [-1/3, -1/3, 0.0],       4.0),
    (PyrShape,  [0.0, 0.0, -1/2],        8/3),
]

# Standard solution-point rules per element type (as used throughout the
# PyFR example configs); needed to instantiate a shape.
_SOLN_PTS = {
    'quad': 'gauss-legendre',
    'hex':  'gauss-legendre',
    'tri':  'williams-shunn',
    'tet':  'shunn-ham',
    'pri':  'williams-shunn~gauss-legendre',
    'pyr':  'gauss-legendre',
}


def _mkcfg(name, order):
    cfg = Inifile()
    cfg.set('solver', 'order', str(order))
    cfg.set(f'solver-elements-{name}', 'soln-pts', _SOLN_PTS[name])
    return cfg


# ---------------------------------------------------------------------------
# BaseShape.std_ele_centroid (highly unit-testable: pure geometry)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('shape,centroid,_vol', _SHAPES,
                         ids=[s[0].name for s in _SHAPES])
def test_std_ele_centroid_value(shape, centroid, _vol):
    assert np.allclose(shape.std_ele_centroid, centroid)


@pytest.mark.parametrize('shape', [s[0] for s in _SHAPES],
                         ids=[s[0].name for s in _SHAPES])
def test_std_ele_centroid_is_readonly(shape):
    # Centroids are cached on the class and shared; they must not be
    # mutable (a stray write would corrupt every consumer).
    assert shape.std_ele_centroid.flags.writeable is False
    with pytest.raises(ValueError):
        shape.std_ele_centroid[0] = 1.0


@pytest.mark.parametrize('shape', [s[0] for s in _SHAPES if s[0] is not PyrShape],
                         ids=[s[0].name for s in _SHAPES if s[0] is not PyrShape])
def test_std_ele_centroid_matches_vertex_mean(shape):
    # For every shape except the pyramid the centroid is the arithmetic
    # mean of the linear-element vertices (true by symmetry).
    assert np.allclose(shape.std_ele_centroid, shape.std_ele(1).mean(axis=0))


def test_std_ele_centroid_pyramid_is_analytic():
    # The square-base pyramid centroid sits at h/4 from the base, NOT at
    # the vertex mean, so it is set analytically.
    assert np.allclose(PyrShape.std_ele_centroid, [0.0, 0.0, -0.5])
    assert not np.allclose(PyrShape.std_ele_centroid,
                           PyrShape.std_ele(1).mean(axis=0))


def test_std_ele_centroid_only_on_concrete_subclasses():
    # The abstract base must not carry a centroid — it is added per
    # concrete shape by __init_subclass__.
    assert not hasattr(BaseShape, 'std_ele_centroid')


# ---------------------------------------------------------------------------
# Delta_e = (V_e / N_upts)^(1/d) via element quadrature
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('shape,_centroid,refvol', _SHAPES,
                         ids=[s[0].name for s in _SHAPES])
def test_element_quadrature_recovers_reference_volume(shape, _centroid, refvol):
    # The SGS filter width uses V_e = sum_q w_q |J|_q.  For the reference
    # element the geometric Jacobian is unity, so the element-quadrature
    # weights must integrate to the known reference-element volume.
    s = shape(None, _mkcfg(shape.name, 2))
    assert np.isclose(s._eqrule.wts.sum(), refvol)


@pytest.mark.parametrize('shape,_centroid,refvol', _SHAPES,
                         ids=[s[0].name for s in _SHAPES])
def test_delta_e_formula(shape, _centroid, refvol):
    # delta_e = (V_e / N_upts)^(1/d).  Drive it with the reference-element
    # volume and check against an independent reimplementation.
    s = shape(None, _mkcfg(shape.name, 3))

    V_e = s._eqrule.wts.sum()
    delta_e = (V_e/s.nupts)**(1.0/s.ndims)

    assert np.isclose(delta_e, (refvol/s.nupts)**(1.0/shape.ndims))
    # Filter width is a positive length scale bounded by the element size.
    assert 0.0 < delta_e < refvol**(1.0/shape.ndims)


# ---------------------------------------------------------------------------
# NavierStokesElements.get_delta_e_for_inters_np ordering
# ---------------------------------------------------------------------------

def _call_get_delta_e(delta_e_np, nfacefpts, eidxs, fidx):
    # Exercise the method against a minimal stub so no backend / mesh is
    # required: it only touches self._delta_e_np and self.nfacefpts[fidx].
    stub = SimpleNamespace(_delta_e_np=np.asarray(delta_e_np, dtype=float),
                           nfacefpts=nfacefpts)
    return NavierStokesElements.get_delta_e_for_inters_np(stub, eidxs, fidx)


def test_get_delta_e_for_inters_np_repeats_per_element():
    delta_e = [10.0, 20.0, 30.0]
    nfp = 4
    eidxs = np.array([2, 0, 1])

    out = _call_get_delta_e(delta_e, {0: nfp}, eidxs, fidx=0)

    # Tuple-of-arrays convention consumed by _const_mat / _get_inter_arrays.
    assert isinstance(out, tuple) and len(out) == 1
    de = out[0]
    assert de.shape == (len(eidxs)*nfp,)

    # Each element's value is REPEATED nfp times (block layout), in the
    # order given by eidxs — distinct values catch a repeat-vs-tile bug.
    assert np.array_equal(de, np.repeat([30.0, 10.0, 20.0], nfp))


def test_get_delta_e_for_inters_np_uses_face_specific_nfp():
    # nfp is looked up per face index, so different faces yield different
    # per-fpt lengths.
    delta_e = [5.0, 7.0]
    eidxs = np.array([0, 1])
    nfacefpts = {0: 3, 1: 9}

    out0 = _call_get_delta_e(delta_e, nfacefpts, eidxs, fidx=0)[0]
    out1 = _call_get_delta_e(delta_e, nfacefpts, eidxs, fidx=1)[0]

    assert np.array_equal(out0, np.repeat([5.0, 7.0], 3))
    assert np.array_equal(out1, np.repeat([5.0, 7.0], 9))


# ---------------------------------------------------------------------------
# _invert_geomap: physical -> reference Newton round-trip
# ---------------------------------------------------------------------------

def _linear_hex_sbasis():
    cfg = Inifile()
    cfg.set('solver', 'order', '3')
    cfg.set('solver-elements-hex', 'soln-pts', 'gauss-legendre')
    # nspts=8 -> linear (trilinear) geometry basis.
    s = HexShape(8, cfg)
    return s, s.sbasis


def test_invert_geomap_round_trip_affine():
    # Map the reference hex through a known non-trivial affine transform
    # (scale + shear + translation), forward-evaluate a set of interior
    # reference points, and check Newton recovers them to ~eps.
    s, sb = _linear_hex_sbasis()
    ref_verts = s.spts                                   # (8, 3)

    A = np.array([[2.0, 0.3, 0.0],
                  [0.0, 1.5, 0.2],
                  [0.1, 0.0, 3.0]])
    b = np.array([5.0, -2.0, 1.0])
    phys_verts = ref_verts @ A.T + b

    rng = np.random.default_rng(0)
    xi_true = rng.uniform(-0.85, 0.85, size=(7, 3))
    plocs = sb.nodal_basis_at(xi_true) @ phys_verts      # forward map

    q = xi_true.shape[0]
    spts = np.repeat(phys_verts[:, None, :], q, axis=1)  # (8, q, 3)
    kt0 = np.tile(s.std_ele_centroid, (q, 1))            # centroid guess

    xi_rec = _invert_geomap(sb, spts, plocs, kt0, niters=50, rtol=1e-12,
                            etype='hex', fidx=0)

    assert np.allclose(xi_rec, xi_true, atol=1e-10)


def test_invert_geomap_recovers_centroid():
    # The element centroid must map back to the reference centroid.
    s, sb = _linear_hex_sbasis()
    ref_verts = s.spts

    A = np.diag([2.0, 3.0, 0.5])
    phys_verts = ref_verts @ A.T + np.array([1.0, 1.0, 1.0])

    x_c = sb.nodal_basis_at(s.std_ele_centroid[None, :]) @ phys_verts  # (1, 3)
    spts = phys_verts[:, None, :]                                      # (8,1,3)
    kt0 = s.std_ele_centroid[None, :].copy()

    xi_rec = _invert_geomap(sb, spts, x_c, kt0, niters=50, rtol=1e-12,
                            etype='hex', fidx=0)

    assert np.allclose(xi_rec, s.std_ele_centroid, atol=1e-12)


def test_invert_geomap_raises_on_nonconvergence():
    # A degenerate (collapsed) element gives a singular Jacobian / target
    # that Newton cannot reach; the helper must raise rather than return
    # a bogus reference coordinate.
    s, sb = _linear_hex_sbasis()
    ref_verts = s.spts

    # Collapse the element to a plane (zero z-extent) and ask for a point
    # off that plane -> unreachable.
    phys_verts = ref_verts.copy().astype(float)
    phys_verts[:, 2] = 0.0
    plocs = np.array([[0.0, 0.0, 1.0]])
    spts = phys_verts[:, None, :]
    kt0 = s.std_ele_centroid[None, :].copy()

    with pytest.raises((RuntimeError, np.linalg.LinAlgError)):
        _invert_geomap(sb, spts, plocs, kt0, niters=50, rtol=1e-10,
                       etype='hex', fidx=0)


# ---------------------------------------------------------------------------
# _check_containment
# ---------------------------------------------------------------------------

def test_check_containment_inside_passes():
    pts = np.array([[0.0, 0.0, 0.0], [0.5, -0.3, 0.9], [-0.9, 0.9, -0.9]])
    # Should not raise.
    _check_containment(pts, HexShape, 0.05, 'hex', 0)


def test_check_containment_borderline_within_tol_passes():
    # tol_abs = 2*tol_frac = 0.1, so |xi| up to 1.1 is tolerated.
    pts = np.array([[1.05, 0.0, 0.0]])
    _check_containment(pts, HexShape, 0.05, 'hex', 0)


def test_check_containment_outside_raises():
    pts = np.array([[1.5, 0.0, 0.0]])
    with pytest.raises(RuntimeError, match='outside reference element'):
        _check_containment(pts, HexShape, 0.05, 'hex', 0)


def test_check_containment_reports_only_bad_points():
    # One of three points is well outside -> raise, and the message counts
    # exactly the offending point.
    pts = np.array([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0], [0.1, 0.1, 0.1]])
    with pytest.raises(RuntimeError, match='for 1 flux points'):
        _check_containment(pts, HexShape, 0.05, 'hex', 0)


# ===========================================================================
# SGS eddy-viscosity macros (sgs.mako): compiled-kernel output vs numpy.
#
# These exercise the actual mako-rendered, C-compiled kernel through an
# OpenMP backend (no mesh required) and compare against an independent
# numpy reference.  They are skipped automatically if no OpenMP-capable C
# compiler is found on the host.
# ===========================================================================

# Candidate C compilers to try for the OpenMP backend (Apple clang lacks
# -fopenmp; GCC from Homebrew works).  None = the backend default ('cc').
_CC_CANDIDATES = [None, 'gcc-15', 'gcc-14', 'gcc-13', 'gcc-12', 'gcc']


def _make_sgs_backend():
    for cc in _CC_CANDIDATES:
        cfg = Inifile()
        if cc is not None:
            cfg.set('backend-openmp', 'cc', cc)
        try:
            be = get_backend('openmp', cfg)
            be.pointwise.register('pyfr.tests.sgs_test_kernel')
            # Smoke-compile the kernel to confirm the toolchain works.
            _run_sgs_kernel(be, 'vreman', np.zeros((1, 3, 3)),
                            np.ones(1), np.ones(1), c_vreman=0.07)
            return be
        except Exception:
            continue
    return None


@pytest.fixture(scope='module')
def sgs_backend():
    be = _make_sgs_backend()
    if be is None:
        pytest.skip('no OpenMP-capable C compiler available for SGS kernel '
                    'tests')
    return be


def _run_sgs_kernel(be, model, alpha, rho, delta, **cextra):
    # alpha: (n, ndims, ndims) with alpha[e, i, j] = du_i/dx_j.
    n, ndims, _ = alpha.shape

    rho_m = be.matrix((1, n), initval=np.asarray(rho, dtype=float)[None, :])
    delta_m = be.matrix((1, n), initval=np.asarray(delta, dtype=float)[None, :])
    # Pointwise arg fpdtype_t[ndims][ndims] over ndim=1 -> ioshape
    # (ndims, ndims, n) with the iteration axis last.
    alpha_m = be.matrix((ndims, ndims, n), initval=np.moveaxis(alpha, 0, -1))
    musgs_m = be.matrix((1, n), initval=np.zeros((1, n)))

    tplargs = dict(ndims=ndims, sgs_model=model, **cextra)
    k = be.kernel('sgs_test_kernel', tplargs=tplargs, dims=[n],
                  rho=rho_m, alpha=alpha_m, delta_e=delta_m, musgs=musgs_m)
    k.run()
    return musgs_m.get().ravel()


def _np_vreman(alpha, rho, delta, C):
    # Reference Vreman in the code's index convention (alpha_ij = du_i/dx_j,
    # beta = Delta^2 alpha alpha^T).  Works for ndims in {2, 3}.
    out = np.empty(len(alpha))
    for e, a in enumerate(alpha):
        nd = a.shape[0]
        beta = delta[e]**2*(a @ a.T)
        B = sum(beta[i, i]*beta[j, j] - beta[i, j]**2
                for i in range(nd) for j in range(i + 1, nd))
        an2 = (a*a).sum()
        nu_t = C*np.sqrt(max(B, 0.0)/an2) if an2 > 1e-15 else 0.0
        out[e] = rho[e]*nu_t
    return out


def _np_sigma(alpha, rho, delta, C):
    # Reference sigma model via SVD ground-truth.  Singular values of alpha
    # are exactly sqrt(eigenvalues(alpha^T alpha)).
    out = np.empty(len(alpha))
    for e, a in enumerate(alpha):
        s1, s2, s3 = np.linalg.svd(a, compute_uv=False)   # descending
        D = (s3*(s1 - s2)*(s2 - s3)/s1**2) if s1**2 > 1e-15 else 0.0
        out[e] = rho[e]*(C*delta[e])**2*D
    return out


# --- Vreman ---------------------------------------------------------------

def test_vreman_kernel_matches_numpy_3d(sgs_backend):
    rng = np.random.default_rng(1)
    n = 16
    alpha = rng.standard_normal((n, 3, 3))
    rho = rng.uniform(0.5, 2.0, n)
    delta = rng.uniform(0.1, 1.0, n)

    got = _run_sgs_kernel(sgs_backend, 'vreman', alpha, rho, delta,
                          c_vreman=0.07)
    ref = _np_vreman(alpha, rho, delta, 0.07)
    assert np.allclose(got, ref, atol=1e-12, rtol=1e-10)


def test_vreman_kernel_matches_numpy_2d(sgs_backend):
    rng = np.random.default_rng(2)
    n = 16
    alpha = rng.standard_normal((n, 2, 2))
    rho = rng.uniform(0.5, 2.0, n)
    delta = rng.uniform(0.1, 1.0, n)

    got = _run_sgs_kernel(sgs_backend, 'vreman', alpha, rho, delta,
                          c_vreman=0.07)
    ref = _np_vreman(alpha, rho, delta, 0.07)
    assert np.allclose(got, ref, atol=1e-12, rtol=1e-10)


def test_vreman_kernel_zero_for_uniform_flow(sgs_backend):
    # Vanishing velocity gradient -> zero SGS viscosity (the alpha_norm2
    # guard path).
    alpha = np.zeros((3, 3, 3))
    rho = np.array([1.0, 1.2, 0.8])
    delta = np.array([0.5, 0.5, 0.5])
    got = _run_sgs_kernel(sgs_backend, 'vreman', alpha, rho, delta,
                          c_vreman=0.07)
    assert np.allclose(got, 0.0)


# --- Sigma ----------------------------------------------------------------

def test_sigma_kernel_matches_numpy_3d(sgs_backend):
    rng = np.random.default_rng(3)
    n = 16
    alpha = rng.standard_normal((n, 3, 3))
    rho = rng.uniform(0.5, 2.0, n)
    delta = rng.uniform(0.1, 1.0, n)

    got = _run_sgs_kernel(sgs_backend, 'sigma', alpha, rho, delta,
                          c_sigma=1.35)
    ref = _np_sigma(alpha, rho, delta, 1.35)
    assert np.allclose(got, ref, atol=1e-10, rtol=1e-8)


def test_sigma_kernel_zero_in_2d(sgs_backend):
    # Rank-deficient (2D) alpha -> sigma_3 = 0 -> nu_t identically 0; the
    # macro emits 0 without computing G.
    rng = np.random.default_rng(4)
    alpha = rng.standard_normal((8, 2, 2))
    rho = rng.uniform(0.5, 2.0, 8)
    delta = rng.uniform(0.1, 1.0, 8)
    got = _run_sgs_kernel(sgs_backend, 'sigma', alpha, rho, delta,
                          c_sigma=1.35)
    assert np.allclose(got, 0.0)


def test_sigma_kernel_zero_for_solid_body_rotation(sgs_backend):
    # Pure rotation -> antisymmetric gradient -> singular values (s, s, 0),
    # so sigma_3 = 0 and D_sigma = 0.
    def skew(w):
        return np.array([[0.0, -w[2], w[1]],
                         [w[2], 0.0, -w[0]],
                         [-w[1], w[0], 0.0]])

    rng = np.random.default_rng(5)
    alpha = np.array([skew(rng.standard_normal(3)) for _ in range(6)])
    rho = np.ones(6)
    delta = np.full(6, 0.5)
    got = _run_sgs_kernel(sgs_backend, 'sigma', alpha, rho, delta,
                          c_sigma=1.35)
    assert np.allclose(got, 0.0, atol=1e-10)


def test_sigma_kernel_zero_for_axisymmetric_strain(sgs_backend):
    # Two equal singular values (sigma_1 = sigma_2) -> (sigma_1 - sigma_2)
    # factor vanishes -> D_sigma = 0.
    alpha = np.array([np.diag([2.0, 2.0, 1.0]),
                      np.diag([3.0, 3.0, 0.5])])
    rho = np.array([1.0, 1.0])
    delta = np.array([0.5, 0.5])
    got = _run_sgs_kernel(sgs_backend, 'sigma', alpha, rho, delta,
                          c_sigma=1.35)
    assert np.allclose(got, 0.0, atol=1e-10)


def test_sigma_kernel_zero_for_uniform_flow(sgs_backend):
    alpha = np.zeros((3, 3, 3))
    rho = np.array([1.0, 1.2, 0.8])
    delta = np.array([0.5, 0.5, 0.5])
    got = _run_sgs_kernel(sgs_backend, 'sigma', alpha, rho, delta,
                          c_sigma=1.35)
    assert np.allclose(got, 0.0)


# ===========================================================================
# Cross-rank Delta_e exchange (NavierStokesMPIInters): 2-rank mpiexec tests.
#
# Two complementary workers, each launched with mpiexec -n 2:
#   _mpi_delta_e_worker.py    -- replicates the Sendrecv pattern in isolation
#                                (routing, tags, ordering, buffer safety).
#   _mpi_mpiinters_worker.py  -- constructs a REAL NavierStokesMPIInters with
#                                an SGS model active, so the genuine one-shot
#                                exchange in its __init__ runs against real
#                                elements whose delta_e varies along the
#                                interface; checks neighbour blocks land in
#                                the correct order.
# Both skip cleanly if no MPI launcher is available.
# ===========================================================================

def _run_mpi_worker(worker_basename, label):
    launcher = shutil.which('mpiexec') or shutil.which('mpirun')
    if launcher is None:
        pytest.skip('no MPI launcher (mpiexec/mpirun) available')

    worker = os.path.join(os.path.dirname(__file__), worker_basename)
    cmd = [launcher, '-n', '2', '--oversubscribe', sys.executable, worker]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        pytest.fail(f'{label} worker timed out')
    except FileNotFoundError:
        pytest.skip(f'could not launch {launcher!r}')

    assert proc.returncode == 0, (
        f'{label} worker failed (rc={proc.returncode})\n'
        f'--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}'
    )
    # Both ranks must have reported success.
    assert proc.stdout.count('PASS') == 2, (
        f'{label}: expected 2 PASS lines, got:\n{proc.stdout}'
    )


def test_delta_e_mpi_exchange_two_ranks():
    _run_mpi_worker('_mpi_delta_e_worker.py', 'Delta_e Sendrecv')


def test_mpiinters_delta_e_ordering_two_ranks():
    _run_mpi_worker('_mpi_mpiinters_worker.py', 'MPIInters Delta_e ordering')


# ===========================================================================
# setup_matching_groups on a single regular hex (no backend, no real mesh).
#
# A NavierStokesElements object plus a hand-built Connectivity are enough to
# drive the centroid-based matching geometry directly.  The element box is
# axis-aligned with distinct edge lengths so each face exercises a different
# wall-normal extent.
# ===========================================================================

# Box edge lengths (distinct so h_wm differs per face axis).
_HEX_L = np.array([3.0, 2.0, 1.0])

# (fidx, wall-normal axis) for each hex face, from HexShape.faces normals:
# 0:(0,0,-1) 1:(0,-1,0) 2:(1,0,0) 3:(0,1,0) 4:(-1,0,0) 5:(0,0,1)
_HEX_FACES = [(0, 2), (1, 1), (2, 0), (3, 1), (4, 0), (5, 2)]


def _single_hex_matching(fidx):
    cfg = Inifile()
    cfg.set('solver', 'order', '3')
    cfg.set('solver-elements-hex', 'soln-pts', 'gauss-legendre')
    cfg.set('solver-interfaces-quad', 'flux-pts', 'gauss-legendre')
    cfg.set('constants', 'gamma', '1.4')

    # Regular axis-aligned hex box [0,Lx] x [0,Ly] x [0,Lz], vertices in the
    # shape's std_ele(1) ordering so they line up with the geometry basis.
    ref = HexShape.std_ele(1)
    phys = (ref + 1.0)/2.0*_HEX_L
    eles = NavierStokesElements(HexShape, phys[:, None, :], cfg)

    conn = Connectivity(np.array([0]), np.array([0]), {0: ('hex', fidx)})
    groups, n_total = setup_matching_groups(conn, {'hex': eles})
    return eles, groups, n_total


@pytest.mark.parametrize('fidx,axis', _HEX_FACES)
def test_setup_matching_groups_single_hex_geometry(fidx, axis):
    eles, groups, n_total = _single_hex_matching(fidx)
    nfp = eles.nfacefpts[fidx]

    # Exactly one (etype, fidx) group spanning all the face's flux points.
    assert len(groups) == 1
    assert n_total == nfp
    g = groups[0]
    assert g['etype'] == 'hex' and g['fidx'] == fidx
    assert g['h_wm'].shape == (nfp,)
    assert g['interp'].shape == (nfp, eles.nupts)

    # h_wm = half the wall-normal extent (centroid sits at mid-height), the
    # same for every fpt on a regular box.
    assert np.allclose(g['h_wm'], 0.5*_HEX_L[axis])

    # The matching point lies on the element mid-plane -> its reference
    # coordinate along the wall-normal axis is 0.
    assert np.allclose(g['xi_wm'][:, axis], 0.0, atol=1e-12)

    # Interpolation rows are a partition of unity.
    assert np.allclose(g['interp'].sum(axis=1), 1.0)


@pytest.mark.parametrize('fidx,axis', _HEX_FACES)
def test_setup_matching_groups_recovers_linear_field(fidx, axis):
    # End-to-end: a linear field f(x) = a.x + b sampled at the element's
    # solution points and contracted with the per-fpt interpolation row must
    # reproduce f at the physical matching point x_wm.  This ties together
    # the centroid, h_wm, Newton geomap inversion and interpolation vector.
    eles, groups, _ = _single_hex_matching(fidx)
    g = groups[0]

    a = np.array([0.7, -1.3, 2.1])
    b = 0.5
    upts_phys = eles.ploc_at_np('upts')[:, :, 0]      # (nupts, ndims)
    f_upts = upts_phys @ a + b

    # Physical matching point from the matching geometry (independent of the
    # stored interp): x_wm = x_fp - h_wm * n_outward.
    x_fp, = eles.get_ploc_for_inters(np.array([0]), fidx)
    pn, = eles.get_pnorms_for_inters(np.array([0]), fidx)
    n_hat = pn/np.linalg.norm(pn, axis=-1, keepdims=True)
    x_wm = x_fp - g['h_wm'][:, None]*n_hat

    f_pred = g['interp'] @ f_upts
    f_true = x_wm @ a + b
    assert np.allclose(f_pred, f_true, atol=1e-10)


def test_setup_matching_groups_fpt_count_mismatch_is_detectable():
    # Sanity: the per-group fpt total equals nfp for a single face, so a
    # consumer comparing n_total against an interface fpt count (as
    # _wmles_setup does) sees agreement here.
    eles, groups, n_total = _single_hex_matching(0)
    assert n_total == sum(g['n_fpts'] for g in groups)


# ===========================================================================
# Config validation via REAL wall-BC construction on the OpenMP backend.
#
# These build an actual NavierStokes wall-BC inters object (single regular
# hex element, set_backend + commit, hand-built Connectivity) so the genuine
# __init__ validation runs -- no mock of the validation logic.  For the
# WMLES BCs this also drives the real _wmles_setup matching on the single
# hex.  No SGS/flux kernel is ever compiled (kernels are lazy lambdas), so
# these run wherever the OpenMP backend can be instantiated.
# ===========================================================================

@pytest.fixture(scope='module')
def openmp_ok():
    try:
        get_backend('openmp', Inifile())
    except Exception as e:                                          # pragma: no cover
        pytest.skip(f'OpenMP backend unavailable: {e}')


def _ns_cfg(solver=None, bc=None, bctype='no-slp-adia-wall'):
    cfg = Inifile()
    cfg.set('solver', 'order', '3')
    cfg.set('solver-elements-hex', 'soln-pts', 'gauss-legendre')
    cfg.set('solver-interfaces-quad', 'flux-pts', 'gauss-legendre')
    cfg.set('solver-interfaces', 'riemann-solver', 'rusanov')
    cfg.set('solver-interfaces', 'ldg-beta', '0.5')
    cfg.set('solver-interfaces', 'ldg-tau', '0.1')
    cfg.set('constants', 'gamma', '1.4')
    cfg.set('constants', 'mu', '1e-3')
    cfg.set('constants', 'Pr', '0.71')
    for k, v in (solver or {}).items():
        cfg.set('solver', k, str(v))
    cfg.set('soln-bcs-wall', 'type', bctype)
    for k, v in (bc or {}).items():
        cfg.set('soln-bcs-wall', k, str(v))
    return cfg


def _construct_bc(bc_cls, cfg, *, set_backend_only=False):
    # Full real construction path: element -> set_backend -> commit -> BC.
    be = get_backend('openmp', cfg)
    ref = HexShape.std_ele(1)
    phys = (ref + 1.0)/2.0*_HEX_L
    eles = NavierStokesElements(HexShape, phys[:, None, :], cfg)

    # Element-side validation (sgs-model) fires inside set_backend.
    eles.set_backend(be, '0', 0)
    if set_backend_only:
        return None

    be.commit()
    lhs = Connectivity(np.array([0]), np.array([0]), {0: ('hex', 0)})
    return bc_cls(be, lhs, {'hex': eles}, 'soln-bcs-wall', cfg, MPI.COMM_WORLD)


# --- positive controls: valid configs build real objects ------------------

def test_wall_bc_constructs(openmp_ok):
    bc = _construct_bc(NavierStokesNoSlpAdiaWallBCInters, _ns_cfg())
    assert bc.type == 'no-slp-adia-wall'
    assert bc.ninterfpts == 16


def test_eq_ode_wall_constructs_adiabatic_and_isothermal(openmp_ok):
    bc_a = _construct_bc(NavierStokesEqODEWallBCInters,
                         _ns_cfg(bctype='eq-ode-wall',
                                 bc={'thermal-bc': 'adiabatic'}))
    assert bc_a.type == 'eq-ode-wall'
    bc_i = _construct_bc(NavierStokesEqODEWallBCInters,
                         _ns_cfg(bctype='eq-ode-wall',
                                 bc={'thermal-bc': 'isothermal', 'Tw': '300'}))
    assert bc_i._tplargs['thermal_bc'] == 'isothermal'


def test_alg_wall_constructs(openmp_ok):
    for law in ('log-law', 'spalding', 'reichardt'):
        bc = _construct_bc(NavierStokesAlgWallBCInters,
                           _ns_cfg(bctype='alg-wall', bc={'law': law}))
        assert bc._tplargs['alg_law'] == law


# --- sgs-model validation (duplicated in inters + elements) ---------------

def test_sgs_model_invalid_raises_in_inters(openmp_ok):
    with pytest.raises(ValueError, match='Invalid sgs-model'):
        _construct_bc(NavierStokesNoSlpAdiaWallBCInters,
                      _ns_cfg(solver={'sgs-model': 'bogus'}))


def test_sgs_model_invalid_raises_in_elements(openmp_ok):
    with pytest.raises(ValueError, match='Invalid sgs-model'):
        _construct_bc(NavierStokesNoSlpAdiaWallBCInters,
                      _ns_cfg(solver={'sgs-model': 'bogus'}),
                      set_backend_only=True)


@pytest.mark.parametrize('model', ['none', 'vreman', 'sigma'])
def test_sgs_model_valid_constructs(openmp_ok, model):
    bc = _construct_bc(NavierStokesNoSlpAdiaWallBCInters,
                       _ns_cfg(solver={'sgs-model': model}))
    assert bc.type == 'no-slp-adia-wall'


def test_wall_bc_skips_face_sgs_even_when_enabled(openmp_ok):
    # Design invariant: wall BCs never include the SGS contribution, even
    # with a model configured and sgs-include-faces = yes.  The BC strips
    # sgs_model to 'none' in its tplargs (uses_face_sgs = False).
    bc = _construct_bc(NavierStokesNoSlpAdiaWallBCInters,
                       _ns_cfg(solver={'sgs-model': 'vreman',
                                       'sgs-include-faces': 'yes'}))
    assert bc.uses_face_sgs is False
    assert bc._tplargs['sgs_model'] == 'none'


# --- eq-ode-wall option validation ----------------------------------------

@pytest.mark.parametrize('opt,val', [
    ('thermal-bc', 'bogus'),
    ('mu-t-model', 'bogus'),
    ('damping', 'bogus'),
    ('yp-scaling', 'bogus'),
    ('mu-model', 'bogus'),
])
def test_eq_ode_wall_invalid_option_raises(openmp_ok, opt, val):
    bc = {'thermal-bc': 'adiabatic', opt: val}
    with pytest.raises(ValueError, match=opt):
        _construct_bc(NavierStokesEqODEWallBCInters,
                      _ns_cfg(bctype='eq-ode-wall', bc=bc))


# --- alg-wall option validation -------------------------------------------

def test_alg_wall_invalid_law_raises(openmp_ok):
    with pytest.raises(ValueError, match='law'):
        _construct_bc(NavierStokesAlgWallBCInters,
                      _ns_cfg(bctype='alg-wall', bc={'law': 'bogus'}))


def test_alg_wall_isothermal_not_implemented(openmp_ok):
    with pytest.raises(NotImplementedError, match='adiabatic'):
        _construct_bc(NavierStokesAlgWallBCInters,
                      _ns_cfg(bctype='alg-wall',
                              bc={'thermal-bc': 'isothermal'}))


# ===========================================================================
# Real flux-kernel rendering with SGS.
#
# The SGS / wall-model work added a conditional `delta_e` argument to the
# tflux / intcflux / mpicflux / bccflux kernels.  A `% if` cannot live inside
# a mako tag's argument list, so the body is factored into a def and the tag
# emitted in two conditional forms.  These tests render the REAL kernels via
# the backend (no mesh, no C compilation) for every sgs-model -- guarding the
# whole "kernel doesn't even lex/render" failure class, which the macro-level
# tests above do not exercise.
# ===========================================================================

# Per-kernel tplargs on top of the common set (ndims/nvars/visc_corr/
# shock_capturing/c/sgs_model are shared).
_FLUX_KERNELS = {
    'tflux': dict(nverts=8, ktype=''),
    'intcflux': dict(rsolver='rusanov'),
    'mpicflux': dict(rsolver='rusanov'),
    'bccflux': dict(rsolver='rusanov', bctype='no-slp-adia-wall',
                    bccfluxstate='ghost-imperm'),
}


def _render_flux_kernel(kern, sgs_model):
    be = get_backend('openmp', Inifile())
    be.pointwise.register(f'pyfr.solvers.navstokes.kernels.{kern}')
    c = {'gamma': 1.4, 'mu': 1e-3, 'Pr': 0.72, 'Prt': 0.9,
         'ldg-beta': 0.5, 'ldg-tau': 0.1}
    tpl = dict(ndims=3, nvars=5, sgs_model=sgs_model, c=c,
               visc_corr='none', shock_capturing='none',
               **_FLUX_KERNELS[kern])
    if sgs_model == 'vreman':
        tpl['c_vreman'] = 0.07
    elif sgs_model == 'sigma':
        tpl['c_sigma'] = 1.35
    src, ndim, argn, argt = be.pointwise._render_kernel(
        kern, f'pyfr.solvers.navstokes.kernels.{kern}', {}, tpl)
    return argn


@pytest.mark.parametrize('kern', list(_FLUX_KERNELS))
@pytest.mark.parametrize('sgs_model', ['none', 'vreman', 'sigma'])
def test_navstokes_flux_kernels_render(openmp_ok, kern, sgs_model):
    # The kernel must lex + render for every SGS model (this is exactly what
    # failed when a `% if` was placed inside the tag's argument list).
    argn = _render_flux_kernel(kern, sgs_model)
    has_delta = any('delta_e' in a for a in argn)
    # delta_e appears iff an SGS model is active.
    assert has_delta == (sgs_model != 'none'), (kern, sgs_model, argn)


def test_navstokes_flux_kernels_baseline_has_no_sgs_args(openmp_ok):
    # ILES (sgs-model = none) must render identically to stock PyFR: no
    # delta_e arguments anywhere.
    for kern in _FLUX_KERNELS:
        argn = _render_flux_kernel(kern, 'none')
        assert not any('delta_e' in a for a in argn), (kern, argn)
