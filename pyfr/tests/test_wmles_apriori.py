r"""A priori tests for the WMLES wall models (Phase 2).

Standalone -- no PyFR solver run.  Each wall model is reimplemented in numpy
(mirroring its mako line for line) and validated two ways:

  * Synthetic recovery: a velocity profile generated FROM a given wall law
    at a known friction velocity u_tau must be inverted by the model back to
    that same u_tau (the solver finds the exact root of the law it solves).
  * Kernel cross-check: the REAL compiled mako wall-model kernel is run on
    the same inputs and must agree with the numpy reference (so the numpy
    reference provably matches the code that runs in PyFR).  The kernel
    stores the converged u_tau in its warm_state extern, which we read back.

The compiled-kernel tests skip automatically if no OpenMP-capable C compiler
is available.
"""
import numpy as np
import pytest

from pyfr.backends import get_backend
from pyfr.inifile import Inifile


# ---------------------------------------------------------------------------
# numpy reference: algebraic equilibrium wall model (alg-wall.mako)
# ---------------------------------------------------------------------------

def alg_wall_utau(U_LES, h_wm, nu_w, *, law, kappa=0.41, B=5.0, C=7.3,
                  B1=11.0, B2=3.0, n_iter=8, rel_tol=1e-3, warm=None):
    # Faithful numpy mirror of the Newton solve in alg-wall.mako: same f,
    # f', initial guess, positivity clamp and per-fpt relative-update break.
    U_LES = np.atleast_1d(np.asarray(U_LES, dtype=float))
    h_wm = np.broadcast_to(np.asarray(h_wm, float), U_LES.shape).astype(float)
    nu_w = np.broadcast_to(np.asarray(nu_w, float), U_LES.shape).astype(float)

    heuristic = np.maximum(0.05*U_LES, nu_w/h_wm)
    if warm is None:
        u_tau = heuristic.copy()
    else:
        warm = np.broadcast_to(np.asarray(warm, float), U_LES.shape)
        u_tau = np.where(warm > 0.0, warm, heuristic)

    expmkB = np.exp(-kappa*B)
    active = np.ones(U_LES.shape, dtype=bool)

    for _ in range(n_iter):
        Up = U_LES/u_tau
        yp = h_wm*u_tau/nu_w

        if law == 'log-law':
            f = Up - (1.0/kappa)*np.log(yp) - B
            fp = -U_LES/u_tau**2 - 1.0/(kappa*u_tau)
        elif law == 'spalding':
            X = kappa*Up
            eX = np.exp(X)
            bracket = eX - 1.0 - X - 0.5*X**2 - X**3/6.0
            f = yp - Up - expmkB*bracket
            d_brk = eX - 1.0 - X - 0.5*X**2
            fp = h_wm/nu_w + U_LES/u_tau**2 + expmkB*X*d_brk/u_tau
        elif law == 'reichardt':
            opk = 1.0 + kappa*yp
            e1 = np.exp(-yp/B1)
            e2 = np.exp(-yp/B2)
            f = Up - (1.0/kappa)*np.log(opk) - C*(1.0 - e1 - (yp/B1)*e2)
            dUp_dyp = 1.0/opk + C*(e1/B1 + e2*(yp/(B1*B2) - 1.0/B1))
            fp = -U_LES/u_tau**2 - dUp_dyp*h_wm/nu_w
        else:
            raise ValueError(f'unknown law {law!r}')

        du = f/fp
        u_new = np.maximum(u_tau - du, 1e-14)
        u_tau = np.where(active, u_new, u_tau)
        active &= np.abs(du) >= rel_tol*u_tau

    return u_tau


# --- synthetic profile generators (forward, no inversion) -----------------

def _loglaw_Uplus(yp, kappa, B):
    return (1.0/kappa)*np.log(yp) + B


def _reichardt_Uplus(yp, kappa, C, B1, B2):
    return ((1.0/kappa)*np.log(1.0 + kappa*yp)
            + C*(1.0 - np.exp(-yp/B1) - (yp/B1)*np.exp(-yp/B2)))


def _spalding_yplus(Uplus, kappa, B):
    X = kappa*Uplus
    return Uplus + np.exp(-kappa*B)*(np.exp(X) - 1.0 - X - 0.5*X**2
                                     - X**3/6.0)


# ===========================================================================
# Synthetic recovery: model inverts a profile back to the generating u_tau.
# ===========================================================================

# Friction velocities and a wall kinematic viscosity spanning a few decades
# of y+ at the match point.
_UTAU_TRUE = np.array([0.05, 0.2, 0.5, 1.0, 3.0])
_NU_W = 1.5e-5


@pytest.mark.parametrize('law', ['log-law', 'reichardt'])
def test_alg_wall_recovers_utau_loglaw_reichardt(law):
    kappa, B, C, B1, B2 = 0.41, 5.0, 7.3, 11.0, 3.0
    # Match-point y+ in the log layer where these laws are well posed.
    yp = np.array([50.0, 100.0, 300.0, 600.0, 1000.0])
    h_wm = yp*_NU_W/_UTAU_TRUE

    if law == 'log-law':
        Uplus = _loglaw_Uplus(yp, kappa, B)
    else:
        Uplus = _reichardt_Uplus(yp, kappa, C, B1, B2)
    U_LES = _UTAU_TRUE*Uplus

    # Tight tolerance / many iterations: the solve must hit the exact root.
    u_tau = alg_wall_utau(U_LES, h_wm, _NU_W, law=law,
                          n_iter=60, rel_tol=1e-13)
    assert np.allclose(u_tau, _UTAU_TRUE, rtol=1e-9)


def test_alg_wall_recovers_utau_spalding():
    kappa, B = 0.41, 5.0
    # Pick U+ in the log region, derive the consistent y+ from Spalding.
    Uplus = np.array([12.0, 16.0, 20.0, 24.0, 28.0])
    yp = _spalding_yplus(Uplus, kappa, B)
    h_wm = yp*_NU_W/_UTAU_TRUE
    U_LES = _UTAU_TRUE*Uplus

    u_tau = alg_wall_utau(U_LES, h_wm, _NU_W, law='spalding',
                          n_iter=60, rel_tol=1e-13)
    assert np.allclose(u_tau, _UTAU_TRUE, rtol=1e-9)


def test_alg_wall_default_iters_are_accurate():
    # With the production defaults (n_iter=8, rel_tol=1e-3) and a cold start,
    # the recovered u_tau is still within ~0.5% of truth.
    kappa, B = 0.41, 5.0
    yp = np.array([50.0, 100.0, 300.0, 600.0, 1000.0])
    h_wm = yp*_NU_W/_UTAU_TRUE
    U_LES = _UTAU_TRUE*_loglaw_Uplus(yp, kappa, B)

    u_tau = alg_wall_utau(U_LES, h_wm, _NU_W, law='log-law')
    assert np.allclose(u_tau, _UTAU_TRUE, rtol=5e-3)


# ===========================================================================
# Physical sanity / sensitivity.
# ===========================================================================

def test_alg_wall_laws_agree_at_high_yplus():
    # Deep in the log layer all three laws should infer nearly the same
    # u_tau from the same resolved velocity.
    U_LES = 2.0
    h_wm = 0.02
    nu_w = _NU_W
    us = {law: float(alg_wall_utau(U_LES, h_wm, nu_w, law=law,
                                   n_iter=60, rel_tol=1e-13)[0])
          for law in ('log-law', 'spalding', 'reichardt')}
    umean = np.mean(list(us.values()))
    for law, u in us.items():
        assert abs(u - umean)/umean < 0.05, (law, us)


def test_alg_wall_utau_monotonic_in_velocity():
    # Larger resolved velocity at the same match point -> larger u_tau.
    U_LES = np.array([0.5, 1.0, 2.0, 4.0, 8.0])
    u_tau = alg_wall_utau(U_LES, 0.02, _NU_W, law='log-law',
                          n_iter=60, rel_tol=1e-13)
    assert np.all(np.diff(u_tau) > 0)


# ===========================================================================
# Kernel cross-check: the REAL alg-wall mako kernel vs the numpy reference.
#
# A thin kernel (pyfr/tests/alg_wall_test_kernel.mako) expands the actual
# bc_common_flux_state macro with ul/nl as direct args and u_match/h_wm/
# warm_state as externs.  The macro writes the converged u_tau into
# warm_state[0], which we read back.  Skips if no OpenMP compiler is found.
# ===========================================================================

_CC_CANDIDATES = [None, 'gcc-15', 'gcc-14', 'gcc-13', 'gcc-12', 'gcc']
_NDIMS, _NVARS = 3, 5
_NPRIMS = _NDIMS + 2


def _run_alg_wall_kernel(be, law, U_LES, h_wm, rho, mu, *, kappa=0.41, B=5.0,
                         C=7.3, B1=11.0, B2=3.0, n_iter=50, rel_tol=1e-12,
                         warm=None):
    U_LES = np.atleast_1d(np.asarray(U_LES, float))
    h_wm = np.broadcast_to(np.asarray(h_wm, float), U_LES.shape).astype(float)
    n = U_LES.size

    c = {'gamma': 1.4, 'alg_mu': mu, 'alg_kappa': kappa, 'alg_B': B,
         'alg_C': C, 'alg_B1': B1, 'alg_B2': B2, 'alg_n_iter': n_iter,
         'alg_rel_tol': rel_tol, 'alg_expmkB': float(np.exp(-kappa*B))}
    tpl = dict(ndims=_NDIMS, nvars=_NVARS, rsolver='rusanov', alg_law=law, c=c)

    # ul: rho in slot 0, a positive total energy in the last slot.
    ul = np.zeros((_NVARS, n)); ul[0] = rho; ul[-1] = 2.5
    # Wall normal +z; resolved velocity purely tangential (x) of magnitude U.
    nl = np.zeros((_NDIMS, n)); nl[2] = 1.0
    um = np.zeros((_NPRIMS, n)); um[0] = rho; um[1] = U_LES
    ws = np.zeros((1, n)) if warm is None else \
        np.broadcast_to(np.asarray(warm, float), (1, n)).astype(float)

    mats = {k: be.matrix(v.shape, initval=v) for k, v in
            dict(ul=ul, nl=nl, um=um, hw=h_wm[None, :], ws=ws).items()}

    k = be.kernel('alg_wall_test_kernel', tplargs=tpl, dims=[n],
                  extrns={'u_match': f'in fpdtype_t[{_NPRIMS}]',
                          'h_wm': 'in fpdtype_t',
                          'warm_state': 'inout fpdtype_t[1]'},
                  ul=mats['ul'], nl=mats['nl'], u_match=mats['um'],
                  h_wm=mats['hw'], warm_state=mats['ws'])
    k.run()
    return mats['ws'].get().ravel()


def _make_wall_backend():
    for cc in _CC_CANDIDATES:
        cfg = Inifile()
        if cc is not None:
            cfg.set('backend-openmp', 'cc', cc)
        try:
            be = get_backend('openmp', cfg)
            be.pointwise.register('pyfr.tests.alg_wall_test_kernel')
            _run_alg_wall_kernel(be, 'log-law', [1.0], [0.01], 1.0, 1.5e-5)
            return be
        except Exception:
            continue
    return None


@pytest.fixture(scope='module')
def wall_backend():
    be = _make_wall_backend()
    if be is None:
        pytest.skip('no OpenMP-capable C compiler for wall-model kernel tests')
    return be


def _synthetic_profile(law, utrue, nu_w, kappa=0.41, B=5.0, C=7.3, B1=11.0,
                       B2=3.0):
    if law == 'spalding':
        Uplus = np.array([12.0, 16.0, 20.0, 24.0, 28.0])
        yp = _spalding_yplus(Uplus, kappa, B)
    else:
        yp = np.array([50.0, 100.0, 300.0, 600.0, 1000.0])
        Uplus = (_loglaw_Uplus(yp, kappa, B) if law == 'log-law'
                 else _reichardt_Uplus(yp, kappa, C, B1, B2))
    return utrue*Uplus, yp*nu_w/utrue


@pytest.mark.parametrize('law', ['log-law', 'spalding', 'reichardt'])
def test_alg_wall_kernel_matches_reference(wall_backend, law):
    utrue = _UTAU_TRUE
    nu_w = _NU_W
    rho = 1.2
    mu = nu_w*rho
    U_LES, h_wm = _synthetic_profile(law, utrue, nu_w)

    u_kernel = _run_alg_wall_kernel(wall_backend, law, U_LES, h_wm, rho, mu)
    u_ref = alg_wall_utau(U_LES, h_wm, nu_w, law=law, n_iter=50, rel_tol=1e-12)

    # Real mako == numpy reference, and both recover the generating u_tau.
    assert np.allclose(u_kernel, u_ref, rtol=1e-6)
    assert np.allclose(u_kernel, utrue, rtol=1e-6)


def test_alg_wall_kernel_uses_warm_start(wall_backend):
    # With zero Newton iterations the kernel returns its initial guess: the
    # warm-start value when warm_state > 0, else the cold heuristic.
    U_LES = np.array([1.0, 2.0])
    h_wm = np.array([0.02, 0.02])
    rho, mu = 1.2, 1.5e-5*1.2

    warm = np.array([0.123, 0.456])
    u_warm = _run_alg_wall_kernel(wall_backend, 'log-law', U_LES, h_wm, rho,
                                  mu, n_iter=0, warm=warm)
    assert np.allclose(u_warm, warm)

    u_cold = _run_alg_wall_kernel(wall_backend, 'log-law', U_LES, h_wm, rho,
                                  mu, n_iter=0)
    heuristic = np.maximum(0.05*U_LES, (mu/rho)/h_wm)
    assert np.allclose(u_cold, heuristic)


# ===========================================================================
# numpy reference: integral equilibrium GQWM (gq-ode-wall.mako)
#
# Solves  U_LES = \int_0^h_wm du/dy dy  with the Van-Driest mixing-length
# closure  du/dy = 2 u_tau^2 / (nu (1 + sqrt(1 + 4 lm^2))),
#   lm = kappa y+ (1 - exp(-y+/A+)),  y+ = y u_tau / nu,
# by GL quadrature on an exponential mapping of [0, h_wm], secant on u_tau.
# ===========================================================================

from numpy.polynomial.legendre import leggauss                    # noqa: E402


def gq_nodes(h_wm, n_quad):
    # Per-fpt mapped GL nodes/weights, mirroring the construction in
    # NavierStokesGQODEWallBCInters.__init__ (weights absorb dy/dxi).
    xi, w = leggauss(n_quad)
    denom = np.exp(2.0) - 1.0
    exp_xi = np.exp(xi + 1.0)
    h_wm = np.atleast_1d(np.asarray(h_wm, float))
    y_q = h_wm[None, :]*(exp_xi[:, None] - 1.0)/denom
    w_q = w[:, None]*h_wm[None, :]*exp_xi[:, None]/denom
    return y_q, w_q


def gq_integral(u_tau, h_wm, nu_w, *, kappa=0.41, Aplus=26.0, n_quad=15):
    # The quadrature estimate of U(h_wm) for a given u_tau.
    u_tau = np.atleast_1d(np.asarray(u_tau, float))
    nu_w = np.broadcast_to(np.asarray(nu_w, float), u_tau.shape).astype(float)
    y_q, w_q = gq_nodes(h_wm, n_quad)
    yp = y_q*u_tau[None, :]/nu_w[None, :]
    lm = kappa*yp*(1.0 - np.exp(-yp/Aplus))
    integ = 2.0*u_tau[None, :]**2/(nu_w[None, :]*(1.0 + np.sqrt(1.0 + 4.0*lm**2)))
    return (w_q*integ).sum(axis=0)


def gq_ode_wall_utau(U_LES, h_wm, nu_w, *, kappa=0.41, Aplus=26.0, n_quad=15,
                     n_iter=6, rel_tol=1e-3, warm=None):
    # Faithful numpy mirror of the secant solve in gq-ode-wall.mako.
    U_LES = np.atleast_1d(np.asarray(U_LES, float))
    h_wm = np.broadcast_to(np.asarray(h_wm, float), U_LES.shape).astype(float)
    nu_w = np.broadcast_to(np.asarray(nu_w, float), U_LES.shape).astype(float)

    def F(u):
        return gq_integral(u, h_wm, nu_w, kappa=kappa, Aplus=Aplus,
                           n_quad=n_quad) - U_LES

    heuristic = np.maximum(0.05*U_LES, nu_w/h_wm)
    if warm is None:
        u0 = heuristic.copy()
    else:
        warm = np.broadcast_to(np.asarray(warm, float), U_LES.shape)
        u0 = np.where(warm > 0.0, warm, heuristic)
    u1 = 1.1*u0
    f0, f1 = F(u0), F(u1)
    active = np.ones(U_LES.shape, dtype=bool)

    for _ in range(n_iter):
        df = f1 - f0
        safe = np.where(np.abs(df) > 1e-15, df, 1e-15)
        new = u1 - f1*(u1 - u0)/safe
        du = new - u1

        u0 = np.where(active, u1, u0)
        f0 = np.where(active, f1, f0)
        u1 = np.where(active, np.maximum(new, 1e-14), u1)
        f1 = np.where(active, F(u1), f1)
        active &= np.abs(du) >= rel_tol*u1

    return np.maximum(u1, 0.0)


# ===========================================================================
# GQWM synthetic recovery + physical sanity (numpy reference).
# ===========================================================================

def test_gq_ode_wall_self_consistent_recovery():
    # Generate U_LES from the model's own quadrature at u_tau_true, then the
    # secant must return u_tau_true (exact root of the discrete equation).
    utrue = _UTAU_TRUE
    yp_h = np.array([50.0, 100.0, 300.0, 600.0, 1000.0])
    h_wm = yp_h*_NU_W/utrue
    U_LES = gq_integral(utrue, h_wm, _NU_W)

    u_tau = gq_ode_wall_utau(U_LES, h_wm, _NU_W, n_iter=80, rel_tol=1e-13)
    assert np.allclose(u_tau, utrue, rtol=1e-8)


def test_gq_quadrature_integrates_jacobian_exactly():
    # sum_qi w_qi = \int_0^h_wm dy = h_wm: the exponential-map weights must
    # reproduce the element height (spectral accuracy on the smooth map).
    h_wm = np.array([0.5, 1.0, 3.7])
    for n_quad in (8, 15, 30):
        _, w_q = gq_nodes(h_wm, n_quad)
        assert np.allclose(w_q.sum(axis=0), h_wm, rtol=1e-12)


def test_gq_ode_wall_laminar_limit():
    # With kappa = 0 the mixing length vanishes and du/dy = u_tau^2/nu, so
    # U_LES = u_tau^2 h_wm / nu  ->  u_tau = sqrt(U_LES nu / h_wm).  Feeding
    # the analytic laminar U_LES must recover u_tau_true (weights integrate
    # the constant exactly).
    utrue = _UTAU_TRUE
    h_wm = np.array([1e-4, 2e-4, 5e-4, 1e-3, 2e-3])
    U_LES = utrue**2*h_wm/_NU_W
    u_tau = gq_ode_wall_utau(U_LES, h_wm, _NU_W, kappa=0.0, n_iter=80,
                             rel_tol=1e-13)
    assert np.allclose(u_tau, utrue, rtol=1e-9)


def test_gq_ode_wall_monotonic_in_velocity():
    U_LES = np.array([0.5, 1.0, 2.0, 4.0, 8.0])
    u_tau = gq_ode_wall_utau(U_LES, 0.02, _NU_W, n_iter=80, rel_tol=1e-13)
    assert np.all(np.diff(u_tau) > 0)


def test_gq_ode_wall_log_layer_slope():
    # Deep in the log layer the inferred profile U+(y+) has slope ~ 1/kappa.
    utau = 0.5
    yp = np.array([500.0, 2000.0])
    h = yp*_NU_W/utau
    Uplus = gq_integral(utau, h, _NU_W)/utau
    slope = (Uplus[1] - Uplus[0])/np.log(yp[1]/yp[0])
    assert abs(slope - 1.0/0.41)/(1.0/0.41) < 0.12


# ===========================================================================
# Kernel cross-check: the REAL gq-ode-wall mako kernel vs the numpy reference.
# Extra externs gq_y/gq_w (per-fpt mapped quadrature nodes/weights) are built
# exactly as the BC __init__ does, via gq_nodes().
# ===========================================================================

def _run_gq_wall_kernel(be, U_LES, h_wm, rho, mu, *, kappa=0.41, Aplus=26.0,
                        n_quad=15, n_iter=80, rel_tol=1e-13, warm=None):
    be.pointwise.register('pyfr.tests.gq_wall_test_kernel')

    U_LES = np.atleast_1d(np.asarray(U_LES, float))
    h_wm = np.broadcast_to(np.asarray(h_wm, float), U_LES.shape).astype(float)
    n = U_LES.size

    c = {'gamma': 1.4, 'gq_mu': mu, 'gq_kappa': kappa, 'gq_Aplus': Aplus,
         'gq_nquad': n_quad, 'gq_n_iter': n_iter, 'gq_rel_tol': rel_tol}
    tpl = dict(ndims=_NDIMS, nvars=_NVARS, rsolver='rusanov', c=c)

    ul = np.zeros((_NVARS, n)); ul[0] = rho; ul[-1] = 2.5
    nl = np.zeros((_NDIMS, n)); nl[2] = 1.0
    um = np.zeros((_NPRIMS, n)); um[0] = rho; um[1] = U_LES
    ws = np.zeros((1, n)) if warm is None else \
        np.broadcast_to(np.asarray(warm, float), (1, n)).astype(float)
    y_q, w_q = gq_nodes(h_wm, n_quad)

    mats = {k: be.matrix(v.shape, initval=v) for k, v in
            dict(ul=ul, nl=nl, um=um, hw=h_wm[None, :], ws=ws,
                 gy=y_q, gw=w_q).items()}

    k = be.kernel('gq_wall_test_kernel', tplargs=tpl, dims=[n],
                  extrns={'u_match': f'in fpdtype_t[{_NPRIMS}]',
                          'h_wm': 'in fpdtype_t',
                          'warm_state': 'inout fpdtype_t[1]',
                          'gq_y': f'in fpdtype_t[{n_quad}]',
                          'gq_w': f'in fpdtype_t[{n_quad}]'},
                  ul=mats['ul'], nl=mats['nl'], u_match=mats['um'],
                  h_wm=mats['hw'], warm_state=mats['ws'],
                  gq_y=mats['gy'], gq_w=mats['gw'])
    k.run()
    return mats['ws'].get().ravel()


def test_gq_ode_wall_kernel_matches_reference(wall_backend):
    utrue = _UTAU_TRUE
    nu_w = _NU_W
    rho = 1.2
    mu = nu_w*rho
    yp_h = np.array([50.0, 100.0, 300.0, 600.0, 1000.0])
    h_wm = yp_h*nu_w/utrue
    U_LES = gq_integral(utrue, h_wm, nu_w)

    u_kernel = _run_gq_wall_kernel(wall_backend, U_LES, h_wm, rho, mu)
    u_ref = gq_ode_wall_utau(U_LES, h_wm, nu_w, n_iter=80, rel_tol=1e-13)

    assert np.allclose(u_kernel, u_ref, rtol=1e-6)
    assert np.allclose(u_kernel, utrue, rtol=1e-6)


# ===========================================================================
# numpy reference: equilibrium 2-equation ODE wall model (eq-ode-wall.mako)
#
# Only the PRIMARY ODE (du/dy, dT/dy) is mirrored -- the converged shooting
# root (tau_w, p2) is defined entirely by the primary system reaching
# (U_LES, T_m) at h_wm; the mako's sensitivity equations only drive Newton
# convergence speed, not the root.  Here we integrate with a fine fixed-step
# RK4 and shoot with a 2D finite-difference Newton.  All compositional
# branches (mu_t-model x damping x y+-scaling x viscosity-law x thermal-bc)
# are implemented, matching the mako's eq_compute_rhs.
# ===========================================================================

def _eq_mu_local(T, P):
    if P['mu_model'] == 'sutherland':
        Tr = T/P['Tref']
        return P['mu_ref']*Tr*np.sqrt(Tr)*(P['Tref'] + P['S'])/(T + P['S'])
    return P['mu_c']


def _eq_rhs(y, u, T, tau_w, q_w, rho_w, mu_w, P):
    g, cp = P['gamma'], P['cp']
    T_safe = max(T, 1e-3)
    mu_l = _eq_mu_local(T_safe, P)
    rho_l = g*P['p_m']/((g - 1)*cp*T_safe)
    tau_safe = max(tau_w, 1e-12)
    utau = np.sqrt(tau_safe/rho_w)

    sc = P['yp_scaling']
    if sc == 'wall':
        yp = rho_w*utau*y/mu_w
    elif sc == 'semi-local':
        yp = np.sqrt(rho_l*rho_w)*utau*y/mu_l
    else:
        ypw = rho_w*utau*y/mu_w
        ypl = rho_l*utau*y/mu_l
        yps = np.sqrt(rho_l*rho_w)*utau*y/mu_l
        yp = min(0.5*(ypw + yps), 0.5*(ypl + yps))

    dmp = P['damping']
    if dmp == 'van-driest':
        D = (1.0 - np.exp(-yp/P['Aplus']))**2
    elif dmp == 'piomelli':
        D = 1.0 - np.exp(-(yp/P['Aplus'])**3)
    else:
        D = 1.0

    Cv13 = P['Cv1']**3
    if P['mu_t_model'] == 'johnson-king':
        muhat = P['kappa']*y*np.sqrt(rho_l*tau_safe)
        if dmp == 'spalart-allmaras':
            D = muhat**3/(muhat**3 + Cv13*mu_l**3 + 1e-12)
        mu_t = muhat*D
        dudy = tau_w/(mu_l + mu_t)
    else:
        alphaP = rho_l*P['kappa']**2*y*y
        if dmp == 'spalart-allmaras':
            dudy, dudy_prev, res_prev = tau_w/mu_l, 0.0, 0.0
            for it in range(3):
                muhat = alphaP*abs(dudy)
                D = muhat**3/(muhat**3 + Cv13*mu_l**3 + 1e-12)
                mu_t = muhat*D
                T_dudy = tau_w/(mu_l + mu_t)
                res = T_dudy - dudy
                dres = res - res_prev
                new = ((res*dudy_prev - res_prev*dudy)/dres
                       if (it > 0 and abs(dres) > 1e-12) else T_dudy)
                if new <= 0.0:
                    new = T_dudy
                dudy_prev, res_prev, dudy = dudy, res, new
        else:
            a = alphaP*D
            disc = mu_l*mu_l + 4.0*a*tau_w
            sq = np.sqrt(max(disc, 0.0))
            dudy = (-mu_l + sq)/(2.0*a) if a > 1e-12 else tau_w/mu_l
            mu_t = alphaP*D*abs(dudy)

    alpha_T = cp*(mu_l/P['Pr'] + mu_t/P['Prt'])
    dTdy = -(q_w + u*tau_w)/alpha_T
    return dudy, dTdy


def _eq_integrate(p1, p2, P, h_wm, n_steps=4000):
    g, cp = P['gamma'], P['cp']
    if P['thermal_bc'] == 'isothermal':
        Tw, q_w = P['eq_Tw'], -p2
    else:
        Tw, q_w = max(p2, 1e-3), 0.0
    rho_w = g*P['p_m']/((g - 1)*cp*Tw)
    mu_w = _eq_mu_local(Tw, P)
    tau_w = p1
    u, T, h = 0.0, Tw, h_wm/n_steps
    for i in range(n_steps):
        y = i*h
        k1u, k1T = _eq_rhs(y, u, T, tau_w, q_w, rho_w, mu_w, P)
        k2u, k2T = _eq_rhs(y + 0.5*h, u + 0.5*h*k1u, T + 0.5*h*k1T,
                           tau_w, q_w, rho_w, mu_w, P)
        k3u, k3T = _eq_rhs(y + 0.5*h, u + 0.5*h*k2u, T + 0.5*h*k2T,
                           tau_w, q_w, rho_w, mu_w, P)
        k4u, k4T = _eq_rhs(y + h, u + h*k3u, T + h*k3T,
                           tau_w, q_w, rho_w, mu_w, P)
        u += h*(k1u + 2*k2u + 2*k3u + k4u)/6.0
        T += h*(k1T + 2*k2T + 2*k3T + k4T)/6.0
    return u, T


def eq_ode_wall_solve(U_LES, T_m, P, h_wm, n_newton=60, tol=1e-12):
    g, cp = P['gamma'], P['cp']
    rho_m = g*P['p_m']/((g - 1)*cp*T_m)
    if P['thermal_bc'] == 'isothermal':
        rho_w0 = g*P['p_m']/((g - 1)*cp*P['eq_Tw'])
        p2 = 0.0
    else:
        rho_w0 = rho_m
        p2 = T_m
    utau0 = max(0.05*U_LES, P['mu_c']/(rho_w0*h_wm))
    p1 = rho_w0*utau0*utau0

    for _ in range(n_newton):
        u0, T0 = _eq_integrate(p1, p2, P, h_wm)
        r = np.array([u0 - U_LES, T0 - T_m])
        if np.hypot(r[0]/max(abs(U_LES), 1e-14),
                    r[1]/max(abs(T_m), 1e-14)) < tol:
            break
        dp1 = max(1e-6*abs(p1), 1e-9)
        dp2 = max(1e-6*abs(p2), 1e-6)
        ua, Ta = _eq_integrate(p1 + dp1, p2, P, h_wm)
        ub, Tb = _eq_integrate(p1, p2 + dp2, P, h_wm)
        J = np.array([[(ua - u0)/dp1, (ub - u0)/dp2],
                      [(Ta - T0)/dp1, (Tb - T0)/dp2]])
        d = np.linalg.solve(J, -r)
        p1 += d[0]
        p2 += d[1]
    return p1, p2


# Air-like constants shared by the eq-ode reference and kernel tests.
_EQ_K = dict(gamma=1.4, cp=1005.0, Pr=0.71, Prt=0.9, mu=1.8e-5, kappa=0.41,
             Cv1=7.1, mu_ref=1.716e-5, Tref=273.15, S=110.4)
_EQ_PM = 1.0*(_EQ_K['cp']*(_EQ_K['gamma'] - 1)/_EQ_K['gamma'])*300.0  # rho=1,T=300
_EQ_HWM = 1e-3

# Representative configurations covering every branch value at least once.
# Each: (id, branch-opts, generating tau_w, generating p2).
#   isothermal: p2 = -q_w   |   adiabatic: p2 = T_w
_EQ_CONFIGS = [
    ('iso-jk-vd-wall-const',
     dict(thermal_bc='isothermal', mu_t_model='johnson-king',
          damping='van-driest', yp_scaling='wall', mu_model='constant',
          eq_Tw=300.0, Aplus=17.0), 1.6, -10.0),
    ('adi-jk-vd-wall-const',
     dict(thermal_bc='adiabatic', mu_t_model='johnson-king',
          damping='van-driest', yp_scaling='wall', mu_model='constant',
          Aplus=17.0), 1.6, 350.0),
    ('iso-prandtl-vd-wall-const',
     dict(thermal_bc='isothermal', mu_t_model='prandtl',
          damping='van-driest', yp_scaling='wall', mu_model='constant',
          eq_Tw=300.0, Aplus=26.0), 1.6, -10.0),
    ('iso-jk-piomelli-wall-const',
     dict(thermal_bc='isothermal', mu_t_model='johnson-king',
          damping='piomelli', yp_scaling='wall', mu_model='constant',
          eq_Tw=300.0, Aplus=17.0), 1.6, -10.0),
    ('iso-jk-sa-wall-const',
     dict(thermal_bc='isothermal', mu_t_model='johnson-king',
          damping='spalart-allmaras', yp_scaling='wall', mu_model='constant',
          eq_Tw=300.0, Aplus=17.0), 1.6, -10.0),
    ('iso-prandtl-sa-wall-const',
     dict(thermal_bc='isothermal', mu_t_model='prandtl',
          damping='spalart-allmaras', yp_scaling='wall', mu_model='constant',
          eq_Tw=300.0, Aplus=26.0), 1.6, -10.0),
    ('iso-jk-vd-semilocal-const',
     dict(thermal_bc='isothermal', mu_t_model='johnson-king',
          damping='van-driest', yp_scaling='semi-local', mu_model='constant',
          eq_Tw=300.0, Aplus=17.0), 1.6, -10.0),
    ('iso-jk-vd-mixedmin2-const',
     dict(thermal_bc='isothermal', mu_t_model='johnson-king',
          damping='van-driest', yp_scaling='mixedmin2', mu_model='constant',
          eq_Tw=300.0, Aplus=17.0), 1.6, -10.0),
    ('iso-jk-vd-wall-sutherland',
     dict(thermal_bc='isothermal', mu_t_model='johnson-king',
          damping='van-driest', yp_scaling='wall', mu_model='sutherland',
          eq_Tw=300.0, Aplus=17.0), 1.6, -10.0),
    ('adi-prandtl-piomelli-mixedmin2-sutherland',
     dict(thermal_bc='adiabatic', mu_t_model='prandtl', damping='piomelli',
          yp_scaling='mixedmin2', mu_model='sutherland', Aplus=26.0),
     1.6, 350.0),
]


def _eq_params(opts):
    P = dict(gamma=_EQ_K['gamma'], cp=_EQ_K['cp'], Pr=_EQ_K['Pr'],
             Prt=_EQ_K['Prt'], mu_c=_EQ_K['mu'], kappa=_EQ_K['kappa'],
             Cv1=_EQ_K['Cv1'], mu_ref=_EQ_K['mu_ref'], Tref=_EQ_K['Tref'],
             S=_EQ_K['S'], p_m=_EQ_PM)
    P.update(opts)
    return P


@pytest.mark.parametrize('cid,opts,gen_tau,gen_p2', _EQ_CONFIGS,
                         ids=[c[0] for c in _EQ_CONFIGS])
def test_eq_ode_wall_self_consistent_recovery(cid, opts, gen_tau, gen_p2):
    # Generate (U_LES, T_m) by integrating the primary ODE at known
    # (tau_w, p2), then shoot to recover them.
    P = _eq_params(opts)
    U_LES, T_m = _eq_integrate(gen_tau, gen_p2, P, _EQ_HWM)
    p1, p2 = eq_ode_wall_solve(U_LES, T_m, P, _EQ_HWM)
    assert np.isclose(p1, gen_tau, rtol=1e-7)
    assert np.isclose(p2, gen_p2, rtol=1e-7)


# ===========================================================================
# Kernel cross-check: REAL eq-ode-wall mako kernel vs the numpy reference.
# The macro stashes the converged (tau_w, p2) in warm_state[0..1].
# ===========================================================================

def _run_eq_wall_kernel(be, P, U_LES, T_m, h_wm):
    be.pointwise.register('pyfr.tests.eq_wall_test_kernel')
    g, cp = P['gamma'], P['cp']
    rho_m = g*P['p_m']/((g - 1)*cp*T_m)

    c = {'gamma': g, 'cp': cp, 'eq_mu': P['mu_c'], 'eq_Pr': P['Pr'],
         'eq_Prt': P['Prt'], 'eq_kappa': P['kappa'], 'eq_Aplus': P['Aplus'],
         'eq_Cv1': P['Cv1']}
    if P['thermal_bc'] == 'isothermal':
        c['eq_Tw'] = P['eq_Tw']
    if P['mu_model'] == 'sutherland':
        c.update({'eq_mu_ref': P['mu_ref'], 'eq_Tref': P['Tref'],
                  'eq_Sconst': P['S']})

    tpl = dict(ndims=_NDIMS, nvars=_NVARS, rsolver='rusanov',
               thermal_bc=P['thermal_bc'], eq_mu_t_model=P['mu_t_model'],
               eq_damping=P['damping'], eq_yp_scaling=P['yp_scaling'],
               eq_mu_model=P['mu_model'], eq_newton_max_iters=60,
               eq_newton_rtol=1e-10, eq_ode_max_steps=8000,
               eq_ode_rtol=1e-9, eq_ode_atol=1e-11, c=c)

    n = 1
    ul = np.zeros((_NVARS, n)); ul[0] = rho_m
    ul[-1] = P['p_m']/(g - 1) + 0.5*rho_m*U_LES**2
    nl = np.zeros((_NDIMS, n)); nl[2] = 1.0
    um = np.zeros((_NPRIMS, n)); um[0] = rho_m; um[1] = U_LES
    um[_NPRIMS - 1] = P['p_m']
    ws = np.zeros((2, n))
    mats = {k: be.matrix(v.shape, initval=v) for k, v in
            dict(ul=ul, nl=nl, um=um, hw=np.array([[h_wm]]), ws=ws).items()}

    be.kernel('eq_wall_test_kernel', tplargs=tpl, dims=[n],
              extrns={'u_match': f'in fpdtype_t[{_NPRIMS}]',
                      'h_wm': 'in fpdtype_t',
                      'warm_state': 'inout fpdtype_t[2]'},
              ul=mats['ul'], nl=mats['nl'], u_match=mats['um'],
              h_wm=mats['hw'], warm_state=mats['ws']).run()
    return mats['ws'].get().ravel()


@pytest.mark.parametrize('cid,opts,gen_tau,gen_p2', _EQ_CONFIGS,
                         ids=[c[0] for c in _EQ_CONFIGS])
def test_eq_ode_wall_kernel_matches_reference(wall_backend, cid, opts,
                                              gen_tau, gen_p2):
    P = _eq_params(opts)
    U_LES, T_m = _eq_integrate(gen_tau, gen_p2, P, _EQ_HWM)
    p1_ref, p2_ref = eq_ode_wall_solve(U_LES, T_m, P, _EQ_HWM)
    p1_k, p2_k = _run_eq_wall_kernel(wall_backend, P, U_LES, T_m, _EQ_HWM)

    # Real mako == numpy reference, and both recover the generating values.
    assert np.isclose(p1_k, p1_ref, rtol=1e-4)
    assert np.isclose(p2_k, p2_ref, rtol=1e-4)
    assert np.isclose(p1_k, gen_tau, rtol=1e-4)
    assert np.isclose(p2_k, gen_p2, rtol=1e-4)


# ===========================================================================
# A priori vs real DNS: turbulent channel at Re_tau ~ 180 (mean profile).
#
# Space- and time-averaged 1D profile (bundled channel-180-dns-profile.npz,
# derived from the user's PyFR DNS).  The equilibrium wall models are fed the
# DNS *mean* tangential velocity at a matching height y_match and must
# recover the true friction velocity u_tau (from the DNS wall gradient,
# tau_w = mu dU/dy|_wall) to within a few percent across the log layer.
# This is the canonical mean a priori test for equilibrium wall models.
#
# Case (incompressible, Ma ~ 0.1, adiabatic walls): delta=1, rho=1,
# mu=3.5714286e-4  ->  nu = mu/rho,  nominal Re_tau = 180.
# ===========================================================================

import io                                                          # noqa: E402
import pkgutil                                                     # noqa: E402

_DNS_MU = 3.5714286e-4
_DNS_RHO = 1.0
_DNS_DELTA = 1.0
_DNS_NU = _DNS_MU/_DNS_RHO


def _load_dns_channel():
    raw = pkgutil.get_data('pyfr.tests', 'channel-180-dns-profile.npz')
    d = np.load(io.BytesIO(raw))
    o = np.argsort(d['y'])
    y, u, dudy = d['y'][o], np.abs(d['u'])[o], d['dudy'][o]
    rho, p = d['rho'][o], d['p'][o]
    # Per-wall friction velocity from the wall velocity gradient.
    u_tau_bot = np.sqrt(_DNS_MU*abs(dudy[0])/_DNS_RHO)
    u_tau_top = np.sqrt(_DNS_MU*abs(dudy[-1])/_DNS_RHO)
    return dict(y=y, U=u, dudy=dudy, p=p, rho=rho,
                u_tau_bot=u_tau_bot, u_tau_top=u_tau_top,
                u_tau=0.5*(u_tau_bot + u_tau_top))


_DNS = _load_dns_channel()
# Matching heights spanning the (thin) log layer at Re_tau=180.
_DNS_YP = [30, 50, 70, 90, 110, 140]


def _dns_match(yp_target, wall='bottom'):
    # Returns (U_match, h_wm, p_m, rho_m) at the requested wall y+, matching
    # inward from the bottom (y=-1) or top (y=+1) wall.
    u_tau = _DNS['u_tau_bot'] if wall == 'bottom' else _DNS['u_tau_top']
    h_wm = yp_target*_DNS_NU/u_tau
    y_match = -1.0 + h_wm if wall == 'bottom' else 1.0 - h_wm
    U = float(np.interp(y_match, _DNS['y'], _DNS['U']))
    p_m = float(np.interp(y_match, _DNS['y'], _DNS['p']))
    rho_m = float(np.interp(y_match, _DNS['y'], _DNS['rho']))
    return U, h_wm, p_m, rho_m


def test_dns_reference_utau_is_retau180():
    # The bundled profile must describe a Re_tau ~ 180 channel with no-slip
    # walls -- validates the ground truth the models are compared against.
    Re_tau = _DNS['u_tau']*_DNS_DELTA/_DNS_NU
    assert 179.0 < Re_tau < 182.0
    assert _DNS['U'][0] < 1e-3*_DNS['u_tau']*20      # ~no-slip at the wall


@pytest.mark.parametrize('law', ['log-law', 'spalding', 'reichardt'])
@pytest.mark.parametrize('yp', _DNS_YP)
def test_alg_wall_apriori_dns(law, yp):
    U, h_wm, _, _ = _dns_match(yp)
    u_tau = float(alg_wall_utau(U, h_wm, _DNS_NU, law=law,
                                n_iter=80, rel_tol=1e-13)[0])
    rel_err = abs(u_tau - _DNS['u_tau'])/_DNS['u_tau']
    # Equilibrium algebraic models recover the DNS wall stress to <8% across
    # the log layer at this (low) Reynolds number.
    assert rel_err < 0.08, (law, yp, u_tau, _DNS['u_tau'], rel_err)


@pytest.mark.parametrize('yp', _DNS_YP)
def test_gq_ode_wall_apriori_dns(yp):
    U, h_wm, _, _ = _dns_match(yp)
    u_tau = float(gq_ode_wall_utau(U, h_wm, _DNS_NU,
                                   n_iter=99, rel_tol=1e-13)[0])
    rel_err = abs(u_tau - _DNS['u_tau'])/_DNS['u_tau']
    # The integral (Van-Driest) model is the most accurate here (~2%).
    assert rel_err < 0.05, (yp, u_tau, _DNS['u_tau'], rel_err)


@pytest.mark.parametrize('yp', _DNS_YP)
def test_eq_ode_wall_apriori_dns(yp):
    # Adiabatic, constant-mu, JK/van-driest/wall.  cp is arbitrary for this
    # incompressible case (the velocity prediction is cp-insensitive); use a
    # representative air value and derive T_m from the EOS.
    U, h_wm, p_m, rho_m = _dns_match(yp)
    g, cp = 1.4, 1005.0
    P = dict(gamma=g, cp=cp, Pr=0.71, Prt=0.9, mu_c=_DNS_MU, kappa=0.41,
             Cv1=7.1, Aplus=17.0, mu_model='constant', yp_scaling='wall',
             damping='van-driest', mu_t_model='johnson-king',
             thermal_bc='adiabatic', p_m=p_m)
    T_m = g*p_m/((g - 1)*cp*rho_m)
    p1, p2 = eq_ode_wall_solve(U, T_m, P, h_wm)
    rho_w = g*p_m/((g - 1)*cp*max(p2, 1e-3))
    u_tau = np.sqrt(max(p1, 0.0)/rho_w)
    rel_err = abs(u_tau - _DNS['u_tau'])/_DNS['u_tau']
    assert rel_err < 0.08, (yp, u_tau, _DNS['u_tau'], rel_err)


def test_dns_channel_is_symmetric():
    # Both walls must give the same friction velocity (statistical symmetry
    # of the converged channel), and U(y) = U(-y).
    assert abs(_DNS['u_tau_bot'] - _DNS['u_tau_top'])/_DNS['u_tau_bot'] < 5e-3
    yy = np.linspace(-0.95, -0.05, 40)
    U_bot = np.interp(yy, _DNS['y'], _DNS['U'])
    U_top = np.interp(-yy, _DNS['y'], _DNS['U'])
    assert np.max(np.abs(U_bot - U_top))/np.max(_DNS['U']) < 5e-3


@pytest.mark.parametrize('model', ['log-law', 'gq'])
@pytest.mark.parametrize('yp', _DNS_YP)
def test_dns_apriori_both_walls_agree(model, yp):
    # The a priori prediction must be wall-independent: matching inward from
    # the bottom and top walls yields the same u_tau (to channel symmetry).
    def predict(wall):
        U, h_wm, _, _ = _dns_match(yp, wall)
        if model == 'gq':
            return float(gq_ode_wall_utau(U, h_wm, _DNS_NU,
                                          n_iter=99, rel_tol=1e-13)[0])
        return float(alg_wall_utau(U, h_wm, _DNS_NU, law=model,
                                   n_iter=80, rel_tol=1e-13)[0])
    u_bot, u_top = predict('bottom'), predict('top')
    assert abs(u_bot - u_top)/u_bot < 0.01


def _eq_dns_params(p_m, thermal_bc, eq_Tw=None):
    P = dict(gamma=1.4, cp=1005.0, Pr=0.71, Prt=0.9, mu_c=_DNS_MU, kappa=0.41,
             Cv1=7.1, Aplus=17.0, mu_model='constant', yp_scaling='wall',
             damping='van-driest', mu_t_model='johnson-king',
             thermal_bc=thermal_bc, p_m=p_m)
    if eq_Tw is not None:
        P['eq_Tw'] = eq_Tw
    return P


@pytest.mark.parametrize('yp', _DNS_YP)
def test_eq_ode_wall_dns_adiabatic_recovers_wall_temperature(yp):
    # Adiabatic walls (the real BC): q_w = 0 by construction, and the model
    # must integrate the energy equation to recover the DNS wall temperature
    # (from the EOS at the wall) from the matching-point temperature.
    g, cp = 1.4, 1005.0
    U, h_wm, p_m, rho_m = _dns_match(yp)
    T_m = g*p_m/((g - 1)*cp*rho_m)
    Tw_dns = g*_DNS['p'][0]/((g - 1)*cp*_DNS['rho'][0])

    P = _eq_dns_params(p_m, 'adiabatic')
    p1, p2 = eq_ode_wall_solve(U, T_m, P, h_wm)   # p2 = recovered T_w
    # Recovered wall temperature matches the DNS wall temperature.
    assert abs(p2 - Tw_dns)/Tw_dns < 2e-3
    # Wall slightly hotter than the matching point (viscous heating), but the
    # channel is near-isothermal so the offset is tiny.
    assert 0.0 <= (p2 - T_m)/T_m < 5e-3


@pytest.mark.parametrize('yp', _DNS_YP)
def test_eq_ode_wall_dns_heat_flux_is_small(yp):
    # Run isothermal at the DNS wall temperature: the predicted wall heat
    # flux must be small (the DNS is adiabatic), i.e. tiny relative to the
    # viscous-heating scale tau_w * U_LES.
    g, cp = 1.4, 1005.0
    U, h_wm, p_m, rho_m = _dns_match(yp)
    T_m = g*p_m/((g - 1)*cp*rho_m)
    Tw_dns = g*_DNS['p'][0]/((g - 1)*cp*_DNS['rho'][0])

    P = _eq_dns_params(p_m, 'isothermal', eq_Tw=Tw_dns)
    p1, p2 = eq_ode_wall_solve(U, T_m, P, h_wm)   # p2 = -q_w
    q_w = -p2
    assert abs(q_w)/(p1*U) < 0.1


# ===========================================================================
# Instantaneous a priori vs DNS (Re_tau~180 channel).
#
# A compact bundled fixture (channel-180-dns-columns.npz) holds the
# instantaneous streamwise velocity profile u(y) for 4096 wall-normal columns
# sampled across the homogeneous x-z plane of one DNS snapshot (bottom half,
# 48 y-levels).  For each column we form the true instantaneous wall stress
# tau_w = mu dU/dy|_wall (near-wall polynomial fit) and feed the matching-
# height velocity to a wall model.  This exposes the known equilibrium-model
# behaviour: the MEAN wall stress is captured, but the instantaneous value
# correlates only weakly and the model under-predicts its fluctuation -- and
# both worsen as the matching point moves away from the wall.
# ===========================================================================

def _load_dns_columns():
    raw = pkgutil.get_data('pyfr.tests', 'channel-180-dns-columns.npz')
    d = np.load(io.BytesIO(raw))
    return (d['y'], d['u'].astype(np.float64), float(d['mu']),
            float(d['rho']), float(d['u_tau_forcing']))


_DNS_COL_Y, _DNS_COL_U, _DNS_COL_MU, _DNS_COL_RHO, _DNS_UTAU_F = \
    _load_dns_columns()


def _dns_true_tau_w():
    # Near-wall polynomial fit u = c1*eta + c2*eta^2 + c3*eta^3 (eta = y+1,
    # u(wall)=0); c1 = dU/dy|_wall -> tau_w = mu c1, per column.
    k = 6
    eta = _DNS_COL_Y[:k] + 1.0
    M = np.stack([eta, eta**2, eta**3], axis=1)
    c1 = np.linalg.lstsq(M, _DNS_COL_U[:, :k].T, rcond=None)[0][0]
    return _DNS_COL_MU*c1


def _dns_instantaneous_stats(model, yp):
    tau_true = _dns_true_tau_w()
    h_wm = yp*(_DNS_COL_MU/_DNS_COL_RHO)/_DNS_UTAU_F
    y_match = -1.0 + h_wm
    j = np.searchsorted(_DNS_COL_Y, y_match) - 1
    fr = (y_match - _DNS_COL_Y[j])/(_DNS_COL_Y[j + 1] - _DNS_COL_Y[j])
    U = np.abs((1 - fr)*_DNS_COL_U[:, j] + fr*_DNS_COL_U[:, j + 1])
    nu = _DNS_COL_MU/_DNS_COL_RHO
    if model == 'gq':
        u_tau = gq_ode_wall_utau(U, h_wm, nu, n_iter=40, rel_tol=1e-8)
    else:
        u_tau = alg_wall_utau(U, h_wm, nu, law=model, n_iter=40, rel_tol=1e-8)
    tau_model = _DNS_COL_RHO*u_tau**2
    return dict(mean_ratio=tau_model.mean()/tau_true.mean(),
                rms_ratio=tau_model.std()/tau_true.std(),
                corr=float(np.corrcoef(tau_model, tau_true)[0, 1]),
                u_tau_true=np.sqrt(tau_true.mean()/_DNS_COL_RHO))


def test_dns_instantaneous_true_utau_matches_forcing():
    # The instantaneous wall stress recovered from the near-wall fit must
    # average to the forcing-balance value (tau_w = -dp/dx * delta).
    u_tau = _dns_instantaneous_stats('log-law', 50)['u_tau_true']
    assert abs(u_tau - _DNS_UTAU_F)/_DNS_UTAU_F < 0.01


@pytest.mark.parametrize('model', ['log-law', 'gq'])
@pytest.mark.parametrize('yp', [30, 50, 90])
def test_dns_instantaneous_mean_captured(model, yp):
    # Averaged over instantaneous inputs the model recovers the mean wall
    # stress to within ~15% (slight equilibrium over-prediction at low Re).
    s = _dns_instantaneous_stats(model, yp)
    assert 1.0 < s['mean_ratio'] < 1.15


@pytest.mark.parametrize('model', ['log-law', 'gq'])
def test_dns_instantaneous_correlation_decays_with_height(model):
    # Hallmark of equilibrium wall models: weak instantaneous correlation
    # that collapses as the matching point moves outward.
    c30 = _dns_instantaneous_stats(model, 30)['corr']
    c50 = _dns_instantaneous_stats(model, 50)['corr']
    c90 = _dns_instantaneous_stats(model, 90)['corr']
    assert c30 > c50 > c90
    assert 0.15 < c30 < 0.32      # modest even close to the wall
    assert c90 < 0.10             # essentially decorrelated far out


@pytest.mark.parametrize('model', ['log-law', 'gq'])
def test_dns_instantaneous_rms_underpredicted(model):
    # The model smooths the input, under-predicting the wall-stress
    # fluctuation, increasingly so with matching height.
    r30 = _dns_instantaneous_stats(model, 30)['rms_ratio']
    r90 = _dns_instantaneous_stats(model, 90)['rms_ratio']
    assert r30 < 1.0
    assert r90 < r30
