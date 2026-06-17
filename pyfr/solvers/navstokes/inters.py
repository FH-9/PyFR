import numpy as np
from numpy.polynomial.legendre import leggauss

from pyfr.mpiutil import get_comm_rank_root
from pyfr.solvers.base.inters import _get_inter_arrays
from pyfr.solvers.baseadvecdiff import (BaseAdvectionDiffusionBCInters,
                                        BaseAdvectionDiffusionIntInters,
                                        BaseAdvectionDiffusionMPIInters)
from pyfr.solvers.euler.inters import MassFlowBCMixin, PressureBCMixin
from pyfr.solvers.navstokes.wmles import WMLESMatchingMixin


class TplargsMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        rsolver = self.cfg.get('solver-interfaces', 'riemann-solver')
        visc_corr = self.cfg.get('solver', 'viscosity-correction', 'none')
        shock_capturing = self.cfg.get('solver', 'shock-capturing', 'none')
        if shock_capturing == 'entropy-filter':
            self.p_min = self.cfg.getfloat('solver-entropy-filter', 'p-min',
                                           1e-6)
        else:
            self.p_min = self.cfg.getfloat('solver-interfaces', 'p-min',
                                           5*self._be.fpdtype_eps)

        # SGS model. Two independent switches:
        #   sgs-model        : selects the SGS closure ('none', 'vreman')
        #   sgs-include-faces: whether the face-flux kernels also include
        #                      the SGS contribution (default: no).
        # The volume kernel (tflux) always uses the SGS model when one is
        # configured (see elements.py). Face kernels honour both switches.
        cfg_sgs_model = self.cfg.get('solver', 'sgs-model', 'none')
        if cfg_sgs_model not in {'none', 'vreman', 'sigma'}:
            raise ValueError(f'Invalid sgs-model: {cfg_sgs_model!r}')

        include_faces = self.cfg.getbool('solver', 'sgs-include-faces',
                                          False)

        # Effective SGS model seen by face kernels: 'none' when face SGS
        # is disabled, regardless of the configured closure. The
        # face_sgs_model attribute is the post-gating value used by all
        # subsequent SGS plumbing (delta_e view setup, MPI exchange
        # registration, etc.) — checking it directly makes the gating
        # obvious at every call site.
        self.face_sgs_model = cfg_sgs_model if include_faces else 'none'

        self._tplargs = dict(ndims=self.ndims, nvars=self.nvars,
                             rsolver=rsolver, visc_corr=visc_corr,
                             shock_capturing=shock_capturing, c=self.c,
                             p_min=self.p_min,
                             sgs_model=self.face_sgs_model)

        # SGS-specific tplargs (only when face SGS is active)
        if self.face_sgs_model != 'none':
            if self.face_sgs_model == 'vreman':
                self._tplargs['c_vreman'] = self.cfg.getfloat(
                    'solver', 'c-vreman', 0.07)
            elif self.face_sgs_model == 'sigma':
                self._tplargs['c_sigma'] = self.cfg.getfloat(
                    'solver', 'c-sigma', 1.35)
            # Turbulent Prandtl number — mirror elements.py default.
            self._tplargs['c'].setdefault('Prt', self.cfg.getfloat(
                'constants', 'Prt', 0.9))


class NavierStokesIntInters(TplargsMixin,
                            BaseAdvectionDiffusionIntInters):
    def __init__(self, be, lhs, rhs, elemap, cfg):
        super().__init__(be, lhs, rhs, elemap, cfg)

        self._be.pointwise.register('pyfr.solvers.navstokes.kernels.intconu')
        self._be.pointwise.register('pyfr.solvers.navstokes.kernels.intcflux')

        self.kernels['con_u'] = lambda: self._be.kernel(
            'intconu', tplargs=self._tplargs, dims=[self.ninterfpts],
            ulin=self.scal_lhs, urin=self.scal_rhs,
            ulout=self._comm_lhs, urout=self._comm_rhs
        )

        # Per-side SGS filter-width per-fpt const matrices (only when
        # face SGS is on). delta_e is per-element and constant in time
        # (no h-/p-adaptivity), so we materialise the per-fpt values
        # once at setup as a regular const_matrix — no runtime view or
        # MPI buffer needed.
        extra = {}
        if self.face_sgs_model != 'none':
            extra['delta_e_l'] = self._const_mat(
                lhs, 'get_delta_e_for_inters_np')
            extra['delta_e_r'] = self._const_mat(
                rhs, 'get_delta_e_for_inters_np')

        self.kernels['comm_flux'] = lambda: self._be.kernel(
            'intcflux', tplargs=self._tplargs, dims=[self.ninterfpts],
            ul=self.scal_lhs, ur=self.scal_rhs,
            gradul=self._vect_lhs, gradur=self._vect_rhs,
            artvisc=self.artvisc, nl=self._pnorm_lhs, **extra
        )


class NavierStokesMPIInters(TplargsMixin,
                            BaseAdvectionDiffusionMPIInters):
    def __init__(self, be, lhs, rhsrank, elemap, cfg):
        super().__init__(be, lhs, rhsrank, elemap, cfg)

        self._be.pointwise.register('pyfr.solvers.navstokes.kernels.mpiconu')
        self._be.pointwise.register('pyfr.solvers.navstokes.kernels.mpicflux')

        self.kernels['con_u'] = lambda: self._be.kernel(
            'mpiconu', tplargs=self._tplargs, dims=[self.ninterfpts],
            ulin=self.scal_lhs, urin=self.scal_rhs, ulout=self._comm_lhs
        )

        # Per-side SGS filter-width as per-fpt const matrices (only when
        # face SGS is on). delta_e is per-element and constant in time
        # so we exchange the neighbour's values ONCE at setup via a
        # direct mpi4py Sendrecv — no runtime MPI traffic, no per-step
        # pack/unpack overhead. Both delta_e_l (local) and delta_e_r
        # (received) are stored as regular const_matrices.
        extra = {}
        if self.face_sgs_model != 'none':
            # Local LHS delta_e per fpt — same iteration as _const_mat
            # would produce, but we also need the numpy array for the
            # MPI send buffer.
            local_arr = _get_inter_arrays(
                lhs, 'get_delta_e_for_inters_np', elemap, self._perm
            )
            local_arr = local_arr[0] if local_arr else np.empty(0)

            # One-shot exchange with the neighbour rank. The LHS-RHS
            # fpt pairing is consistent across ranks (set up by PyFR's
            # interface topology), so the array order on our send side
            # matches the order on the neighbour's recv side and vice
            # versa.
            comm, _, _ = get_comm_rank_root()
            remote_arr = np.empty_like(local_arr)
            tag = self.next_mpi_tag()
            comm.Sendrecv(local_arr, dest=self.rhsrank, sendtag=tag,
                          recvbuf=remote_arr, source=self.rhsrank,
                          recvtag=tag)

            # Wrap both as const_matrices for the kernel.
            extra['delta_e_l'] = self._be.const_matrix(
                np.atleast_2d(local_arr), tags={'align'})
            extra['delta_e_r'] = self._be.const_matrix(
                np.atleast_2d(remote_arr), tags={'align'})

        self.kernels['comm_flux'] = lambda: self._be.kernel(
            'mpicflux', tplargs=self._tplargs, dims=[self.ninterfpts],
            ul=self.scal_lhs, ur=self.scal_rhs,
            gradul=self._vect_lhs, gradur=self._vect_rhs,
            artvisc=self.artvisc, nl=self._pnorm_lhs, **extra
        )


class NavierStokesBaseBCInters(TplargsMixin, BaseAdvectionDiffusionBCInters):
    cflux_state = None

    # Whether this BC's bc_common_flux_state invokes viscous_flux_add and
    # should therefore honour the SGS contribution. All wall BCs override
    # this to False — at viscous walls the unresolved turbulence is
    # highly anisotropic and standard SGS closures break down in the
    # viscous sublayer; for wall-modeled walls the wall model already
    # represents the unresolved physics. Non-wall BCs (inflow, outflow,
    # far-field) keep the default True.
    uses_face_sgs = True

    def __init__(self, be, lhs, elemap, cfgsect, cfg, bccomm):
        super().__init__(be, lhs, elemap, cfgsect, cfg, bccomm)

        # Additional BC specific template arguments
        self._tplargs['bctype'] = self.type
        self._tplargs['bccfluxstate'] = self.cflux_state

        # Per-BC SGS override: if this BC doesn't use viscous_flux_add,
        # strip SGS from its bccflux kernel entirely (no delta_e arg in
        # the kernel signature, no SGS branch in viscous_flux_add).
        if not self.uses_face_sgs:
            self._tplargs['sgs_model'] = 'none'

        self._be.pointwise.register('pyfr.solvers.navstokes.kernels.bcconu')
        self._be.pointwise.register('pyfr.solvers.navstokes.kernels.bccflux')

        # SGS filter-width per-fpt const matrix for the LHS (interior)
        # side. BCs have no genuine RHS element, so there is no
        # delta_e_r — the BC's own bc_common_flux_state operates on a
        # ghost state derived from the LHS and uses the LHS delta_e.
        extra = {}
        if self._tplargs['sgs_model'] != 'none':
            extra['delta_e_l'] = self._const_mat(
                lhs, 'get_delta_e_for_inters_np')

        self.kernels['con_u'] = lambda: self._be.kernel(
            'bcconu', tplargs=self._tplargs, dims=[self.ninterfpts],
            extrns=self._external_args, ulin=self.scal_lhs,
            ulout=self._comm_lhs, nlin=self._pnorm_lhs,
            **self._external_vals
        )
        self.kernels['comm_flux'] = lambda: self._be.kernel(
            'bccflux', tplargs=self._tplargs, dims=[self.ninterfpts],
            extrns=self._external_args, ul=self.scal_lhs,
            gradul=self._vect_lhs, nl=self._pnorm_lhs,
            artvisc=self.artvisc, **self._external_vals, **extra
        )

    def comm_entropy_kernel(self, entmin_lhs):
        # Physics-specific callback for entropy filtering
        self._be.pointwise.register(
            'pyfr.solvers.navstokes.kernels.bccent'
        )

        return lambda: self._be.kernel(
            'bccent', tplargs=self._tplargs, dims=[self.ninterfpts],
            extrns=self._external_args, entmin_lhs=entmin_lhs,
            nl=self._pnorm_lhs, ul=self.scal_lhs, **self._external_vals
        )


class NavierStokesNoSlpIsotWallBCInters(NavierStokesBaseBCInters):
    type = 'no-slp-isot-wall'
    cflux_state = 'ghost-imperm'
    uses_face_sgs = False

    def __init__(self, be, lhs, elemap, cfgsect, cfg, bccomm):
        super().__init__(be, lhs, elemap, cfgsect, cfg, bccomm)

        self.c['cpTw'], = self._eval_opts(['cpTw'])
        self.c |= self._exp_opts('uvw'[:self.ndims], lhs,
                                 default={'u': 0, 'v': 0, 'w': 0})


class NavierStokesNoSlpAdiaWallBCInters(NavierStokesBaseBCInters):
    type = 'no-slp-adia-wall'
    cflux_state = 'ghost-imperm'
    uses_face_sgs = False


class NavierStokesSlpAdiaWallBCInters(NavierStokesBaseBCInters):
    type = 'slp-adia-wall'
    cflux_state = None
    uses_face_sgs = False


class NavierStokesSlpWallBCInters(NavierStokesBaseBCInters):
    type = 'slp-wall'
    cflux_state = None
    uses_face_sgs = False

    def __init__(self, be, lhs, elemap, cfgsect, cfg, bccomm):
        super().__init__(be, lhs, elemap, cfgsect, cfg, bccomm)

        # Prescribed slip length l — compiled as a kernel constant
        self.c['slip_l'] = cfg.getfloat(cfgsect, 'slip-length', 0.0)

        # Wall velocity components (default zero for stationary wall)
        self.c |= self._exp_opts('uvw'[:self.ndims], lhs,
                                 default={'u': 0, 'v': 0, 'w': 0})


class NavierStokesCharRiemInvBCInters(NavierStokesBaseBCInters):
    type = 'char-riem-inv'
    cflux_state = 'ghost'

    def __init__(self, be, lhs, elemap, cfgsect, cfg, bccomm):
        super().__init__(be, lhs, elemap, cfgsect, cfg, bccomm)

        self.c |= self._exp_opts(
            ['rho', 'p', 'u', 'v', 'w'][:self.ndims + 2], lhs
        )


class NavierStokesSupInflowBCInters(NavierStokesBaseBCInters):
    type = 'sup-in-fa'
    cflux_state = 'ghost'

    def __init__(self, be, lhs, elemap, cfgsect, cfg, bccomm):
        super().__init__(be, lhs, elemap, cfgsect, cfg, bccomm)

        self.c |= self._exp_opts(
            ['rho', 'p', 'u', 'v', 'w'][:self.ndims + 2], lhs
        )


class NavierStokesSupOutflowBCInters(NavierStokesBaseBCInters):
    type = 'sup-out-fn'
    cflux_state = 'ghost'


class NavierStokesSubInflowFrvBCInters(NavierStokesBaseBCInters):
    type = 'sub-in-frv'
    cflux_state = 'ghost'

    def __init__(self, be, lhs, elemap, cfgsect, cfg, bccomm):
        super().__init__(be, lhs, elemap, cfgsect, cfg, bccomm)

        self.c |= self._exp_opts(
            ['rho', 'u', 'v', 'w'][:self.ndims + 1], lhs,
            default={'u': 0, 'v': 0, 'w': 0}
        )


class NavierStokesSubInflowFtpttangBCInters(NavierStokesBaseBCInters):
    type = 'sub-in-ftpttang'
    cflux_state = 'ghost'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        gamma = self.cfg.getfloat('constants', 'gamma')

        # Pass boundary constants to the backend
        self.c['cpTt'], = self._eval_opts(['cpTt'])
        self.c['pt'], = self._eval_opts(['pt'])
        self.c['Rdcp'] = (gamma - 1.0)/gamma

        # Calculate u, v velocity components from the inflow angle
        theta = self._eval_opts(['theta'])[0]*np.pi/180.0
        velcomps = np.array([np.cos(theta), np.sin(theta), 1.0])

        # Adjust u, v and calculate w velocity components for 3-D
        if self.ndims == 3:
            phi = self._eval_opts(['phi'])[0]*np.pi/180.0
            velcomps[:2] *= np.sin(phi)
            velcomps[2] *= np.cos(phi)

        self.c['vc'] = velcomps[:self.ndims]


class NavierStokesSubOutflowBCInters(NavierStokesBaseBCInters):
    type = 'sub-out-fp'
    cflux_state = 'ghost'

    def __init__(self, be, lhs, elemap, cfgsect, cfg, bccomm):
        super().__init__(be, lhs, elemap, cfgsect, cfg, bccomm)

        self.c |= self._exp_opts(['p'], lhs)


class NavierStokesCharRiemInvMassFlowBCInters(MassFlowBCMixin,
                                              NavierStokesBaseBCInters):
    type = 'char-riem-inv-mass-flow'
    cflux_state = 'ghost'


class NavierStokesCharRiemInvPressureBCInters(PressureBCMixin,
                                              NavierStokesBaseBCInters):
    type = 'char-riem-inv-pressure'
    cflux_state = 'ghost'


class NavierStokesGQODEWallBCInters(WMLESMatchingMixin,
                                    NavierStokesBaseBCInters):
    type = 'gq-ode-wall'
    cflux_state = None
    uses_face_sgs = False

    def __init__(self, be, lhs, elemap, cfgsect, cfg, bccomm):
        super().__init__(be, lhs, elemap, cfgsect, cfg, bccomm)

        # Per-fpt matching infrastructure (cp config, geometry, matching
        # kernel, buffer externs). GQWM only warm-starts u_tau, so only
        # one slot in the warm-state buffer is needed.
        self._wmles_setup(be, lhs, elemap, cfgsect, cfg, n_warm_params=1)

        # van Driest model parameters
        kappa = cfg.getfloat(cfgsect, 'kappa', 0.41)
        Aplus = cfg.getfloat(cfgsect, 'A-plus', 26.0)
        n_quad = cfg.getint(cfgsect, 'n-quad', 15)
        n_iter = cfg.getint(cfgsect, 'n-iter', 6)
        rel_tol = cfg.getfloat(cfgsect, 'rel-tol', 1e-3)

        # Standard GL nodes/weights on [-1, 1]
        xi_std, w_std = leggauss(n_quad)

        # Exponential mapping ξ ∈ [-1, 1] → y ∈ [0, h_wm]:
        #   y(ξ) = h_wm (exp(ξ+1) − 1) / (exp(2) − 1)
        #   dy/dξ = h_wm exp(ξ+1) / (exp(2) − 1)
        # Per-fpt mapped y and weights w_phys are precomputed once at
        # setup and exposed to the kernel via externs.
        h_wm_all = np.concatenate([g['h_wm'] for g in self._wmles_groups])
        h_wm_bc = h_wm_all[self._wmles_concat_to_bc_order]      # (n_fpts,)
        exp_xi_p1 = np.exp(xi_std + 1.0)                        # (n_quad,)
        denom = np.exp(2.0) - 1.0
        y_map = (h_wm_bc[None, :] * (exp_xi_p1[:, None] - 1.0) / denom)
        w_map = (w_std[:, None] * h_wm_bc[None, :] * exp_xi_p1[:, None]
                 / denom)
        self._gq_y_mat = be.const_matrix(y_map, tags={'align'})
        self._gq_w_mat = be.const_matrix(w_map, tags={'align'})
        self.set_external('gq_y', f'in fpdtype_t[{n_quad}]',
                          value=self._gq_y_mat)
        self.set_external('gq_w', f'in fpdtype_t[{n_quad}]',
                          value=self._gq_w_mat)

        # Physics constants embedded in kernel as compile-time values
        self.c['gq_mu'] = cfg.getfloat('constants', 'mu')
        self.c['gq_kappa'] = kappa
        self.c['gq_Aplus'] = Aplus
        self.c['gq_nquad'] = n_quad
        self.c['gq_n_iter'] = n_iter
        self.c['gq_rel_tol'] = rel_tol


class NavierStokesEqODEWallBCInters(WMLESMatchingMixin,
                                    NavierStokesBaseBCInters):
    type = 'eq-ode-wall'
    cflux_state = None
    uses_face_sgs = False

    def __init__(self, be, lhs, elemap, cfgsect, cfg, bccomm):
        super().__init__(be, lhs, elemap, cfgsect, cfg, bccomm)

        # Matching infrastructure shared with all WMLES BCs
        self._wmles_setup(be, lhs, elemap, cfgsect, cfg)

        # --- Thermal-BC variant ---
        thermal_bc = cfg.get(cfgsect, 'thermal-bc', 'isothermal')
        if thermal_bc not in ('isothermal', 'adiabatic'):
            raise ValueError(
                f"[{cfgsect}] thermal-bc must be 'isothermal' or 'adiabatic'"
            )

        # --- Eddy-viscosity, damping, y+-scaling, viscosity-law options ---
        mu_t_model = cfg.get(cfgsect, 'mu-t-model', 'johnson-king')
        if mu_t_model not in ('johnson-king', 'prandtl'):
            raise ValueError(
                f"[{cfgsect}] mu-t-model must be 'johnson-king' or 'prandtl'"
            )

        damping = cfg.get(cfgsect, 'damping', 'van-driest')
        if damping not in ('van-driest', 'spalart-allmaras', 'piomelli'):
            raise ValueError(
                f"[{cfgsect}] damping must be one of 'van-driest', "
                f"'spalart-allmaras', 'piomelli'"
            )

        yp_scaling = cfg.get(cfgsect, 'yp-scaling', 'wall')
        if yp_scaling not in ('wall', 'semi-local', 'mixedmin2'):
            raise ValueError(
                f"[{cfgsect}] yp-scaling must be one of 'wall', 'semi-local', "
                f"'mixedmin2'"
            )

        mu_model = cfg.get(cfgsect, 'mu-model', 'constant')
        if mu_model not in ('constant', 'sutherland'):
            raise ValueError(
                f"[{cfgsect}] mu-model must be 'constant' or 'sutherland'"
            )

        # --- Physical constants ---
        kappa = cfg.getfloat(cfgsect, 'kappa', 0.41)
        Aplus_default = 17.0 if mu_t_model == 'johnson-king' else 26.0
        Aplus = cfg.getfloat(cfgsect, 'A-plus', Aplus_default)
        Cv1 = cfg.getfloat(cfgsect, 'C-v1', 7.1)
        Prt = cfg.getfloat(cfgsect, 'Pr-t', 0.9)

        # Required from [constants] for the wall model
        self.c['eq_mu'] = cfg.getfloat('constants', 'mu')
        self.c['eq_Pr'] = cfg.getfloat('constants', 'Pr')

        # BC-level constants (avoid polluting global [constants])
        self.c['eq_kappa'] = kappa
        self.c['eq_Aplus'] = Aplus
        self.c['eq_Cv1'] = Cv1
        self.c['eq_Prt'] = Prt

        if thermal_bc == 'isothermal':
            self.c['eq_Tw'] = cfg.getfloat(cfgsect, 'Tw')

        # Sutherland's-law parameters (only needed when mu-model is sutherland)
        if mu_model == 'sutherland':
            self.c['eq_mu_ref'] = cfg.getfloat(cfgsect, 'mu-ref')
            self.c['eq_Tref']   = cfg.getfloat(cfgsect, 'T-ref')
            self.c['eq_Sconst'] = cfg.getfloat(cfgsect, 'S-const')

        # --- Newton + ODE-integrator parameters ---
        newton_max_iters = cfg.getint(cfgsect, 'newton-max-iters', 10)
        newton_rtol = cfg.getfloat(cfgsect, 'newton-rtol', 1e-4)
        ode_max_steps = cfg.getint(cfgsect, 'ode-max-steps', 200)
        ode_rtol = cfg.getfloat(cfgsect, 'ode-rtol', 1e-4)
        ode_atol = cfg.getfloat(cfgsect, 'ode-atol', 1e-6)

        # --- Tplargs ---
        self._tplargs['thermal_bc'] = thermal_bc
        self._tplargs['eq_mu_t_model'] = mu_t_model
        self._tplargs['eq_damping'] = damping
        self._tplargs['eq_yp_scaling'] = yp_scaling
        self._tplargs['eq_mu_model'] = mu_model
        self._tplargs['eq_newton_max_iters'] = newton_max_iters
        self._tplargs['eq_newton_rtol'] = newton_rtol
        self._tplargs['eq_ode_max_steps'] = ode_max_steps
        self._tplargs['eq_ode_rtol'] = ode_rtol
        self._tplargs['eq_ode_atol'] = ode_atol


class NavierStokesAlgWallBCInters(WMLESMatchingMixin,
                                  NavierStokesBaseBCInters):
    type = 'alg-wall'
    cflux_state = None
    uses_face_sgs = False

    def __init__(self, be, lhs, elemap, cfgsect, cfg, bccomm):
        super().__init__(be, lhs, elemap, cfgsect, cfg, bccomm)

        # Algebraic wall models do 1D Newton on u_tau — one warm slot.
        self._wmles_setup(be, lhs, elemap, cfgsect, cfg, n_warm_params=1)

        # --- Wall-function law selection ---
        law = cfg.get(cfgsect, 'law', 'log-law')
        if law not in ('log-law', 'spalding', 'reichardt'):
            raise ValueError(
                f"[{cfgsect}] law must be 'log-law', 'spalding', or "
                f"'reichardt'"
            )

        # --- Thermal BC (alg-wall is adiabatic only for now) ---
        thermal_bc = cfg.get(cfgsect, 'thermal-bc', 'adiabatic')
        if thermal_bc != 'adiabatic':
            raise NotImplementedError(
                f"[{cfgsect}] alg-wall only supports thermal-bc = 'adiabatic' "
                f"in this version; isothermal support pending."
            )

        # --- Physical constants ---
        kappa = cfg.getfloat(cfgsect, 'kappa', 0.41)
        B_const = cfg.getfloat(cfgsect, 'B-const', 5.0)
        C_const = cfg.getfloat(cfgsect, 'C-const', 7.3)
        B1_const = cfg.getfloat(cfgsect, 'B1-const', 11.0)
        B2_const = cfg.getfloat(cfgsect, 'B2-const', 3.0)

        self.c['alg_mu'] = cfg.getfloat('constants', 'mu')
        self.c['alg_kappa'] = kappa
        self.c['alg_B'] = B_const
        self.c['alg_C'] = C_const
        self.c['alg_B1'] = B1_const
        self.c['alg_B2'] = B2_const
        # Precomputed exp(-kappa*B), needed by the Spalding branch as a
        # compile-time constant; doing it here avoids needing math.exp
        # in the mako render context.
        self.c['alg_expmkB'] = np.exp(-kappa*B_const)

        # --- Iteration parameters ---
        n_iter = cfg.getint(cfgsect, 'n-iter', 8)
        rel_tol = cfg.getfloat(cfgsect, 'rel-tol', 1e-3)

        # --- Tplargs ---
        self._tplargs['alg_law'] = law
        self.c['alg_n_iter'] = n_iter
        self.c['alg_rel_tol'] = rel_tol
