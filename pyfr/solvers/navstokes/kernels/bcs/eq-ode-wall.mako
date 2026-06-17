<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%include file='pyfr.solvers.euler.kernels.rsolvers.${rsolver}'/>
<%include file='pyfr.solvers.navstokes.kernels.bcs.common'/>

<%pyfr:macro name='bc_rsolve_state' params='ul, nl, ur' externs='ploc, t'>
    // No-slip impermeable mirror ghost for the Riemann solver
    // (negate momentum, preserve total energy).
    ur[0] = ul[0];
% for i in range(ndims):
    ur[${i + 1}] = -ul[${i + 1}];
% endfor
    ur[${nvars - 1}] = ul[${nvars - 1}];
</%pyfr:macro>

<%pyfr:macro name='bc_ldg_state' params='ul, nl, ur' externs='ploc, t'>
    // No-slip impermeable actual-state ghost for the LDG common solution
    // (zero momentum, strip kinetic energy to leave internal energy).
    ur[0] = ul[0];
% for i in range(ndims):
    ur[${i + 1}] = 0;
% endfor
    ur[${nvars - 1}] = ul[${nvars - 1}]
                     - (0.5/ul[0])*${pyfr.dot('ul[{i}]', i=(1, ndims + 1))};
</%pyfr:macro>

## bc_ldg_grad_state is never expanded for this BC: ghost.mako and
## ghost-imperm.mako (the only consumers) are only included by the
## framework when cflux_state is set, and this BC uses cflux_state =
## None with a custom bc_common_flux_state that doesn't call
## viscous_flux_add or use gradur. The alias is provided defensively in
## case the BC is ever wired to the standard ghost flux path in the
## future; bc_common_grad_zero is the safest default because our wall
## model already provides the tangential stress, so a non-zero ghost
## gradient would double-count it. The alias is a mako-level
## declaration only — no C code is generated for it in this path.
<%pyfr:alias name='bc_ldg_grad_state' func='bc_common_grad_zero'/>

##
## ============================================================
## Equilibrium two-equation ODE wall model.
##
## State vector layout (length 6):
##   s[0]  = u           (tangential velocity)
##   s[1]  = T           (temperature)
##   s[2]  = du/dp1      (sensitivity, 1st shooting var)
##   s[3]  = du/dp2      (sensitivity, 2nd shooting var)
##   s[4]  = dT/dp1
##   s[5]  = dT/dp2
##
## Shooting variables (p1, p2):
##   isothermal: (tau_w, -q_w)
##   adiabatic : (tau_w, T_w)
##
## Boundary at y = 0:
##   u = 0 always.
##   T = T_w_given      (isothermal)
##   T = p2 = T_w guess (adiabatic)
##   S(0) = 0 except for the adiabatic dT/d(T_w) entry which is 1.
##
## ODE integration: Bogacki-Shampine RK23 with PI step-size control.
## Both primary and sensitivity equations advance together as one 6D IVP.
##
## Configurable branches (selected at template compile time via tplargs):
##   eq_mu_t_model  : johnson-king | prandtl
##   eq_damping     : van-driest | piomelli | spalart-allmaras
##   eq_yp_scaling  : wall | semi-local | mixedmin2
##   eq_mu_model    : constant | sutherland
##   thermal_bc     : isothermal | adiabatic
## ============================================================
##

<%pyfr:macro name='eq_compute_rhs'
             params='y, s, p, p_m, rho_w, mu_w, mu_c, rhs'>
    // s     = (u, T, S_uu, S_uT, S_Tu, S_TT)
    // p     = (p1, p2) shooting parameters
    // p_m   = matching-point pressure (used to compute local rho via EOS).
    // rho_w = wall density (used for u_tau and the wall y+ scaling). In
    //         adiabatic mode this changes between Newton iterations because
    //         it is recomputed from T_w = p2 via the BL-thin EOS.
    // mu_w  = wall viscosity  (used for the wall y+ scaling). Constant
    //         throughout one ODE integration but changes between Newton
    //         iterations when T_w is itself a shooting variable.
    // mu_c  = constant-mu value (used when eq_mu_model is 'constant')
    // Computes rhs[0..5] = (du/dy, dT/dy, dS_uu/dy, dS_uT/dy,
    //                       dS_Tu/dy, dS_TT/dy).

    fpdtype_t u  = s[0];
    fpdtype_t T_ = s[1];
    fpdtype_t p1 = p[0];
    fpdtype_t p2 = p[1];

    // Safety-clamped local temperature, reused throughout the macro.
    fpdtype_t T_safe = fmax(T_, (fpdtype_t)1e-3);

% if thermal_bc == 'isothermal':
    // Shooting variables follow document convention: p = (tau_w, -q_w),
    // so q_w = -p2.
    fpdtype_t tau_w = p1;
    fpdtype_t q_w   = -p2;
% else:
    fpdtype_t tau_w = p1;
    fpdtype_t q_w   = 0.0;
% endif

    // ----- local viscosity ----------------------------------------------
% if eq_mu_model == 'sutherland':
    // mu(T) = mu_ref (T/T_ref)^(3/2) (T_ref+S)/(T+S); x*sqrt(x) for x^(3/2)
    fpdtype_t Trat = T_safe/(fpdtype_t)${c['eq_Tref']};
    fpdtype_t mu_l = (fpdtype_t)${c['eq_mu_ref']}*Trat*sqrt(Trat)
                    *(fpdtype_t)${c['eq_Tref'] + c['eq_Sconst']}
                    /(T_safe + (fpdtype_t)${c['eq_Sconst']});
% else:
    fpdtype_t mu_l = mu_c;
% endif

    // ----- local density ------------------------------------------------
    // Boundary-layer-thin assumption gives p(y) ≈ p_m, so
    // rho_local(T) = gamma p_m / ((gamma-1) c_p T) from the ideal-gas EOS.
    fpdtype_t rho_l = (fpdtype_t)${c['gamma']/((c['gamma'] - 1)*c['cp'])}
                      *p_m/T_safe;

    // ----- friction velocity, y+ -----------------------------------------
    // u_tau = sqrt(tau_w / rho_w) is the wall-density friction velocity
    // used for the y+ scalings. The JK eddy viscosity uses a different
    // scale velocity built from local density; computed inline below.
    // tau_w may be negative during early Newton iterations — clamp.
    fpdtype_t tau_safe = fmax(tau_w, (fpdtype_t)1e-12);
    fpdtype_t utau = sqrt(tau_safe/rho_w);

    // y+ uses wall viscosity in the wall component and local viscosity in
    // the semi-local / local components.
% if eq_yp_scaling == 'wall':
    fpdtype_t yp = rho_w*utau*y/mu_w;
% elif eq_yp_scaling == 'semi-local':
    fpdtype_t yp = sqrt(rho_l*rho_w)*utau*y/mu_l;
% else:  ## mixedmin2
    fpdtype_t yp_wall   = rho_w*utau*y/mu_w;
    fpdtype_t yp_local  = rho_l*utau*y/mu_l;
    fpdtype_t yp_semi   = sqrt(rho_l*rho_w)*utau*y/mu_l;
    fpdtype_t yp_mixed  = (fpdtype_t)0.5*(yp_wall  + yp_semi);
    fpdtype_t yp_mixed2 = (fpdtype_t)0.5*(yp_local + yp_semi);
    fpdtype_t yp = fmin(yp_mixed, yp_mixed2);
% endif

    // ----- damping function D --------------------------------------------
    // For Spalart-Allmaras, D is a function of the *undamped* mu_t which
    // depends on du/dy in the Prandtl case — that branch is handled
    // below alongside the eddy-viscosity solve.
% if eq_damping == 'van-driest':
    fpdtype_t Dfac = (fpdtype_t)1.0
                   - exp(-yp/(fpdtype_t)${c['eq_Aplus']});
    fpdtype_t D = Dfac*Dfac;
% elif eq_damping == 'piomelli':
    fpdtype_t yp_a = yp/(fpdtype_t)${c['eq_Aplus']};
    fpdtype_t D = (fpdtype_t)1.0 - exp(-yp_a*yp_a*yp_a);
% else:  ## spalart-allmaras — D set inside the mu_t branch below
    fpdtype_t D = (fpdtype_t)1.0;
% endif

    // ----- eddy viscosity mu_t and du/dy ---------------------------------
    fpdtype_t mu_t;
    fpdtype_t dudy;
% if eq_mu_t_model == 'johnson-king':
    // Undamped mu_hat_t = kappa * y * sqrt(rho * tau_w)  (uses LOCAL rho;
    // equivalent to rho * kappa y * sqrt(tau_w/rho)). Explicit in du/dy.
    fpdtype_t muhat = (fpdtype_t)${c['eq_kappa']}*y*sqrt(rho_l*tau_safe);
% if eq_damping == 'spalart-allmaras':
    fpdtype_t mu3    = mu_l*mu_l*mu_l;
    fpdtype_t muhat3 = muhat*muhat*muhat;
    fpdtype_t Cv13   = (fpdtype_t)${c['eq_Cv1']**3};
    D = muhat3/(muhat3 + Cv13*mu3 + (fpdtype_t)1e-12);
% endif
    mu_t = muhat*D;
    dudy = tau_w/(mu_l + mu_t);
% else:  ## prandtl
    // mu_t = rho * kappa^2 * y^2 * |du/dy| * D
    // (mu + mu_t) du/dy = tau_w
    fpdtype_t alphaP = rho_l*(fpdtype_t)${c['eq_kappa']**2}*y*y;
% if eq_damping == 'spalart-allmaras':
    // D depends on undamped mu_hat_t = alphaP * |du/dy|, which depends on
    // du/dy. Anderson(1)-accelerated fixed-point: equivalent to a secant
    // step on the FP residual res(x) = f(x) - x. One extra scalar of history.
    // First iter and last iter use plain FP — first has no history, last
    // ensures dudy and mu_t at loop exit satisfy dudy*(mu_l + mu_t) = tau_w
    // exactly (the sensitivity-Jacobian block below depends on this).
    fpdtype_t mu3  = mu_l*mu_l*mu_l;
    fpdtype_t Cv13 = (fpdtype_t)${c['eq_Cv1']**3};
    dudy = tau_w/mu_l;                     // laminar initial guess
    fpdtype_t dudy_prev = (fpdtype_t)0.0;
    fpdtype_t res_prev  = (fpdtype_t)0.0;
    for (int it = 0; it < 3; ++it) {
        fpdtype_t muhat  = alphaP*fabs(dudy);
        fpdtype_t muhat3 = muhat*muhat*muhat;
        D    = muhat3/(muhat3 + Cv13*mu3 + (fpdtype_t)1e-12);
        mu_t = muhat*D;
        fpdtype_t T_dudy = tau_w/(mu_l + mu_t);
        fpdtype_t res  = T_dudy - dudy;
        fpdtype_t dres = res - res_prev;
        // Anderson(1) extrapolation when warmed up and the secant
        // denominator is safe; plain FP on iter 0 (no history) and as
        // fallback. Positivity guard catches AA overshoot.
        fpdtype_t dudy_new =
            (it > 0 && fabs(dres) > (fpdtype_t)1e-12)
            ? (res*dudy_prev - res_prev*dudy)/dres
            : T_dudy;
        if (dudy_new <= (fpdtype_t)0.0) dudy_new = T_dudy;
        dudy_prev = dudy;  res_prev = res;  dudy = dudy_new;
    }
% else:
    // D is explicit (Van Driest or Piomelli) → closed-form quadratic.
    // (alphaP * D) (du/dy)^2 + mu (du/dy) - tau_w = 0
    fpdtype_t a = alphaP*D;
    fpdtype_t disc = mu_l*mu_l + (fpdtype_t)4.0*a*tau_w;
    fpdtype_t sqrt_disc = sqrt(fmax(disc, (fpdtype_t)0.0));
    dudy = (a > (fpdtype_t)1e-12)
           ? (-mu_l + sqrt_disc)/((fpdtype_t)2.0*a)
           : tau_w/mu_l;
    mu_t = alphaP*D*fabs(dudy);
% endif
% endif

    // ----- temperature gradient ------------------------------------------
    fpdtype_t alpha_T = (fpdtype_t)${c['cp']}
                       *(mu_l/(fpdtype_t)${c['eq_Pr']}
                         + mu_t/(fpdtype_t)${c['eq_Prt']});
    fpdtype_t dTdy = -(q_w + u*tau_w)/alpha_T;

    rhs[0] = dudy;
    rhs[1] = dTdy;

    // ===== Sensitivity Jacobians: A = dF/d(u,T), B = dF/d(tau_w, -q_w) ==
    // Compositional recipe — atoms then F1, F2 partials. See the user's
    // derivation document for the algebra.
    //
    // Universal simplifications across all listed model combinations:
    //   * mu_t has no explicit u-dependence  =>  dF1/du = 0, dmu_t/du = 0
    //   * F1 has no q_w-dependence           =>  dF1/d(-q_w) = 0
    //   * dF2/d(-q_w) = 1/D2
    //
    // T_safe (declared at the top of the macro) is reused here.

    // --- Atomic derivatives: dmu/dT, drho/dT, dutau/dtau_w --------------
    fpdtype_t dmu_dT;
% if eq_mu_model == 'sutherland':
    dmu_dT = mu_l*(T_safe + (fpdtype_t)${3.0*c['eq_Sconst']})
             /((fpdtype_t)2.0*T_safe*(T_safe + (fpdtype_t)${c['eq_Sconst']}));
% else:
    dmu_dT = (fpdtype_t)0.0;
% endif

    fpdtype_t drho_dT = -rho_l/T_safe;

    // --- y+ derivatives (dyp/du = 0 in every scaling) -------------------
    fpdtype_t dyp_dT = (fpdtype_t)0.0;
    fpdtype_t dyp_dtau = (fpdtype_t)0.0;

% if eq_yp_scaling == 'wall':
    // yp_wall = y sqrt(rho_w tau_w)/mu_w; mu_w, rho_w fixed with respect to local T.
    dyp_dT = (fpdtype_t)0.0;
    dyp_dtau = yp/((fpdtype_t)2.0*tau_safe);
% elif eq_yp_scaling == 'semi-local':
    // yp_SL ∝ sqrt(rho)*sqrt(tau_w)/mu_l
    dyp_dT = yp*(-(fpdtype_t)1.0/((fpdtype_t)2.0*T_safe) - dmu_dT/mu_l);
    dyp_dtau = yp/((fpdtype_t)2.0*tau_safe);
% else:  ## mixedmin2
    // Semi-local component appears in both yp_mixed and yp_mixed2, so always
    // needed. Wall-component derivatives are needed only if yp_mixed wins;
    // local-component derivatives only if yp_mixed2 wins. Compute the
    // winning branch only to save the redundant pair.
    fpdtype_t dyps_dT  = yp_semi*(-(fpdtype_t)1.0/((fpdtype_t)2.0*T_safe)
                                  - dmu_dT/mu_l);
    fpdtype_t dyps_dtau = yp_semi/((fpdtype_t)2.0*tau_safe);
    if (yp_mixed < yp_mixed2) {
        // wall leg: y+_wall has dT/dT = 0 (mu_w, rho_w fixed wrt local T)
        // and dy+_wall/dtau = y+_wall/(2 tau_safe).
        dyp_dT   = (fpdtype_t)0.5*dyps_dT;
        dyp_dtau = (fpdtype_t)0.5*(yp_wall/((fpdtype_t)2.0*tau_safe)
                                   + dyps_dtau);
    } else {
        // local leg: y+_local ∝ rho_l/mu_l with full -1/T log-derivative
        // (no factor 1/2 since y+_local contains rho_l, not sqrt(rho_l)).
        dyp_dT   = (fpdtype_t)0.5*(yp_local*(-(fpdtype_t)1.0/T_safe
                                             - dmu_dT/mu_l) + dyps_dT);
        dyp_dtau = (fpdtype_t)0.5*(yp_local/((fpdtype_t)2.0*tau_safe)
                                   + dyps_dtau);
    }
% endif

    // --- Damping derivatives --------------------------------------------
    // For Spalart-Allmaras these depend on dmuhat/dX and dmu/dX, which
    // need the eddy-viscosity model — those are computed in the mu_t
    // block below where they are used.
    fpdtype_t dD_dT = (fpdtype_t)0.0;
    fpdtype_t dD_dtau = (fpdtype_t)0.0;

% if eq_damping == 'van-driest':
    {
        fpdtype_t e_neg_yp = exp(-yp/(fpdtype_t)${c['eq_Aplus']});
        fpdtype_t dD_dyp = (fpdtype_t)2.0*sqrt(D)*e_neg_yp
                           /(fpdtype_t)${c['eq_Aplus']};
        dD_dT   = dD_dyp*dyp_dT;
        dD_dtau = dD_dyp*dyp_dtau;
    }
% elif eq_damping == 'piomelli':
    {
        fpdtype_t dD_dyp = (fpdtype_t)3.0*yp*yp*((fpdtype_t)1.0 - D)
                           /(fpdtype_t)${c['eq_Aplus']**3};
        dD_dT   = dD_dyp*dyp_dT;
        dD_dtau = dD_dyp*dyp_dtau;
    }
% endif
    // Spalart-Allmaras damping derivatives are filled in below within
    // the mu_t-model branch where the right dmuhat/dX is available.

    // --- F1 partials and dmu_t/dX (varies by mu_t-model x damping) -----
    fpdtype_t dF1_dT;
    fpdtype_t dF1_dtau;
    fpdtype_t dmut_dT;
    fpdtype_t dmut_dtau;

% if eq_mu_t_model == 'johnson-king':
    // muhat^JK = kappa y sqrt(rho_l tau_w). Atomic partials:
    //   dmuhat/du   = 0
    //   dmuhat/dT   = muhat/(2 rho_l) drho/dT = -muhat/(2 T)   (ideal gas)
    //   dmuhat/dtau = muhat/(2 tau_w)
    fpdtype_t dmuhat_dT_jk   = (fpdtype_t)0.5*muhat*drho_dT/rho_l;
    fpdtype_t dmuhat_dtau_jk = muhat/((fpdtype_t)2.0*tau_safe);
    fpdtype_t muhat_safe = fmax(muhat, (fpdtype_t)1e-12);

% if eq_damping == 'spalart-allmaras':
    // SA damping derivatives (Section 3.6):
    //   dD/dX = 3 D (1-D) [dmuhat/dX / muhat - dmu/dX / mu]
    {
        fpdtype_t pre = (fpdtype_t)3.0*D*((fpdtype_t)1.0 - D);
        dD_dT   = pre*(dmuhat_dT_jk/muhat_safe - dmu_dT/mu_l);
        dD_dtau = pre*(dmuhat_dtau_jk/muhat_safe);   // dmu/dtau = 0
    }
% endif

    // mu_t = muhat * D  =>  dmu_t/dX = D dmuhat/dX + muhat dD/dX
    dmut_dT   = D*dmuhat_dT_jk   + muhat*dD_dT;
    dmut_dtau = D*dmuhat_dtau_jk + muhat*dD_dtau;

    // F1 = tau_w / (mu_l + mu_t)   (explicit — Section 2.2)
    {
        fpdtype_t inv_mut = (fpdtype_t)1.0/(mu_l + mu_t);
        dF1_dT   = -dudy*inv_mut*(dmu_dT + dmut_dT);
        dF1_dtau = ((fpdtype_t)1.0 - dudy*dmut_dtau)*inv_mut;
    }

% else:   ## prandtl mixing length

% if eq_damping == 'spalart-allmaras':
    // Implicit framework (Section 2.4).
    //   G(F1, T, tau_w) = (mu_l + mu_t) F1 - tau_w = 0
    //   dG/dF1   = mu_l + mu_t (5 - 3 D)
    //   dG/du    = 0    (mu_t has no explicit u)
    //   dG/dtau  = -1   (no explicit tau_w in mu_t at fixed F1)
    //   dG/dT|F1 = F1 [dmu/dT + dmu_t/dT|F1]
    fpdtype_t dG_dF1 = mu_l + mu_t*((fpdtype_t)5.0 - (fpdtype_t)3.0*D);
    fpdtype_t inv_dG_dF1 = (fpdtype_t)1.0/dG_dF1;

    // dmu_t/dT |_F1 = -mu_t [(4-3D)/T + 3(1-D)/mu dmu/dT]
    fpdtype_t dmut_dT_F1 =
        -mu_t*(((fpdtype_t)4.0 - (fpdtype_t)3.0*D)/T_safe
               + (fpdtype_t)3.0*((fpdtype_t)1.0 - D)/mu_l*dmu_dT);
    fpdtype_t dG_dT = dudy*(dmu_dT + dmut_dT_F1);

    dF1_dT   = -dG_dT*inv_dG_dF1;
    dF1_dtau = inv_dG_dF1;

    // Total dmu_t/dX along the constraint surface:
    //   dmu_t/dX|total = dmu_t/dF1 * dF1/dX + dmu_t/dX|F1
    // with dmu_t/dF1 = mu_t (4-3D)/F1.
    fpdtype_t dudy_safe = (fabs(dudy) > (fpdtype_t)1e-15)
                          ? dudy : (fpdtype_t)1e-15;
    fpdtype_t dmut_dF1 = mu_t*((fpdtype_t)4.0 - (fpdtype_t)3.0*D)/dudy_safe;
    dmut_dT   = dmut_dF1*dF1_dT + dmut_dT_F1;
    dmut_dtau = dmut_dF1*dF1_dtau;   // no explicit tau_w dep at fixed F1

% else:   ## prandtl with van-driest or piomelli (explicit D)
    // Closed-form quadratic — use the Delta identity (Section 2.3):
    //   Delta = sqrt(mu_l^2 + 4 rho_l kappa^2 y^2 D tau_w)
    //   F1 = 2 tau_w / (mu_l + Delta),  mu_t = (Delta - mu_l)/2
    fpdtype_t kappa2_y2 = (fpdtype_t)${c['eq_kappa']**2}*y*y;
    fpdtype_t alphaPyy  = rho_l*kappa2_y2;
    fpdtype_t Delta     = sqrt(mu_l*mu_l + (fpdtype_t)4.0*alphaPyy*D*tau_w);
    fpdtype_t Delta_safe = fmax(Delta, (fpdtype_t)1e-12);
    fpdtype_t inv_Delta = (fpdtype_t)1.0/Delta_safe;
    fpdtype_t inv_mu_plus_Delta = (fpdtype_t)1.0/(mu_l + Delta);

    // dDelta/dT = [mu_l dmu/dT + 2 kappa^2 y^2 tau_w (D drho/dT + rho dD/dT)]/Delta
    fpdtype_t dDelta_dT = (mu_l*dmu_dT
                          + (fpdtype_t)2.0*kappa2_y2*tau_w
                            *(D*drho_dT + rho_l*dD_dT))*inv_Delta;

    // dDelta/dtau_w = (2 rho_l kappa^2 y^2 / Delta)*(D + tau_w dD/dtau)
    fpdtype_t dDelta_dtau = (fpdtype_t)2.0*alphaPyy*inv_Delta
                            *(D + tau_w*dD_dtau);

    // F1 partials
    dF1_dT   = -dudy*inv_mu_plus_Delta*(dmu_dT + dDelta_dT);
    dF1_dtau = (fpdtype_t)2.0*inv_mu_plus_Delta
               - dudy*inv_mu_plus_Delta*dDelta_dtau;

    // mu_t partials from mu_t = (Delta - mu_l)/2
    dmut_dT   = (fpdtype_t)0.5*(dDelta_dT   - dmu_dT);
    dmut_dtau = (fpdtype_t)0.5*dDelta_dtau;
% endif
% endif

    // --- F2 partials (master formula, Section 2.1) ---------------------
    // F2 = -(q_w + u tau_w)/D2;  D2 = alpha_T already computed
    fpdtype_t N2     = q_w + u*tau_w;
    fpdtype_t inv_D2 = (fpdtype_t)1.0/alpha_T;
    fpdtype_t cp_eff = (fpdtype_t)${c['cp']};
    fpdtype_t Pr_eff = (fpdtype_t)${c['eq_Pr']};
    fpdtype_t Prt_eff = (fpdtype_t)${c['eq_Prt']};
    fpdtype_t coef_F2 = N2*inv_D2*inv_D2*cp_eff;

    fpdtype_t dF2_du   = -tau_w*inv_D2;     // dmu_t/du = 0 simplifies
    fpdtype_t dF2_dT   = coef_F2*(dmu_dT/Pr_eff + dmut_dT/Prt_eff);
    fpdtype_t dF2_dtau = -u*inv_D2 + coef_F2/Prt_eff*dmut_dtau;

    // Explicit dependence of F1, F2 on the 2nd shooting parameter p2.
    //   isothermal: p2 = -q_w,  dF2/dp2 = 1/D2,  dF1/dp2 = 0.
    //   adiabatic : p2 = T_w (an IC parameter; q_w = 0 is baked in). Two
    //     chains can introduce explicit p2-dependence:
    //       (a) rho_w(T_w) = gamma p_m / ((gamma-1) c_p T_w)
    //             ALWAYS active (independent of viscosity law).
    //             dy+_wall/dp2  = -y+_wall/(2 T_w)  (rho_w in numerator)
    //             dy+_local/dp2 = +y+_local/(2 T_w) (rho_w in denominator
    //                                                via u_tau, only inside
    //                                                mixedmin2's local leg)
    //             dy+_semi/dp2  = 0                 (rho_w cancels)
    //       (b) mu_w(T_w) via Sutherland — only in y+ flavors containing
    //           mu_w (wall and wall-leg of mixedmin2). Constant-mu kills
    //           this chain.
    //     SA damping kills the whole correction at dD/dy+ = 0 (D doesn't
    //     depend on y+ for SA). mu_l, alpha_Pyy and muhat^JK all use LOCAL
    //     T / rho, not wall — so the only chain to mu_t is via D.
    fpdtype_t dF1_dp2 = (fpdtype_t)0.0;
% if thermal_bc == 'isothermal':
    fpdtype_t dF2_dp2 = inv_D2;
% else:
    fpdtype_t dF2_dp2 = (fpdtype_t)0.0;
% endif

% if thermal_bc == 'adiabatic' and eq_damping in ('van-driest', 'piomelli') and eq_yp_scaling in ('wall', 'mixedmin2'):
    // T_w-explicit correction in adiabatic mode. Always carries the rho_w
    // chain; carries the Sutherland mu_w chain only where applicable.
    {
        fpdtype_t Tw_safe = fmax(p2, (fpdtype_t)1e-3);
% if eq_mu_model == 'sutherland':
        fpdtype_t dmuw_dp2 = mu_w*(Tw_safe + (fpdtype_t)${3.0*c['eq_Sconst']})
                             /((fpdtype_t)2.0*Tw_safe
                               *(Tw_safe + (fpdtype_t)${c['eq_Sconst']}));
% endif

% if eq_yp_scaling == 'wall':
        // y+_wall ∝ sqrt(rho_w tau_w)/mu_w — both chains present.
        fpdtype_t dyp_dp2_e = -yp/((fpdtype_t)2.0*Tw_safe);
% if eq_mu_model == 'sutherland':
        dyp_dp2_e += -yp/mu_w*dmuw_dp2;
% endif
% else:  ## mixedmin2: hard-min — branch on the winning leg
        fpdtype_t dyp_dp2_e = (fpdtype_t)0.0;
        if (yp_mixed < yp_mixed2) {
            // wall leg (semi contributes 0): dy+_mixed/dp2 = 0.5 dy+_wall/dp2
            fpdtype_t dyw_dp2 = -yp_wall/((fpdtype_t)2.0*Tw_safe);
% if eq_mu_model == 'sutherland':
            dyw_dp2 += -yp_wall/mu_w*dmuw_dp2;
% endif
            dyp_dp2_e = (fpdtype_t)0.5*dyw_dp2;
        } else {
            // local leg (semi contributes 0): only rho_w chain.
            // y+_local ∝ rho_l sqrt(tau_w/rho_w)/mu_l — rho_w in denom via
            // u_tau, sign flipped vs y+_wall.
            dyp_dp2_e = (fpdtype_t)0.5*(yp_local/((fpdtype_t)2.0*Tw_safe));
        }
% endif

% if eq_damping == 'van-driest':
        fpdtype_t e_w = exp(-yp/(fpdtype_t)${c['eq_Aplus']});
        fpdtype_t dD_dyp_lcl = (fpdtype_t)2.0*sqrt(D)*e_w
                               /(fpdtype_t)${c['eq_Aplus']};
% else:  ## piomelli
        fpdtype_t dD_dyp_lcl = (fpdtype_t)3.0*yp*yp*((fpdtype_t)1.0 - D)
                               /(fpdtype_t)${c['eq_Aplus']**3};
% endif
        fpdtype_t dD_dp2_e = dD_dyp_lcl*dyp_dp2_e;

% if eq_mu_t_model == 'johnson-king':
        // muhat^JK = kappa y sqrt(rho_l tau_w) — no rho_w, no T_w. Only D
        // carries the T_w chain to mu_t.
        fpdtype_t dmut_dp2_e = muhat*dD_dp2_e;
        dF1_dp2 = -dudy/(mu_l + mu_t)*dmut_dp2_e;
% else:  ## prandtl + (VD or Piomelli) — re-use Delta_safe, alphaPyy
        // Delta = sqrt(mu_l^2 + 4 alphaPyy D tau_w); alpha_Pyy uses rho_l,
        // so only D depends on T_w. dDelta/dp2 = (2 alphaPyy tau_w/Delta) dD/dp2.
        fpdtype_t dDelta_dp2 = (fpdtype_t)2.0*alphaPyy*tau_w*dD_dp2_e
                               /Delta_safe;
        // mu_t = (Delta - mu_l)/2 with dmu_l/dp2 = 0
        fpdtype_t dmut_dp2_e = (fpdtype_t)0.5*dDelta_dp2;
        dF1_dp2 = -dudy/(mu_l + Delta_safe)*dDelta_dp2;
% endif

        // F2 master formula (Section 2.1) with dmu_l/dp2 = 0:
        //   dF2/dp2|explicit = (N2/D2^2)(cp/Prt) dmu_t/dp2|explicit
        dF2_dp2 += coef_F2/Prt_eff*dmut_dp2_e;
    }
% endif

    // --- Assemble A and B ----------------------------------------------
    fpdtype_t A00 = (fpdtype_t)0.0;   // dF1/du  = 0 universally
    fpdtype_t A01 = dF1_dT;
    fpdtype_t A10 = dF2_du;
    fpdtype_t A11 = dF2_dT;
    fpdtype_t B00 = dF1_dtau;
    fpdtype_t B01 = dF1_dp2;
    fpdtype_t B10 = dF2_dtau;
    fpdtype_t B11 = dF2_dp2;

    rhs[2] = A00*s[2] + A01*s[4] + B00;  // dS_uu/dy
    rhs[3] = A00*s[3] + A01*s[5] + B01;  // dS_uT/dy
    rhs[4] = A10*s[2] + A11*s[4] + B10;  // dS_Tu/dy
    rhs[5] = A10*s[3] + A11*s[5] + B11;  // dS_TT/dy
</%pyfr:macro>


<%pyfr:macro name='bc_common_flux_state'
             params='ul, gradul, artvisc, nl, magnl'
             externs='u_match, h_wm, warm_state'>
    // ===== Matching-point primitives and local viscosity ================
    fpdtype_t u_mvec[${ndims}];
% for i in range(ndims):
    u_mvec[${i}] = u_match[${i + 1}];
% endfor

    // Matching buffer slot 0 holds rho directly; T is derived from
    // the ideal-gas EOS:  T = gamma p / ((gamma - 1) c_p rho).
    fpdtype_t rho_m = u_match[0];
    fpdtype_t p_m   = u_match[${ndims + 1}];
    fpdtype_t T_m   = (fpdtype_t)${c['gamma']/((c['gamma'] - 1)*c['cp'])}
                      *p_m/rho_m;

    // Caller-provided constant viscosity (used by the macro when
    // eq_mu_model is 'constant'; Sutherland recomputes locally).
    fpdtype_t mu_c = (fpdtype_t)${c['eq_mu']};

    // Tangential matching velocity magnitude (nl is unit outward normal)
    fpdtype_t u_n_m = ${' + '.join(f'u_mvec[{i}]*nl[{i}]'
                                   for i in range(ndims))};
% for i in range(ndims):
    fpdtype_t ut_${i} = u_mvec[${i}] - u_n_m*nl[${i}];
% endfor
    fpdtype_t U_LES = sqrt(${' + '.join(f'ut_{i}*ut_{i}'
                                        for i in range(ndims))}
                           + (fpdtype_t)1e-15);

    // ===== Initial guess for the shooting parameters ====================
    // Warm-start from the previous call's converged (p1, p2) when valid
    // (warm_state[0] > 0); otherwise fall back to a cold-start heuristic:
    // tau_w ≈ rho_w (0.05 U_LES)^2 (5% of LES velocity, viscous-floor
    // protected), p2 = 0 (no heat flux) or T_m (≈ T_LES) per thermal BC.
    // In isothermal mode rho_w is known up front from the BL-thin EOS at
    // the prescribed T_w; in adiabatic mode T_w is itself unknown at this
    // point, so rho_m is used as a stand-in.
    fpdtype_t p1;
    fpdtype_t p2;
    if (warm_state[0] > (fpdtype_t)0.0) {
        p1 = warm_state[0];
        p2 = warm_state[1];
    } else {
% if thermal_bc == 'isothermal':
        fpdtype_t rho_w_init =
            (fpdtype_t)${c['gamma']/((c['gamma'] - 1)*c['cp']*c['eq_Tw'])}*p_m;
% else:
        fpdtype_t rho_w_init = rho_m;
% endif
        fpdtype_t utau_init = fmax((fpdtype_t)0.05*U_LES,
                                   mu_c/(rho_w_init*h_wm));
        p1 = rho_w_init*utau_init*utau_init;
% if thermal_bc == 'isothermal':
        p2 = (fpdtype_t)0.0;
% else:
        p2 = T_m;
% endif
    }

    // ===== Outer loop: Newton shooting with backtracking line search ====
    //
    // Each iteration runs one ODE integration. After integrating we either
    //   (i) accept the current (p1, p2) and take a new Newton step, or
    //   (ii) revert to the last accepted iterate and halve the step (up to
    //        a fixed number of backtracks).
    // Total iterations are bounded by eq_newton_max_iters; backtracks
    // count against this budget.
    fpdtype_t p1_prev = p1, p2_prev = p2;
    fpdtype_t dp1 = 0.0, dp2 = 0.0;
    fpdtype_t alpha_ls = 1.0;
    fpdtype_t prev_resid_norm = (fpdtype_t)1e15;
    int n_backtrack = 0;

    fpdtype_t s[6];
    fpdtype_t resid_u = 0.0, resid_T = 0.0, resid_norm = 0.0;
    fpdtype_t scale_u = fmax(fabs(U_LES), (fpdtype_t)1e-14);
    fpdtype_t scale_T = fmax(fabs(T_m),   (fpdtype_t)1e-14);

    for (int newton_it = 0; newton_it < ${eq_newton_max_iters}; ++newton_it) {
        // --- Wall density and viscosity for this iterate ---------------
        // T_w is c['eq_Tw'] (isothermal) or p2 (adiabatic). In adiabatic
        // both rho_w and (under Sutherland) mu_w change every Newton iter
        // because p2 is the shooting variable.
% if thermal_bc == 'isothermal':
        fpdtype_t Tw_now = (fpdtype_t)${c['eq_Tw']};
% else:
        fpdtype_t Tw_now = p2;
% endif
        fpdtype_t Tw_safe = fmax(Tw_now, (fpdtype_t)1e-3);

        // Wall density from the BL-thin EOS: p(y) ≈ p_m so
        //   rho_w = gamma p_m / ((gamma - 1) c_p T_w).
        // Consistent with rho_local(T) used inside eq_compute_rhs.
        fpdtype_t rho_w = (fpdtype_t)${c['gamma']/((c['gamma'] - 1)*c['cp'])}
                          *p_m/Tw_safe;

% if eq_mu_model == 'sutherland':
        // mu_w = mu_ref (T_w/T_ref)^(3/2) (T_ref+S)/(T_w+S); x*sqrt(x) form
        fpdtype_t Tw_rat = Tw_safe/(fpdtype_t)${c['eq_Tref']};
        fpdtype_t mu_w = (fpdtype_t)${c['eq_mu_ref']}*Tw_rat*sqrt(Tw_rat)
              *(fpdtype_t)${c['eq_Tref'] + c['eq_Sconst']}
              /(Tw_safe + (fpdtype_t)${c['eq_Sconst']});
% else:
        fpdtype_t mu_w = mu_c;
% endif

        // --- IC at the wall (y = 0) -------------------------------------
        s[0] = 0.0;                              // u(0) = 0
% if thermal_bc == 'isothermal':
        s[1] = (fpdtype_t)${c['eq_Tw']};         // T(0) = T_w_given
        s[2] = 0.0; s[3] = 0.0;                  // d(u)/d(p1,p2) = 0
        s[4] = 0.0; s[5] = 0.0;                  // d(T)/d(p1,p2) = 0
% else:
        s[1] = p2;                               // T(0) = T_w (shooting)
        s[2] = 0.0; s[3] = 0.0;
        s[4] = 0.0; s[5] = 1.0;                  // dT/d(T_w) = 1
% endif
        fpdtype_t pvec[2] = {p1, p2};

        // --- Bogacki-Shampine RK23 adaptive integration, y: 0 → h_wm ----
        // Embedded error: 3rd-order solution vs 2nd-order estimate. PI
        // step-size control, FSAL (reuse stage-4 RHS as next step's k1).
        fpdtype_t y_pos = 0.0;
        fpdtype_t h = h_wm*(fpdtype_t)0.01;
        fpdtype_t prev_err_norm = 0.0;
        fpdtype_t k1[6], k2[6], k3[6], k4[6];
        fpdtype_t s_temp[6], s_new[6];

        // Initial k1
        ${pyfr.expand('eq_compute_rhs', 'y_pos', 's', 'pvec',
                      'p_m', 'rho_w', 'mu_w', 'mu_c', 'k1')};

        for (int step = 0; step < ${eq_ode_max_steps}; ++step) {
            if (y_pos >= h_wm) break;
            if (y_pos + h > h_wm) h = h_wm - y_pos;

            // Stage 2: f(y + h/2, s + (h/2) k1)
            for (int i = 0; i < 6; ++i)
                s_temp[i] = s[i] + (fpdtype_t)0.5*h*k1[i];
            {
                fpdtype_t y_stg = y_pos + (fpdtype_t)0.5*h;
                ${pyfr.expand('eq_compute_rhs', 'y_stg', 's_temp', 'pvec',
                              'p_m', 'rho_w', 'mu_w', 'mu_c', 'k2')};
            }

            // Stage 3: f(y + 3h/4, s + (3h/4) k2)
            for (int i = 0; i < 6; ++i)
                s_temp[i] = s[i] + (fpdtype_t)0.75*h*k2[i];
            {
                fpdtype_t y_stg = y_pos + (fpdtype_t)0.75*h;
                ${pyfr.expand('eq_compute_rhs', 'y_stg', 's_temp', 'pvec',
                              'p_m', 'rho_w', 'mu_w', 'mu_c', 'k3')};
            }

            // 3rd-order solution: y1 = s + h (2/9 k1 + 1/3 k2 + 4/9 k3)
            for (int i = 0; i < 6; ++i) {
                s_new[i] = s[i] + h*( (fpdtype_t)(2.0/9.0)*k1[i]
                                    + (fpdtype_t)(1.0/3.0)*k2[i]
                                    + (fpdtype_t)(4.0/9.0)*k3[i]);
            }

            // FSAL Stage 4 at y + h with y1
            {
                fpdtype_t y_stg = y_pos + h;
                ${pyfr.expand('eq_compute_rhs', 'y_stg', 's_new', 'pvec',
                              'p_m', 'rho_w', 'mu_w', 'mu_c', 'k4')};
            }

            // Embedded error coefficients: b - b_hat
            // = (2/9 - 7/24, 1/3 - 1/4, 4/9 - 1/3, -1/8)
            // = (-5/72,      1/12,      1/9,       -1/8)
            fpdtype_t err_sq = 0.0;
            for (int i = 0; i < 6; ++i) {
                fpdtype_t e_i = h*( (fpdtype_t)(-5.0/72.0)*k1[i]
                                  + (fpdtype_t)( 1.0/12.0)*k2[i]
                                  + (fpdtype_t)( 1.0/ 9.0)*k3[i]
                                  + (fpdtype_t)(-1.0/ 8.0)*k4[i]);
                fpdtype_t sc = (fpdtype_t)${eq_ode_atol}
                             + (fpdtype_t)${eq_ode_rtol}
                               *fmax(fabs(s[i]), fabs(s_new[i]));
                err_sq += (e_i/sc)*(e_i/sc);
            }
            fpdtype_t err_norm = sqrt(err_sq/(fpdtype_t)6.0);

            if (err_norm <= (fpdtype_t)1.0) {
                // Accept
                y_pos += h;
                for (int i = 0; i < 6; ++i) {
                    s[i]  = s_new[i];
                    k1[i] = k4[i];                          // FSAL
                }

                // PI controller: facmax = 5, facmin = 0.2, safety = 0.9
                // α = 0.7/(p+1), β = 0.4/(p+1) with p = 2 (embedded order)
                fpdtype_t safe_err = fmax(err_norm, (fpdtype_t)1e-10);
                fpdtype_t fac;
                if (prev_err_norm > (fpdtype_t)0.0) {
                    fac = (fpdtype_t)0.9
                          *pow(safe_err,      (fpdtype_t)(-0.7/3.0))
                          *pow(prev_err_norm, (fpdtype_t)( 0.4/3.0));
                } else {
                    fac = (fpdtype_t)0.9
                          *pow(safe_err, (fpdtype_t)(-1.0/3.0));
                }
                fac = fmax((fpdtype_t)0.2, fmin((fpdtype_t)5.0, fac));
                h *= fac;
                prev_err_norm = safe_err;
            } else {
                // Reject — shrink h and retry without advancing y
                fpdtype_t fac = (fpdtype_t)0.9
                                *pow(err_norm, (fpdtype_t)(-1.0/3.0));
                fac = fmax((fpdtype_t)0.1, fac);
                h *= fac;
            }
        }

        // --- Residual and convergence ----------------------------------
        resid_u = s[0] - U_LES;
        resid_T = s[1] - T_m;
        resid_norm = sqrt(  (resid_u/scale_u)*(resid_u/scale_u)
                          + (resid_T/scale_T)*(resid_T/scale_T) );
        if (resid_norm < (fpdtype_t)${eq_newton_rtol}) break;

        // --- Backtracking: if residual got worse, revert and halve -----
        if (newton_it > 0 && resid_norm > prev_resid_norm
            && n_backtrack < 3) {
            alpha_ls *= (fpdtype_t)0.5;
            p1 = p1_prev + alpha_ls*dp1;
            p2 = p2_prev + alpha_ls*dp2;
            n_backtrack++;
            continue;
        }

        // --- Accept current iterate; compute next Newton step ----------
        // J = [[s[2], s[3]], [s[4], s[5]]] is the residual Jacobian.
        // dp = -J^{-1} r.
        fpdtype_t detJ = s[2]*s[5] - s[3]*s[4];
        fpdtype_t safe_det = (fabs(detJ) > (fpdtype_t)1e-15)
                             ? detJ : (fpdtype_t)1e-15;
        fpdtype_t inv_detJ = (fpdtype_t)1.0/safe_det;
        dp1 = -inv_detJ*( s[5]*resid_u - s[3]*resid_T);
        dp2 = -inv_detJ*(-s[4]*resid_u + s[2]*resid_T);

        p1_prev = p1;  p2_prev = p2;
        alpha_ls = (fpdtype_t)1.0;
        n_backtrack = 0;
        prev_resid_norm = resid_norm;
        p1 += dp1;
        p2 += dp2;
    }

    // Warm-start: stash the latest (p1, p2) for the next call. We always
    // overwrite — even when Newton hits max-iters without converging the
    // current iterate is at least "in the ballpark" and serves as a
    // better starting point than the cold-start heuristic.
    warm_state[0] = p1;
    warm_state[1] = p2;

    // ===== Apply (tau_w, q_w) as the wall flux ==========================
    // Shooting convention (document): p2 = -q_w (isothermal) so q_w = -p2.
    fpdtype_t tau_w = p1;
% if thermal_bc == 'isothermal':
    fpdtype_t q_w   = -p2;
% else:
    fpdtype_t q_w   = 0.0;
% endif

    // Riemann mirror ghost from bc_rsolve_state (negated momentum, total
    // energy preserved).
    fpdtype_t ur[${nvars}];
    ${pyfr.expand('bc_rsolve_state', 'ul', 'nl', 'ur')};

    fpdtype_t ficomm[${nvars}];
    ${pyfr.expand('rsolve', 'ul', 'ur', 'nl', 'ficomm')};

    // Flux: inviscid (Riemann) + tangential wall shear stress + wall heat flux
    fpdtype_t ut_mag_inv = 1.0/U_LES;
    ul[0] = magnl*ficomm[0];
% for i in range(ndims):
    ul[${i + 1}] = magnl*(ficomm[${i + 1}] + tau_w*ut_${i}*ut_mag_inv);
% endfor
    ul[${nvars - 1}] = magnl*(ficomm[${nvars - 1}] - q_w);
</%pyfr:macro>
