<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%include file='pyfr.solvers.navstokes.kernels.bcs.common'/>

<%include file='pyfr.solvers.euler.kernels.rsolvers.${rsolver}'/>

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
## Algebraic equilibrium wall model.
##   law = 'log-law'   : U+ = (1/kappa) ln(y+) + B
##   law = 'spalding'  : y+ = U+ + exp(-kappa*B) * [exp(kU+) - 1 - kU+
##                                                 - (kU+)^2/2 - (kU+)^3/6]
##   law = 'reichardt' : U+ = (1/kappa) ln(1 + kappa*y+)
##                          + R [1 - exp(-y+/11) - (y+/11) exp(-y+/3)]
##
## Each is a 1D root-finder for u_tau given U_LES at y = h_wm and ν_w.
## Newton iteration capped at alg_n_iter steps with a relative-update
## convergence test |du| < alg_rel_tol * u_tau. Algebraic wall models are
## inherently crude, so a loose tolerance (default 1e-3) is plenty —
## early exit is more valuable than warp-lockstep uniformity. Adiabatic
## only.
##
<%pyfr:macro name='bc_common_flux_state'
             params='ul, gradul, artvisc, nl, magnl'
             externs='u_match, h_wm, warm_state'>
    // ----- Matching-point wall-tangential velocity ----------------------
    fpdtype_t u_n_m = ${' + '.join(f'u_match[{i + 1}]*nl[{i}]'
                                   for i in range(ndims))};
% for i in range(ndims):
    fpdtype_t ut_${i} = u_match[${i + 1}] - u_n_m*nl[${i}];
% endfor
    fpdtype_t U_LES = sqrt(${' + '.join(f'ut_{i}*ut_{i}'
                                        for i in range(ndims))}
                           + (fpdtype_t)1e-20);

    // ----- Wall density and kinematic viscosity --------------------------
    fpdtype_t rho_w = ul[0];
    fpdtype_t mu_w  = (fpdtype_t)${c['alg_mu']};
    fpdtype_t nu_w  = mu_w/rho_w;

    // ----- Initial guess (warm-start, else heuristic) -------------------
    fpdtype_t u_tau;
    if (warm_state[0] > (fpdtype_t)0.0) {
        u_tau = warm_state[0];
    } else {
        u_tau = fmax((fpdtype_t)0.05*U_LES, nu_w/h_wm);
    }

    // ----- Newton on f(u_tau) = 0 ---------------------------------------
    // Capped at alg_n_iter iterations; breaks early on relative-update
    // convergence |du| < alg_rel_tol * u_tau. Warm-started cases
    // typically converge in 1-3 iterations.
    for (int it = 0; it < ${c['alg_n_iter']}; ++it) {
        fpdtype_t Up = U_LES/u_tau;
        fpdtype_t yp = h_wm*u_tau/nu_w;
% if alg_law == 'log-law':
        // f = Up - (1/kappa) ln(yp) - B
        // f' = -U_LES/u_tau^2 - 1/(kappa*u_tau)
        fpdtype_t f_val = Up
                         - ((fpdtype_t)1.0/(fpdtype_t)${c['alg_kappa']})
                           *log(yp)
                         - (fpdtype_t)${c['alg_B']};
        fpdtype_t f_pr  = -U_LES/(u_tau*u_tau)
                          - (fpdtype_t)1.0
                            /((fpdtype_t)${c['alg_kappa']}*u_tau);
% elif alg_law == 'spalding':
        // f = yp - Up - exp(-kappa*B)*[exp(X) - 1 - X - X^2/2 - X^3/6]
        // X = kappa*Up;  dX/du_tau = -X/u_tau
        fpdtype_t X = (fpdtype_t)${c['alg_kappa']}*Up;
        fpdtype_t exp_X = exp(X);
        fpdtype_t expmkB = (fpdtype_t)${c['alg_expmkB']};
        fpdtype_t bracket = exp_X - (fpdtype_t)1.0 - X
                            - (fpdtype_t)0.5*X*X
                            - (fpdtype_t)(1.0/6.0)*X*X*X;
        fpdtype_t f_val = yp - Up - expmkB*bracket;
        // d[bracket]/du_tau = (exp(X) - 1 - X - X^2/2) * dX/du_tau
        //                   = (exp(X) - 1 - X - X^2/2) * (-X/u_tau)
        fpdtype_t d_brk = exp_X - (fpdtype_t)1.0 - X - (fpdtype_t)0.5*X*X;
        fpdtype_t f_pr  = h_wm/nu_w + U_LES/(u_tau*u_tau)
                          + expmkB*X*d_brk/u_tau;
% else:  ## reichardt
        // f = Up - (1/kappa) ln(1 + kappa*yp)
        //        - C [1 - exp(-yp/B1) - (yp/B1) exp(-yp/B2)]
        fpdtype_t one_p_kyp = (fpdtype_t)1.0
                              + (fpdtype_t)${c['alg_kappa']}*yp;
        fpdtype_t e1 = exp(-yp/(fpdtype_t)${c['alg_B1']});
        fpdtype_t e2 = exp(-yp/(fpdtype_t)${c['alg_B2']});
        fpdtype_t f_val = Up
                         - ((fpdtype_t)1.0/(fpdtype_t)${c['alg_kappa']})
                           *log(one_p_kyp)
                         - (fpdtype_t)${c['alg_C']}
                           *((fpdtype_t)1.0 - e1
                             - (yp/(fpdtype_t)${c['alg_B1']})*e2);
        // dUp_rhs/dyp = 1/(1 + kappa*yp)
        //              + C [exp(-yp/B1)/B1 + exp(-yp/B2)*(yp/(B1*B2) - 1/B1)]
        fpdtype_t dUp_dyp =
              (fpdtype_t)1.0/one_p_kyp
            + (fpdtype_t)${c['alg_C']}
              *(e1/(fpdtype_t)${c['alg_B1']}
                + e2*(yp/(fpdtype_t)${c['alg_B1']*c['alg_B2']}
                      - (fpdtype_t)1.0/(fpdtype_t)${c['alg_B1']}));
        fpdtype_t f_pr = -U_LES/(u_tau*u_tau) - dUp_dyp*h_wm/nu_w;
% endif

        // Newton step (clamp u_tau strictly positive). Break early on
        // relative-update convergence.
        fpdtype_t du = f_val/f_pr;
        u_tau = fmax(u_tau - du, (fpdtype_t)1e-14);
        if (fabs(du) < (fpdtype_t)${c['alg_rel_tol']}*u_tau) break;
    }

    // ----- Wall shear stress and warm-start save ------------------------
    fpdtype_t tau_w = rho_w*u_tau*u_tau;
    warm_state[0] = u_tau;

    // ----- Apply wall flux ----------------------------------------------
    // Riemann mirror ghost from bc_rsolve_state (negated momentum, total
    // energy preserved).
    fpdtype_t ur[${nvars}];
    ${pyfr.expand('bc_rsolve_state', 'ul', 'nl', 'ur')};

    fpdtype_t ficomm[${nvars}];
    ${pyfr.expand('rsolve', 'ul', 'ur', 'nl', 'ficomm')};

    // Inviscid (Riemann) + tangential wall shear stress; adiabatic energy
    fpdtype_t ut_mag_inv = (fpdtype_t)1.0/U_LES;
    ul[0] = magnl*ficomm[0];
% for i in range(ndims):
    ul[${i + 1}] = magnl*(ficomm[${i + 1}] + tau_w*ut_${i}*ut_mag_inv);
% endfor
    ul[${nvars - 1}] = magnl*ficomm[${nvars - 1}];
</%pyfr:macro>
