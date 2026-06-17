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

<%pyfr:macro name='bc_common_flux_state'
             params='ul, gradul, artvisc, nl, magnl'
             externs='u_match, h_wm, warm_state, gq_y, gq_w'>
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
    // GQWM is incompressible by construction; matching-buffer density and
    // pressure slots are ignored, only u_match[1..ndims] is used.
    fpdtype_t rho_w = ul[0];
    fpdtype_t mu_w  = (fpdtype_t)${c['gq_mu']};
    fpdtype_t nu_w  = mu_w/rho_w;

    // ----- Initial guess (warm-start, else heuristic) -------------------
    fpdtype_t gq_utau0;
    if (warm_state[0] > (fpdtype_t)0.0) {
        gq_utau0 = warm_state[0];
    } else {
        gq_utau0 = fmax((fpdtype_t)0.05*U_LES, nu_w/h_wm);
    }
    fpdtype_t gq_utau1 = (fpdtype_t)1.1*gq_utau0;
    fpdtype_t gq_integ, gq_f0, gq_f1;

    // ----- Secant on F(u_tau) = 0 ---------------------------------------
    // gq_y[qi], gq_w[qi] are per-fpt mapped quadrature nodes/weights
    // (exponential transformation y = h_wm (exp(xi+1)-1)/(exp(2)-1)),
    // precomputed once at BC setup.
    // Inline integrand evaluator F(utau): sum_qi w_qi * 2*utau^2 /
    //   (nu (1 + sqrt(1 + 4*lm^2))), with lm = kappa*y+ (1-exp(-y+/A+)),
    //   y+ = gq_y[qi]*utau/nu.
    #define GQ_EVAL(utau, out) do {                                          \
        out = 0;                                                              \
        for (int qi = 0; qi < ${c['gq_nquad']}; ++qi) {                       \
            fpdtype_t _yp = gq_y[qi]*(utau)/nu_w;                             \
            fpdtype_t _lm = (fpdtype_t)${c['gq_kappa']}*_yp                   \
                            *(1.0 - exp(-_yp/(fpdtype_t)${c['gq_Aplus']}));   \
            out += gq_w[qi]*2*(utau)*(utau)                                   \
                   /(nu_w*(1.0 + sqrt(1.0 + 4*_lm*_lm)));                     \
        }                                                                     \
    } while (0)

    GQ_EVAL(gq_utau0, gq_integ); gq_f0 = gq_integ - U_LES;
    GQ_EVAL(gq_utau1, gq_integ); gq_f1 = gq_integ - U_LES;

    // Secant iterations capped at gq_n_iter; early exit on relative
    // change in u_tau below gq_rel_tol.
    for (int it = 0; it < ${c['gq_n_iter']}; ++it) {
        fpdtype_t _df = gq_f1 - gq_f0;
        fpdtype_t _new = gq_utau1 - gq_f1*(gq_utau1 - gq_utau0)
                         /(fabs(_df) > (fpdtype_t)1e-15 ? _df
                                                       : (fpdtype_t)1e-15);
        fpdtype_t _du = _new - gq_utau1;
        gq_utau0 = gq_utau1; gq_f0 = gq_f1;
        gq_utau1 = fmax(_new, (fpdtype_t)1e-14);
        GQ_EVAL(gq_utau1, gq_integ); gq_f1 = gq_integ - U_LES;
        if (fabs(_du) < (fpdtype_t)${c['gq_rel_tol']}*gq_utau1) break;
    }
    #undef GQ_EVAL

    fpdtype_t u_tau = fmax(gq_utau1, (fpdtype_t)0);

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
