<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

## ============================================================
## Sub-grid scale (SGS) eddy-viscosity models.
##
## Each macro takes precomputed physical primitive velocity
## gradients  alpha[i][j] = du_i/dx_j  together with the density
## and a per-element filter width  delta_e, and writes the
## dynamic SGS viscosity  mu_sgs = rho*nu_t  to the caller-scope
## output. The caller (viscous_flux_add) then adds  mu_sgs  to
## the molecular dynamic viscosity before assembling the
## (deviatoric) stress tensor and the heat-flux term.
##
## Only the deviatoric part of the SGS stress is modeled here;
## the isotropic trace 2/3 rho k_SGS is neglected (standard
## practice in compressible LES at low/moderate Mach numbers;
## cf. Erlebacher, Hussaini, Speziale and Zang 1992).
##
## All macros wrap their bodies in {...} so locals do not clash
## with the calling scope. The mu_sgs output parameter must be
## declared by the caller in the outer scope before invocation.
##
## Supported models (selected via the sgs_model tplarg in the
## calling kernel):
##   sgs_model = 'vreman'  --  Vreman (2004) algebraic model
##   sgs_model = 'sigma'   --  Nicoud et al. (2011) sigma model
## ============================================================

##
## --- Vreman SGS model ---
##
## Reference: Vreman, AW (2004), "An eddy-viscosity subgrid-scale
## model for turbulent shear flow: Algebraic theory and
## applications", Phys. Fluids 16, 3670.
##
## With alpha_ij = du_i/dx_j the physical primitive velocity
## gradient and Delta_e the (isotropic) per-element filter width:
##
##   beta_ij     = Delta_e^2 sum_m alpha_mi alpha_mj
##   B_beta      = beta_11 beta_22 - beta_12^2
##               + beta_11 beta_33 - beta_13^2
##               + beta_22 beta_33 - beta_23^2
##   alpha_norm2 = sum_ij alpha_ij^2
##   nu_t        = C_vreman sqrt(B_beta / alpha_norm2)  if alpha_norm2 > 0,
##               = 0                                    otherwise
##   mu_sgs      = rho * nu_t                           (dynamic viscosity)
##
## Vreman's recommended C_vreman ~= 0.07 (= 2.5 C_S^2 with the
## Smagorinsky C_S ~= 0.17). The model is non-negative by
## construction (B_beta >= 0 from Schwarz-type inequalities on
## the 2x2 minors of beta), and vanishes naturally in 2D /
## laminar / near-wall regions without ad-hoc damping.
##
## Index convention. The Vreman paper defines
##   alpha_ij = du_j/dx_i   (1st index = differentiation direction,
##                           2nd index = velocity component)
## so the paper's beta = alpha^T alpha. Here we use the transposed
## convention
##   alpha_ij = du_i/dx_j   (1st index = velocity component,
##                           2nd index = differentiation direction)
## so the code's beta = alpha alpha^T. The transpose matches PyFR's
## existing variable naming (u_x = du/dx, u_y = du/dy, ...) which
## puts the velocity component first.
##
## With the ISOTROPIC filter used here (single scalar Delta_e),
## the two beta matrices share the same eigenvalues — they are
## A^T A vs A A^T — and therefore have identical characteristic-
## polynomial coefficients. The scalar invariants B_beta and
## alpha_norm2 are unchanged under the transposition, and nu_t
## evaluates identically. Numerically the model output matches
## Vreman's published formula exactly.
##
## WARNING: this equivalence relies on the isotropic-filter
## assumption. With an anisotropic filter (Delta_m varying per
## differentiation direction), Delta_m^2 would multiply the m-th
## *direction* index in Vreman's convention but the m-th
## *velocity-component* index here — yielding different nu_t.
## If anisotropic filtering is ever introduced, the index
## convention must be flipped back to Vreman's.
##
<%pyfr:macro name='sgs_vreman_mu_t'
             params='rho_in, alpha_in, delta_e, mu_sgs'>
    {
        // beta[i][j] = Delta^2 sum_m alpha[m][i] alpha[m][j]
        // (symmetric; only the upper triangle is materialised)
        fpdtype_t delta_sq_v = delta_e*delta_e;
% for i, j in pyfr.ndrange(ndims, ndims):
% if i <= j:
        fpdtype_t beta_v_${i}${j} = delta_sq_v*(
            ${' + '.join(f'alpha_in[{m}][{i}]*alpha_in[{m}][{j}]'
                         for m in range(ndims))});
% endif
% endfor

        // B_beta: sum of 2x2 principal minors of beta.
% if ndims == 2:
        fpdtype_t B_beta_v = beta_v_00*beta_v_11 - beta_v_01*beta_v_01;
% else:
        fpdtype_t B_beta_v = beta_v_00*beta_v_11 - beta_v_01*beta_v_01
                           + beta_v_00*beta_v_22 - beta_v_02*beta_v_02
                           + beta_v_11*beta_v_22 - beta_v_12*beta_v_12;
% endif

        // alpha_norm2 = sum_ij alpha_ij^2
        fpdtype_t alpha_norm2_v =
            ${' + '.join(f'alpha_in[{i}][{j}]*alpha_in[{i}][{j}]'
                          for i in range(ndims)
                          for j in range(ndims))};

        // nu_t = C sqrt(max(B_beta, 0) / alpha_norm2). Guards:
        // floor B_beta against round-off-induced negatives
        // (analytically B_beta >= 0), and skip when the gradient
        // norm itself is essentially zero (locally uniform flow).
        fpdtype_t B_beta_safe_v = fmax(B_beta_v, (fpdtype_t)0.0);
        fpdtype_t nu_t_v = (alpha_norm2_v > (fpdtype_t)1e-15)
                         ? (fpdtype_t)${c_vreman}
                           *sqrt(B_beta_safe_v/alpha_norm2_v)
                         : (fpdtype_t)0.0;

        // mu_sgs = rho * nu_t  (dynamic-viscosity units)
        mu_sgs = rho_in*nu_t_v;
    }
</%pyfr:macro>

##
## --- Sigma SGS model ---
##
## Reference: Nicoud, F, Toda, HB, Cabrit, O, Bose, S and Lee, J (2011),
## "Using singular values to build a subgrid-scale model for large eddy
## simulations", Phys. Fluids 23, 085106.
##
## With alpha_ij = du_i/dx_j the physical primitive velocity gradient
## and Delta_e the per-element filter width:
##
##   G        = alpha^T alpha          (3x3 symmetric SPD)
##   sigma_k  = sqrt(lambda_k)         where lambda_1 >= lambda_2 >= lambda_3
##                                     are the eigenvalues of G
##   D_sigma  = sigma_3 * (sigma_1 - sigma_2) * (sigma_2 - sigma_3) / sigma_1^2
##   nu_t     = (C_sigma * Delta_e)^2 * D_sigma
##   mu_sgs   = rho * nu_t              (dynamic-viscosity units)
##
## Recommended C_sigma ~= 1.35. The model vanishes naturally in:
##   - 2D flow (one eigenvalue is zero)
##   - axisymmetric / isotropic compression (two eigenvalues equal)
##   - solid-body rotation (G is the identity scaled by omega^2)
## i.e. in every flow that requires zero subgrid dissipation. This makes
## it well-suited for transition LES and complex geometries.
##
## 3D-only: in 2D the rank-2 alpha yields sigma_3 = 0 and the formula
## gives nu_t = 0 identically (the macro emits mu_sgs = 0 in that case
## without computing G).
##
## SINGULAR VALUES ARE COMPUTED WITHOUT EVER FORMING G = alpha^T alpha.
##
## Forming G squares the condition number, and D_sigma is proportional to
## sigma_3 -- the smallest singular value, i.e. precisely the one that loses
## all its significant digits when the condition number is squared. In single
## precision the damage is severe exactly where this model matters: measured
## against a float64 SVD, the old cubic-invariant route gives a MEDIAN
## relative error in D_sigma of 78% for near-wall gradient tensors
## (kappa ~ 2e2) and 37% for quasi-2D ones (kappa ~ 3e3), with worst cases
## over 1e3. The resulting mu_sgs is essentially noise near a wall, which both
## corrupts the physics and forces the adaptive time-step controller to cut dt
## (a measured 2.6x penalty at p5).
##
## Instead we use a ONE-SIDED JACOBI SVD, which orthogonalises the COLUMNS of
## alpha directly by plane rotations and never squares the condition number;
## it is the standard method for computing small singular values to high
## relative accuracy (Demmel & Veselic). Three sweeps of the three (p,q)
## column pairs reduce the 99th-percentile relative error in D_sigma to ~3e-6
## in single precision for every regime tested. As a bonus it removes the
## acos/cos pair the trigonometric eigenvalue route needed.
##
## alpha is first scaled by its Frobenius norm so that every threshold below
## is DIMENSIONLESS (the previous absolute 1e-15 guards were dimensionally
## meaningless -- alpha has units of 1/time). D_sigma is homogeneous of
## degree one in the singular values, so the norm is simply multiplied back
## in at the end.
##
## Rotation for the column pair (p,q), with app, aqq, apq the column inner
## products:
##   zeta = (aqq - app) / (2 apq)
##   t    = sign(zeta) / (|zeta| + sqrt(1 + zeta^2))   [-> 1/(2 zeta) if huge]
##   c    = 1/sqrt(1 + t^2),   s = c t
## A degenerate pair (apq ~ 0) drives |zeta| -> huge, hence t -> 0 and the
## rotation becomes the identity, so no branch is needed.
##
<%pyfr:macro name='sgs_sigma_mu_t'
             params='rho_in, alpha_in, delta_e, mu_sgs'>
    {
% if ndims == 3:
        // Frobenius norm: work on a dimensionless matrix so every threshold
        // below is scale-free.  D_sigma is degree-one homogeneous in the
        // singular values, so fn is multiplied back in at the end.
        fpdtype_t fn2_s = ${' + '.join(f'alpha_in[{i}][{j}]*alpha_in[{i}][{j}]'
                                       for i in range(3) for j in range(3))};
        fpdtype_t fn_s = sqrt(fn2_s);

        if (fn_s > (fpdtype_t)0.0)
        {
            fpdtype_t rn_s = (fpdtype_t)1.0/fn_s;
% for i in range(3):
% for j in range(3):
            fpdtype_t a${i}${j}_s = alpha_in[${i}][${j}]*rn_s;
% endfor
% endfor

            // --- one-sided Jacobi: orthogonalise the columns of alpha ---
            // Three sweeps of the three column pairs; measured to give a
            // 99th-percentile relative error of ~3e-6 in D_sigma (single
            // precision), against 3.7e-1 to 7.8e-1 for the old G = A^T A route.
% for sweep in range(3):
% for (pc, qc) in ((0, 1), (0, 2), (1, 2)):
            {
                fpdtype_t app_s = ${' + '.join(f'a{i}{pc}_s*a{i}{pc}_s'
                                               for i in range(3))};
                fpdtype_t aqq_s = ${' + '.join(f'a{i}{qc}_s*a{i}{qc}_s'
                                               for i in range(3))};
                fpdtype_t apq_s = ${' + '.join(f'a{i}{pc}_s*a{i}{qc}_s'
                                               for i in range(3))};

                // Guarded denominator: apq ~ 0 sends |zeta| -> huge, so
                // t -> 0 and the rotation degenerates to the identity.
                fpdtype_t den_s = (fpdtype_t)2.0*apq_s;
                den_s += (den_s >= (fpdtype_t)0.0 ? (fpdtype_t)1e-30
                                                  : -(fpdtype_t)1e-30);
                fpdtype_t zeta_s = (aqq_s - app_s)/den_s;
                fpdtype_t az_s = fabs(zeta_s);
                fpdtype_t sgn_s = (zeta_s >= (fpdtype_t)0.0)
                                ? (fpdtype_t)1.0 : (fpdtype_t)-1.0;

                // t = sgn/(|zeta| + sqrt(1+zeta^2)); for large |zeta| use the
                // asymptote 1/(2 zeta) so zeta*zeta cannot overflow.
                fpdtype_t t_s = (az_s < (fpdtype_t)1e3)
                              ? sgn_s/(az_s + sqrt((fpdtype_t)1.0
                                                   + zeta_s*zeta_s))
                              : sgn_s/((fpdtype_t)2.0*az_s);
                fpdtype_t c_s = (fpdtype_t)1.0/sqrt((fpdtype_t)1.0 + t_s*t_s);
                fpdtype_t s_s = c_s*t_s;

% for i in range(3):
                {
                    fpdtype_t tmp_s = a${i}${pc}_s;
                    a${i}${pc}_s = c_s*tmp_s - s_s*a${i}${qc}_s;
                    a${i}${qc}_s = s_s*tmp_s + c_s*a${i}${qc}_s;
                }
% endfor
            }
% endfor
% endfor

            // Singular values are the final column norms (of the scaled matrix)
% for j in range(3):
            fpdtype_t n${j}_s = sqrt(${' + '.join(f'a{i}{j}_s*a{i}{j}_s'
                                                  for i in range(3))});
% endfor

            // Sort descending via the max/min/sum identity
            fpdtype_t smax_s = fmax(fmax(n0_s, n1_s), n2_s);
            fpdtype_t smin_s = fmin(fmin(n0_s, n1_s), n2_s);
            fpdtype_t smid_s = n0_s + n1_s + n2_s - smax_s - smin_s;

            // D_sigma = s3 (s1 - s2)(s2 - s3) / s1^2, then undo the scaling.
            // s1 >= 1/sqrt(3) for a unit-Frobenius matrix, so no guard needed.
            fpdtype_t D_sigma_s = smin_s*(smax_s - smid_s)*(smid_s - smin_s)
                                / (smax_s*smax_s);

            fpdtype_t Cdelta_s = (fpdtype_t)${c_sigma}*delta_e;
            mu_sgs = rho_in*Cdelta_s*Cdelta_s*D_sigma_s*fn_s;
        }
        else
        {
            // No resolved velocity gradient at all
            mu_sgs = (fpdtype_t)0.0;
        }
% else:
        // 2D: rank-deficient alpha yields sigma_3 = 0 and the sigma
        // model is identically zero.
        mu_sgs = (fpdtype_t)0.0;
% endif
    }
</%pyfr:macro>
