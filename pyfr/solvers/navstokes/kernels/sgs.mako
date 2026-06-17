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
## Eigenvalues are computed analytically by Cardano's trigonometric
## formula on the depressed cubic from the characteristic polynomial:
##   det(lambda I - G) = lambda^3 - I1 lambda^2 + I2 lambda - I3 = 0
## with I1, I2, I3 the invariants of G. Setting lambda = x + I1/3:
##   x^3 + p x + q = 0
##   p = I2 - I1^2/3
##   q = -2 I1^3/27 + I1 I2/3 - I3
## For three real roots (p < 0, always true for SPD with distinct
## eigenvalues):
##   m   = 2 sqrt(-p/3)
##   phi = (1/3) acos( 3q / (p m) )
##   x_k = m cos(phi - 2 pi k / 3)   for k = 0, 1, 2
##   lambda_k = x_k + I1/3
## Guards: cos_arg clipped to [-1,1] against round-off; p clipped against
## zero (G ~ scalar -> all eigenvalues equal -> D_sigma = 0); sigma_1^2
## clipped against zero (no resolved gradient -> nu_t = 0).
##
<%pyfr:macro name='sgs_sigma_mu_t'
             params='rho_in, alpha_in, delta_e, mu_sgs'>
    {
% if ndims == 3:
        // G = alpha^T alpha  (symmetric, upper triangle only)
        fpdtype_t G00_s = ${' + '.join(f'alpha_in[{m}][0]*alpha_in[{m}][0]'
                                       for m in range(3))};
        fpdtype_t G11_s = ${' + '.join(f'alpha_in[{m}][1]*alpha_in[{m}][1]'
                                       for m in range(3))};
        fpdtype_t G22_s = ${' + '.join(f'alpha_in[{m}][2]*alpha_in[{m}][2]'
                                       for m in range(3))};
        fpdtype_t G01_s = ${' + '.join(f'alpha_in[{m}][0]*alpha_in[{m}][1]'
                                       for m in range(3))};
        fpdtype_t G02_s = ${' + '.join(f'alpha_in[{m}][0]*alpha_in[{m}][2]'
                                       for m in range(3))};
        fpdtype_t G12_s = ${' + '.join(f'alpha_in[{m}][1]*alpha_in[{m}][2]'
                                       for m in range(3))};

        // Invariants of G
        fpdtype_t I1_s = G00_s + G11_s + G22_s;
        fpdtype_t I2_s = G00_s*G11_s - G01_s*G01_s
                       + G00_s*G22_s - G02_s*G02_s
                       + G11_s*G22_s - G12_s*G12_s;
        fpdtype_t I3_s = G00_s*(G11_s*G22_s - G12_s*G12_s)
                       - G01_s*(G01_s*G22_s - G12_s*G02_s)
                       + G02_s*(G01_s*G12_s - G11_s*G02_s);

        // Depressed cubic coefficients (x = lambda - I1/3)
        fpdtype_t p_s = I2_s - I1_s*I1_s/(fpdtype_t)3.0;
        fpdtype_t q_s = -(fpdtype_t)(2.0/27.0)*I1_s*I1_s*I1_s
                       + I1_s*I2_s/(fpdtype_t)3.0 - I3_s;

        // Trigonometric solution. -p > 0 always for SPD with non-degenerate
        // eigenvalues; clip against round-off-induced negatives or zeros.
        fpdtype_t p_neg_s = fmax(-p_s, (fpdtype_t)1e-15);
        fpdtype_t m_s = (fpdtype_t)2.0*sqrt(p_neg_s/(fpdtype_t)3.0);

        // cos(3 phi) = (3 q) / (p m); clip to [-1, 1] for acos safety.
        fpdtype_t cos_arg_s = (fpdtype_t)3.0*q_s/(p_s*m_s);
        cos_arg_s = fmax((fpdtype_t)-1.0,
                         fmin((fpdtype_t)1.0, cos_arg_s));
        fpdtype_t phi_s = acos(cos_arg_s)/(fpdtype_t)3.0;

        // Three eigenvalues (unsorted)
        fpdtype_t lam0_s = m_s*cos(phi_s) + I1_s/(fpdtype_t)3.0;
        fpdtype_t lam1_s = m_s*cos(phi_s - (fpdtype_t)${2.0*3.141592653589793/3.0})
                         + I1_s/(fpdtype_t)3.0;
        fpdtype_t lam2_s = m_s*cos(phi_s - (fpdtype_t)${4.0*3.141592653589793/3.0})
                         + I1_s/(fpdtype_t)3.0;

        // Sort descending using max+min+(sum−max−min) identity
        fpdtype_t lam_max_s = fmax(fmax(lam0_s, lam1_s), lam2_s);
        fpdtype_t lam_min_s = fmin(fmin(lam0_s, lam1_s), lam2_s);
        fpdtype_t lam_mid_s = lam0_s + lam1_s + lam2_s
                            - lam_max_s - lam_min_s;

        // Singular values (clip to non-negative for safety; lambdas
        // should be non-negative for SPD but round-off can push the
        // smallest one slightly negative)
        fpdtype_t sig1_s = sqrt(fmax(lam_max_s, (fpdtype_t)0.0));
        fpdtype_t sig2_s = sqrt(fmax(lam_mid_s, (fpdtype_t)0.0));
        fpdtype_t sig3_s = sqrt(fmax(lam_min_s, (fpdtype_t)0.0));

        // D_sigma = sigma_3 (sigma_1 - sigma_2) (sigma_2 - sigma_3) / sigma_1^2
        // Guard against sigma_1 ~ 0 (locally uniform velocity field)
        fpdtype_t sig1_sq_s = sig1_s*sig1_s;
        fpdtype_t D_sigma_s = (sig1_sq_s > (fpdtype_t)1e-15)
                            ? sig3_s*(sig1_s - sig2_s)*(sig2_s - sig3_s)
                              /sig1_sq_s
                            : (fpdtype_t)0.0;

        // mu_sgs = rho * (C_sigma * Delta_e)^2 * D_sigma
        fpdtype_t Cdelta_s = (fpdtype_t)${c_sigma}*delta_e;
        mu_sgs = rho_in*Cdelta_s*Cdelta_s*D_sigma_s;
% else:
        // 2D: rank-deficient alpha yields sigma_3 = 0 and the sigma
        // model is identically zero. Emit 0 without computing G.
        mu_sgs = (fpdtype_t)0.0;
% endif
    }
</%pyfr:macro>
