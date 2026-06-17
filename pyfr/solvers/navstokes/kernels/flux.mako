<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%include file='pyfr.solvers.navstokes.kernels.sgs'/>

% if ndims == 2:
<%pyfr:macro name='viscous_flux_add' params='uin, grad_uin, fout'>
    fpdtype_t rho = uin[0], rhou = uin[1], rhov = uin[2], E = uin[3];

    fpdtype_t rcprho = 1.0/rho;
    fpdtype_t u = rcprho*rhou, v = rcprho*rhov;

    fpdtype_t rho_x = grad_uin[0][0];
    fpdtype_t rho_y = grad_uin[1][0];

    // Velocity derivatives (rho*grad[u,v])
    fpdtype_t u_x = grad_uin[0][1] - u*rho_x;
    fpdtype_t u_y = grad_uin[1][1] - u*rho_y;
    fpdtype_t v_x = grad_uin[0][2] - v*rho_x;
    fpdtype_t v_y = grad_uin[1][2] - v*rho_y;

    fpdtype_t E_x = grad_uin[0][3];
    fpdtype_t E_y = grad_uin[1][3];

% if visc_corr == 'sutherland':
    // Compute the temperature and viscosity
    fpdtype_t cpT = ${c['gamma']}*(rcprho*E - 0.5*(u*u + v*v));
    fpdtype_t Trat = ${1/c['cpTref']}*cpT;
    fpdtype_t mu_c = ${c['mu']*(c['cpTref'] + c['cpTs'])}*Trat*sqrt(Trat)
                   / (cpT + ${c['cpTs']});
% else:
    fpdtype_t mu_c = ${c['mu']};
% endif

    // Compute temperature derivatives (c_v*dT/d[x,y])
    fpdtype_t T_x = rcprho*(E_x - (rcprho*rho_x*E + u*u_x + v*v_x));
    fpdtype_t T_y = rcprho*(E_y - (rcprho*rho_y*E + u*u_y + v*v_y));

% if sgs_model != 'none':
    // Build the physical primitive velocity gradient
    //   alpha[i][j] = du_i/dx_j
    // u_x etc. above are rho*du/dx (PyFR's conservative-gradient
    // convention), so divide by rho here.
    fpdtype_t alpha[${ndims}][${ndims}];
    alpha[0][0] = rcprho*u_x;  alpha[0][1] = rcprho*u_y;
    alpha[1][0] = rcprho*v_x;  alpha[1][1] = rcprho*v_y;

    // SGS dynamic viscosity mu_sgs = rho * nu_t. The per-element
    // filter width delta_e must be in scope (provided by the
    // calling kernel via a kernel argument or extern).
    fpdtype_t mu_sgs;
    ${pyfr.expand(f'sgs_{sgs_model}_mu_t', 'rho', 'alpha', 'delta_e', 'mu_sgs')};

    // Effective dynamic viscosity (molecular + SGS) for the
    // deviatoric stress; heat-flux coefficient uses Pr for the
    // molecular part and Pr_t for the SGS part.
    fpdtype_t mu_eff = mu_c + mu_sgs;
    fpdtype_t cT = mu_c*${c['gamma']/c['Pr']}
                 + mu_sgs*${c['gamma']/c['Prt']};
% else:
    fpdtype_t mu_eff = mu_c;
    fpdtype_t cT = mu_c*${c['gamma']/c['Pr']};
% endif

    // Negated stress tensor elements
    fpdtype_t t_xx = -2*mu_eff*rcprho*(u_x - ${1.0/3.0}*(u_x + v_y));
    fpdtype_t t_yy = -2*mu_eff*rcprho*(v_y - ${1.0/3.0}*(u_x + v_y));
    fpdtype_t t_xy = -mu_eff*rcprho*(v_x + u_y);

    fout[0][1] += t_xx;     fout[1][1] += t_xy;
    fout[0][2] += t_xy;     fout[1][2] += t_yy;

    fout[0][3] += u*t_xx + v*t_xy + -cT*T_x;
    fout[1][3] += u*t_xy + v*t_yy + -cT*T_y;
</%pyfr:macro>
% elif ndims == 3:
<%pyfr:macro name='viscous_flux_add' params='uin, grad_uin, fout'>
    fpdtype_t rho  = uin[0];
    fpdtype_t rhou = uin[1], rhov = uin[2], rhow = uin[3];
    fpdtype_t E    = uin[4];

    fpdtype_t rcprho = 1.0/rho;
    fpdtype_t u = rcprho*rhou, v = rcprho*rhov, w = rcprho*rhow;

    fpdtype_t rho_x = grad_uin[0][0];
    fpdtype_t rho_y = grad_uin[1][0];
    fpdtype_t rho_z = grad_uin[2][0];

    // Velocity derivatives (rho*grad[u,v,w])
    fpdtype_t u_x = grad_uin[0][1] - u*rho_x;
    fpdtype_t u_y = grad_uin[1][1] - u*rho_y;
    fpdtype_t u_z = grad_uin[2][1] - u*rho_z;
    fpdtype_t v_x = grad_uin[0][2] - v*rho_x;
    fpdtype_t v_y = grad_uin[1][2] - v*rho_y;
    fpdtype_t v_z = grad_uin[2][2] - v*rho_z;
    fpdtype_t w_x = grad_uin[0][3] - w*rho_x;
    fpdtype_t w_y = grad_uin[1][3] - w*rho_y;
    fpdtype_t w_z = grad_uin[2][3] - w*rho_z;

    fpdtype_t E_x = grad_uin[0][4];
    fpdtype_t E_y = grad_uin[1][4];
    fpdtype_t E_z = grad_uin[2][4];

% if visc_corr == 'sutherland':
    // Compute the temperature and viscosity
    fpdtype_t cpT = ${c['gamma']}*(rcprho*E - 0.5*(u*u + v*v + w*w));
    fpdtype_t Trat = ${1/c['cpTref']}*cpT;
    fpdtype_t mu_c = ${c['mu']*(c['cpTref'] + c['cpTs'])}*Trat*sqrt(Trat)
                   / (cpT + ${c['cpTs']});
% else:
    fpdtype_t mu_c = ${c['mu']};
% endif

    // Compute temperature derivatives (c_v*dT/d[x,y,z])
    fpdtype_t T_x = rcprho*(E_x - (rcprho*rho_x*E + u*u_x + v*v_x + w*w_x));
    fpdtype_t T_y = rcprho*(E_y - (rcprho*rho_y*E + u*u_y + v*v_y + w*w_y));
    fpdtype_t T_z = rcprho*(E_z - (rcprho*rho_z*E + u*u_z + v*v_z + w*w_z));

% if sgs_model != 'none':
    // Build the physical primitive velocity gradient
    //   alpha[i][j] = du_i/dx_j
    // u_x etc. above are rho*du/dx (PyFR's conservative-gradient
    // convention), so divide by rho here.
    fpdtype_t alpha[${ndims}][${ndims}];
    alpha[0][0] = rcprho*u_x;  alpha[0][1] = rcprho*u_y;  alpha[0][2] = rcprho*u_z;
    alpha[1][0] = rcprho*v_x;  alpha[1][1] = rcprho*v_y;  alpha[1][2] = rcprho*v_z;
    alpha[2][0] = rcprho*w_x;  alpha[2][1] = rcprho*w_y;  alpha[2][2] = rcprho*w_z;

    // SGS dynamic viscosity mu_sgs = rho * nu_t. The per-element
    // filter width delta_e must be in scope (provided by the
    // calling kernel via a kernel argument or extern).
    fpdtype_t mu_sgs;
    ${pyfr.expand(f'sgs_{sgs_model}_mu_t', 'rho', 'alpha', 'delta_e', 'mu_sgs')};

    // Effective dynamic viscosity (molecular + SGS) for the
    // deviatoric stress; heat-flux coefficient uses Pr for the
    // molecular part and Pr_t for the SGS part.
    fpdtype_t mu_eff = mu_c + mu_sgs;
    fpdtype_t cT = mu_c*${c['gamma']/c['Pr']}
                 + mu_sgs*${c['gamma']/c['Prt']};
% else:
    fpdtype_t mu_eff = mu_c;
    fpdtype_t cT = mu_c*${c['gamma']/c['Pr']};
% endif

    // Negated stress tensor elements
    fpdtype_t t_xx = -2*mu_eff*rcprho*(u_x - ${1.0/3.0}*(u_x + v_y + w_z));
    fpdtype_t t_yy = -2*mu_eff*rcprho*(v_y - ${1.0/3.0}*(u_x + v_y + w_z));
    fpdtype_t t_zz = -2*mu_eff*rcprho*(w_z - ${1.0/3.0}*(u_x + v_y + w_z));
    fpdtype_t t_xy = -mu_eff*rcprho*(v_x + u_y);
    fpdtype_t t_xz = -mu_eff*rcprho*(u_z + w_x);
    fpdtype_t t_yz = -mu_eff*rcprho*(w_y + v_z);

    fout[0][1] += t_xx;     fout[1][1] += t_xy;     fout[2][1] += t_xz;
    fout[0][2] += t_xy;     fout[1][2] += t_yy;     fout[2][2] += t_yz;
    fout[0][3] += t_xz;     fout[1][3] += t_yz;     fout[2][3] += t_zz;

    fout[0][4] += u*t_xx + v*t_xy + w*t_xz + -cT*T_x;
    fout[1][4] += u*t_xy + v*t_yy + w*t_yz + -cT*T_y;
    fout[2][4] += u*t_xz + v*t_yz + w*t_zz + -cT*T_z;
</%pyfr:macro>
% endif
