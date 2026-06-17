<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%include file='pyfr.solvers.baseadvecdiff.kernels.artvisc'/>
<%include file='pyfr.solvers.euler.kernels.rsolvers.${rsolver}'/>
<%include file='pyfr.solvers.navstokes.kernels.bcs.common'/>
<%include file='pyfr.solvers.navstokes.kernels.flux'/>

<% tau = c['ldg-tau'] %>

<%pyfr:macro name='bc_ldg_state' params='ul, nl, ur' externs='ploc, t'>
    // Free-slip ghost for gradient step: reflect normal momentum component
    fpdtype_t nor = ${' + '.join(f'ul[{i + 1}]*nl[{i}]' for i in range(ndims))};
    ur[0] = ul[0];
% for i in range(ndims):
    ur[${i + 1}] = ul[${i + 1}] - 2*nor*nl[${i}];
% endfor
    ur[${nvars - 1}] = ul[${nvars - 1}];
</%pyfr:macro>

<%pyfr:alias name='bc_rsolve_state' func='bc_ldg_state'/>
<%pyfr:alias name='bc_ldg_grad_state' func='bc_common_grad_copy'/>

<%pyfr:macro name='bc_common_flux_state' params='ul, gradul, artvisc, nl, magnl'>
    // Robin slip condition: u_wall_i = slip_l * du_i/dn + v_i
    // nl is the unit outward normal; gradul[j][i] = d(q_i)/d(x_j) physical gradient
    fpdtype_t rho_inv = 1.0/ul[0];
    fpdtype_t d_rho_n = ${' + '.join(f'gradul[{j}][0]*nl[{j}]' for j in range(ndims))};

    fpdtype_t ur[${nvars}];
    ur[0] = ul[0];
% for i, v in enumerate('uvw'[:ndims]):
    // Normal derivative of primitive u_i: (d(rho*u_i)/dn - u_i*d(rho)/dn) / rho
    // Ghost: ur[i+1] = 2*rho*u_wall_i - ul[i+1]  where u_wall_i = slip_l*du_i/dn + v_i
    fpdtype_t d_rhou${i}_n = ${' + '.join(f'gradul[{j}][{i + 1}]*nl[{j}]' for j in range(ndims))};
    ur[${i + 1}] = 2*${c['slip_l']}*(d_rhou${i}_n - ul[${i + 1}]*rho_inv*d_rho_n)
                 + 2*${c[v]}*ul[0] - ul[${i + 1}];
% endfor
    // Adiabatic: ghost energy = interior internal energy + ghost kinetic energy
    ur[${nvars - 1}] = ul[${nvars - 1}]
                     - (0.5*rho_inv)*${pyfr.dot('ul[{i}]', i=(1, ndims + 1))}
                     + (0.5/ur[0])*${pyfr.dot('ur[{i}]', i=(1, ndims + 1))};

    // Viscous ghost gradient: copy interior gradient (symmetric treatment)
    fpdtype_t gradur[${ndims}][${nvars}];
    ${pyfr.expand('bc_common_grad_copy', 'ul', 'nl', 'gradul', 'gradur')};

    fpdtype_t fvr[${ndims}][${nvars}] = {{0}};
    ${pyfr.expand('viscous_flux_add', 'ur', 'gradur', 'fvr')};
    ${pyfr.expand('artificial_viscosity_add', 'gradur', 'fvr', 'artvisc')};

    // Inviscid Riemann flux using slip ghost state
    fpdtype_t ficomm[${nvars}];
    ${pyfr.expand('rsolve', 'ul', 'ur', 'nl', 'ficomm')};

    // Combine inviscid and viscous fluxes with LDG tau penalty
% for i in range(nvars):
    fpdtype_t fvcomm_${i} = ${' + '.join(f'nl[{j}]*fvr[{j}][{i}]' for j in range(ndims))};
% if tau != 0.0:
    fvcomm_${i} += ${tau}*(ul[${i}] - ur[${i}]);
% endif
    ul[${i}] = magnl*(ficomm[${i}] + fvcomm_${i});
% endfor
</%pyfr:macro>
