<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%include file='pyfr.solvers.navstokes.kernels.bcs.${bctype}'/>

% if bccfluxstate:
<%include file='pyfr.solvers.navstokes.kernels.bcs.${bccfluxstate}'/>
% endif

<%pyfr:kernel name='bccflux' ndim='1'
              ul='inout view fpdtype_t[${str(nvars)}]'
              gradul='in view fpdtype_t[${str(ndims)}][${str(nvars)}]'
              artvisc='in view fpdtype_t'
              nl='in fpdtype_t[${str(ndims)}]'
% if sgs_model != 'none':
              delta_e_l='in fpdtype_t'
% endif
              >
    fpdtype_t mag_nl = sqrt(${pyfr.dot('nl[{i}]', i=ndims)});
    fpdtype_t norm_nl[] = ${pyfr.array('(1 / mag_nl)*nl[{i}]', i=ndims)};

% if sgs_model != 'none':
    // viscous_flux_add (called transitively via bc_common_flux_state)
    // reads delta_e from the calling scope. BCs have only one side.
    fpdtype_t delta_e = delta_e_l;
% endif

    ${pyfr.expand('bc_common_flux_state', 'ul', 'gradul', 'artvisc', 'norm_nl', 'mag_nl')};
</%pyfr:kernel>
