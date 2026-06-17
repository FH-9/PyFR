<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%include file='pyfr.solvers.navstokes.kernels.bcs.gq-ode-wall'/>

## Thin wrapper exercising the real gq-ode-wall bc_common_flux_state macro.
## ul/nl are direct args; u_match/h_wm/warm_state/gq_y/gq_w are externs.
## The macro stashes the converged u_tau in warm_state[0].
<%pyfr:kernel name='gq_wall_test_kernel' ndim='1'
              ul='inout fpdtype_t[${str(nvars)}]'
              nl='in fpdtype_t[${str(ndims)}]'>
    fpdtype_t magnl = 1.0;
    fpdtype_t artvisc = 0.0;
    fpdtype_t gradul[${ndims}][${nvars}];
    ${pyfr.expand('bc_common_flux_state', 'ul', 'gradul', 'artvisc',
                  'nl', 'magnl')};
</%pyfr:kernel>
