<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%include file='pyfr.solvers.navstokes.kernels.bcs.alg-wall'/>

## Thin wrapper exercising the real alg-wall bc_common_flux_state macro.
## ul/nl are direct kernel args; u_match/h_wm/warm_state are externs (passed
## via extrns).  The macro stashes the converged u_tau in warm_state[0],
## which the test reads back.
<%pyfr:kernel name='alg_wall_test_kernel' ndim='1'
              ul='inout fpdtype_t[${str(nvars)}]'
              nl='in fpdtype_t[${str(ndims)}]'>
    fpdtype_t magnl = 1.0;
    fpdtype_t artvisc = 0.0;
    fpdtype_t gradul[${ndims}][${nvars}];
    ${pyfr.expand('bc_common_flux_state', 'ul', 'gradul', 'artvisc',
                  'nl', 'magnl')};
</%pyfr:kernel>
