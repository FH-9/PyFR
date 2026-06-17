<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%include file='pyfr.solvers.navstokes.kernels.sgs'/>

<%pyfr:kernel name='sgs_test_kernel' ndim='1'
              rho='in fpdtype_t'
              alpha='in fpdtype_t[${str(ndims)}][${str(ndims)}]'
              delta_e='in fpdtype_t'
              musgs='out fpdtype_t'>
    ${pyfr.expand(f'sgs_{sgs_model}_mu_t', 'rho', 'alpha', 'delta_e', 'musgs')};
</%pyfr:kernel>
