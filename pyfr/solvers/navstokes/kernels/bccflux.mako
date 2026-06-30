<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%include file='pyfr.solvers.navstokes.kernels.bcs.${bctype}'/>

% if bccfluxstate:
<%include file='pyfr.solvers.navstokes.kernels.bcs.${bccfluxstate}'/>
% endif

## Body shared by both forms; the SGS form gains a delta_e_l argument.
## mako cannot place a `% if` inside a tag argument list, so the body lives
## in a def and the tag is emitted in two conditional forms.
<%def name='bccflux_body()'>
    fpdtype_t mag_nl = sqrt(${pyfr.dot('nl[{i}]', i=ndims)});
    fpdtype_t norm_nl[] = ${pyfr.array('(1 / mag_nl)*nl[{i}]', i=ndims)};

% if sgs_model != 'none':
    // viscous_flux_add (called transitively via bc_common_flux_state)
    // reads delta_e from the calling scope. BCs have only one side.
    fpdtype_t delta_e = delta_e_l;
% endif

    ${pyfr.expand('bc_common_flux_state', 'ul', 'gradul', 'artvisc', 'norm_nl', 'mag_nl')};
</%def>
% if sgs_model != 'none':
<%pyfr:kernel name='bccflux' ndim='1'
              ul='inout view fpdtype_t[${str(nvars)}]'
              gradul='in view fpdtype_t[${str(ndims)}][${str(nvars)}]'
              artvisc='in view fpdtype_t'
              nl='in fpdtype_t[${str(ndims)}]'
              delta_e_l='in fpdtype_t'>
${bccflux_body()}
</%pyfr:kernel>
% else:
<%pyfr:kernel name='bccflux' ndim='1'
              ul='inout view fpdtype_t[${str(nvars)}]'
              gradul='in view fpdtype_t[${str(ndims)}][${str(nvars)}]'
              artvisc='in view fpdtype_t'
              nl='in fpdtype_t[${str(ndims)}]'>
${bccflux_body()}
</%pyfr:kernel>
% endif
