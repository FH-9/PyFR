<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%include file='pyfr.solvers.baseadvec.kernels.smats'/>
<%include file='pyfr.solvers.baseadvecdiff.kernels.artvisc'/>
<%include file='pyfr.solvers.baseadvecdiff.kernels.transform_grad'/>
<%include file='pyfr.solvers.euler.kernels.flux'/>
<%include file='pyfr.solvers.navstokes.kernels.flux'/>

<% gradu = 'gradu' if 'fused' in ktype else 'f' %>
<% smats = 'smats_l' if 'linear' in ktype else 'smats' %>
<% rcpdjac = 'rcpdjac_l' if 'linear' in ktype else 'rcpdjac' %>

## The kernel body is shared; only the argument list differs when an SGS
## model is active (it gains the per-element filter width delta_e). mako
## cannot place a `% if` inside a tag's argument list, so the body lives in
## a def and the tag is emitted in two conditional forms.
<%def name='tflux_body()'>
% if 'linear' in ktype:
    // Compute the S matrices
    fpdtype_t ${smats}[${ndims}][${ndims}], djac;
    ${pyfr.expand('calc_smats_detj', 'verts', 'upts', smats, 'djac')};
    fpdtype_t ${rcpdjac} = 1 / djac;
% endif

% if 'fused' in ktype:
    // Transform the corrected gradient
    ${pyfr.expand('transform_grad', gradu, smats, rcpdjac)};
% endif

    // Compute the flux (F = Fi + Fv)
    fpdtype_t ftemp[${ndims}][${nvars}];
    fpdtype_t p, v[${ndims}];
    ${pyfr.expand('inviscid_flux', 'u', 'ftemp', 'p', 'v')};
    ${pyfr.expand('viscous_flux_add', 'u', gradu, 'ftemp')};
% if shock_capturing == 'artificial-viscosity':
    fpdtype_t artvisc;
    ${pyfr.expand('interp_artvisc', 'artvisc_vtx', 'upts', 'artvisc')};
    ${pyfr.expand('artificial_viscosity_add', gradu, 'ftemp', 'artvisc')};
% endif

    // Transform the fluxes
% for i, j in pyfr.ndrange(ndims, nvars):
    f[${i}][${j}] = ${' + '.join(f'{smats}[{i}][{k}]*ftemp[{k}][{j}]'
                                 for k in range(ndims))};
% endfor
</%def>
% if sgs_model != 'none':
<%pyfr:kernel name='tflux' ndim='2'
              u='in fpdtype_t[${str(nvars)}]'
              artvisc_vtx='in broadcast-col view(${str(nverts)}) fpdtype_t'
              f='inout fpdtype_t[${str(ndims)}][${str(nvars)}]'
              gradu='inout fpdtype_t[${str(ndims)}][${str(nvars)}]'
              smats='in fpdtype_t[${str(ndims)}][${str(ndims)}]'
              rcpdjac='in fpdtype_t'
              verts='in broadcast-col fpdtype_t[${str(nverts)}][${str(ndims)}]'
              upts='in broadcast-row fpdtype_t[${str(ndims)}]'
              delta_e='in broadcast-col fpdtype_t'>
${tflux_body()}
</%pyfr:kernel>
% else:
<%pyfr:kernel name='tflux' ndim='2'
              u='in fpdtype_t[${str(nvars)}]'
              artvisc_vtx='in broadcast-col view(${str(nverts)}) fpdtype_t'
              f='inout fpdtype_t[${str(ndims)}][${str(nvars)}]'
              gradu='inout fpdtype_t[${str(ndims)}][${str(nvars)}]'
              smats='in fpdtype_t[${str(ndims)}][${str(ndims)}]'
              rcpdjac='in fpdtype_t'
              verts='in broadcast-col fpdtype_t[${str(nverts)}][${str(ndims)}]'
              upts='in broadcast-row fpdtype_t[${str(ndims)}]'>
${tflux_body()}
</%pyfr:kernel>
% endif
