<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:kernel name='wmles_matching' ndim='1'
              u_volume='in view fpdtype_t[${str(nupts)}][${str(nvars)}]'
              interp='in fpdtype_t[${str(nupts)}]'
              u_match='out view fpdtype_t[${str(nprims)}][1]'>
    // Interpolate conservatives at the matching point inside the parent
    // element: u_c[v] = sum_k interp[k] * u_volume[k][v]
    fpdtype_t u_c[${nvars}];
% for v in range(nvars):
    u_c[${v}] = ${pyfr.dot('interp[{k}]*u_volume[{k}][' + str(v) + ']',
                           k=nupts)};
% endfor

    fpdtype_t rcprho = 1.0/u_c[0];

% for i in range(ndims):
    u_match[${i + 1}][0] = u_c[${i + 1}]*rcprho;
% endfor

    // Slot 0: density (taken directly from the interpolated conservatives
    // — no EOS conversion needed). Slot ndims+1: pressure via ideal-gas
    // EOS p = (gamma-1)(E - 0.5 rho |v|^2).
    fpdtype_t v2 = ${pyfr.dot('u_c[{i}]*u_c[{i}]', i=(1, ndims + 1))}
                   *rcprho*rcprho;
    u_match[0][0] = u_c[0];
    u_match[${ndims + 1}][0] = ${c['gamma'] - 1}*(u_c[${ndims + 1}]
                                                  - 0.5*u_c[0]*v2);
</%pyfr:kernel>
