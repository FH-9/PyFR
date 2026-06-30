<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:macro name='massflowsource' params='t, u, ploc, src' externs='forcing'>
    src[${dir}] += forcing[0];
</%pyfr:macro>
