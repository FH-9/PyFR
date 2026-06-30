import numpy as np

from pyfr.mpiutil import get_comm_rank_root, mpi
from pyfr.plugins.solver.base import BaseSolverPlugin
from pyfr.quadrules.surface import SurfaceIntegrator
from pyfr.readers.native import Connectivity


class MassFlowSourcePlugin(BaseSolverPlugin):
    name = 'massflowsource'
    systems = 'navier-stokes|euler'
    dimensions = '2|3'

    def __init__(self, intg, cfgsect):
        super().__init__(intg, cfgsect)

        # Target mass flow rate and controller gain
        self.target_mfr = self.cfg.getfloat(cfgsect, 'target-mfr')
        self.Ac = self.cfg.getfloat(cfgsect, 'Ac')

        # Conserved-variable index the body force is added to: 1 = x-momentum,
        # 2 = y-momentum, 3 = z-momentum (0 is continuity, last is energy).
        self.direction = self.cfg.getint(cfgsect, 'direction')
        if not 1 <= self.direction <= self.ndims:
            raise ValueError('massflowsource direction must be in '
                             f'[1, {self.ndims}]')

        # Control every nsteps; resume the pressure gradient from a restart
        self.nsteps = self.cfg.getint(cfgsect, 'nsteps', 200)
        self.dpdx = self.cfg.getfloat(cfgsect, 'dpdx', 0.0)

        # Controller history (seed current/previous flow at the target)
        self.t_prev = intg.tcurr
        self.m_n = self.m_n_1 = self.target_mfr

        # Build the integration surface and its quadrature operators
        con = self._build_surface(intg)
        self.mf_int = SurfaceIntegrator(self.cfg, cfgsect,
                                        intg.system.ele_map, con, flags='s')

        # Per-element forcing matrices wired into the source macro via an
        # external; updated in place each control step.
        self.dpdx_mats = {}
        for etype, eles in intg.system.ele_map.items():
            eles.add_src_macro('pyfr.plugins.solver.kernels.massflowsource',
                               'massflowsource', {'dir': self.direction})

            m = intg.backend.matrix((1, eles.neles), tags={'align'})
            self.dpdx_mats[etype] = (m, eles.neles)
            eles.set_external('forcing', 'in broadcast-col fpdtype_t[1]',
                              value=m)

        # Push the initial pressure gradient to the device
        self._set_forcing()

    def _build_surface(self, intg):
        # Connectivity for the surface the mass flow is integrated over.
        mesh = intg.system.mesh
        bcname = self.cfg.get(self.cfgsect, 'boundary-face')

        # Named (physical) boundary: connectivity is ready-made
        if not bcname.startswith('periodic'):
            con = mesh.bcon.get(bcname)
            if con is None:
                raise ValueError(f'Boundary {bcname} not found in mesh')

            return con

        # Periodic plane (e.g. a channel inflow/outflow): periodic faces are
        # matched into the interior connectivity, so reconstruct one side of
        # the pairing from the raw mesh into a Connectivity.  Config form is
        # `boundary-face = periodic_<group>` (group as in the mesh, e.g. 0).
        grp = bcname.split('_', 1)[1]
        if 'periodic' not in mesh.raw or grp not in mesh.raw['periodic']:
            raise ValueError(f'Periodic group {grp!r} not found in mesh')

        # Each row pairs the two sides; take side A = pairs[:, 0]
        pairs = mesh.raw['periodic'][grp][()]
        side = pairs[:, 0]

        # Global element id -> partition-local index, per element type
        g2l = {et: {int(g): l for l, g in enumerate(gids)}
               for et, gids in mesh.eidxs.items()}

        # Keep only faces whose owning element lives on this rank
        cidxs, leidx = [], []
        for cidx, off in zip(side['cidx'], side['off']):
            etype, _ = mesh.cidxmap[cidx]
            loc = g2l.get(etype, {}).get(int(off))
            if loc is not None:
                cidxs.append(cidx)
                leidx.append(loc)

        return Connectivity(np.array(cidxs, dtype=np.int16),
                            np.array(leidx, dtype=int), mesh.cidxmap)

    def _compute_mfr(self, intg):
        # Mass flow rate = integral of rho*u . n over the surface
        solns = dict(zip(intg.system.ele_types, intg.soln))

        mf = 0.0
        for (etype, fidx), m0 in self.mf_int.m0.items():
            eidxs = self.mf_int.eidxs[etype, fidx]
            qwts = self.mf_int.qwts[etype, fidx]
            norms = self.mf_int.norms[etype, fidx]      # (ndims, nfpts, neles)

            # Momentum (rho*u_i) at solution points, then to face points
            mom = solns[etype][:, 1:1 + self.ndims, eidxs]
            momf = np.einsum('fp,pdn->fdn', m0, mom)    # (nfpts, ndims, neles)

            # Quadrature of rho*u . n
            mf += np.einsum('f,fdn,dfn->', qwts, momf, norms)

        comm, rank, root = get_comm_rank_root()
        return abs(comm.allreduce(mf, op=mpi.SUM))

    def _set_forcing(self):
        for m, ncol in self.dpdx_mats.values():
            m.set(np.full((1, ncol), -self.dpdx))

    def __call__(self, intg):
        # Only act on accepted steps at the control interval
        if intg.nacptsteps % self.nsteps or intg.tcurr == 0.0:
            return

        comm, rank, root = get_comm_rank_root()

        # Current mass flow rate
        mf = self._compute_mfr(intg)

        # Update history and the pressure gradient (predictor-corrected
        # integral control towards the target mass flow rate)
        self.m_n_1, self.m_n = self.m_n, mf
        dt = intg.tcurr - self.t_prev
        self.dpdx -= (self.target_mfr - 2*self.m_n + self.m_n_1) / (self.Ac*dt)
        self.t_prev = intg.tcurr

        # Push the new forcing to the device
        self._set_forcing()

        # Periodic status line
        if rank == root and intg.nacptsteps % (750*self.nsteps) == 0:
            print(f'\nMFR control: target={self.target_mfr}, current={mf}, '
                  f'dp/dx={self.dpdx}')
