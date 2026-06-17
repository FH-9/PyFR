import numpy as np


def setup_matching_groups(lhs, elemap, *, tol_frac=0.05,
                          newton_niters=50, newton_rtol=1e-4):
    groups = []
    n_total = 0

    for etype, fidx, eidxs, idx in lhs.foreach():
        eles = elemap[etype]
        shape = type(eles.basis)

        nspts = eles.nspts
        ndims = eles.ndims
        nfp = eles.nfacefpts[fidx]
        n_eles_grp = len(eidxs)
        n_fpts_grp = n_eles_grp*nfp

        # Physical centroid per boundary-adjacent element. Inline matmul
        # mirroring eles.ploc_at_np, but evaluated only on the subset of
        # elements indexed by eidxs (the wall-adjacent ones) to avoid
        # wasted work on the full element set.
        ref_cent = shape.std_ele_centroid[None, :]
        sop_c = eles.basis.sbasis.nodal_basis_at(ref_cent)
        spts = eles.eles[:, eidxs, :]
        x_c = (sop_c @ spts.reshape(nspts, -1)).reshape(1, n_eles_grp, ndims)[0]

        # Wall flux-point physical positions and outward unit normals
        x_fp_flat, = eles.get_ploc_for_inters(eidxs, fidx)
        pn_flat, = eles.get_pnorms_for_inters(eidxs, fidx)
        x_fp = x_fp_flat.reshape(n_eles_grp, nfp, ndims)
        pn = pn_flat.reshape(n_eles_grp, nfp, ndims)
        n_fp = pn / np.linalg.norm(pn, axis=-1, keepdims=True)

        # h_wm = (x_c - x_fp) · (-n_fp); matching point moves inward
        h_wm = np.einsum('eid,eid->ei', x_c[:, None, :] - x_fp, -n_fp)
        x_wm = x_fp - h_wm[..., None]*n_fp

        # Invert geometric mapping per fpt: physical → reference. Each
        # element's shape points are repeated nfp times so query points sit
        # in 1-to-1 correspondence with the second axis.
        spts_tiled = np.repeat(spts, nfp, axis=1)
        x_wm_flat = x_wm.reshape(n_fpts_grp, ndims)
        kt0 = np.tile(shape.std_ele_centroid, (n_fpts_grp, 1))

        xi_wm = _invert_geomap(eles.basis.sbasis, spts_tiled, x_wm_flat, kt0,
                               niters=newton_niters, rtol=newton_rtol,
                               etype=etype, fidx=fidx)

        # Reference-element containment check (abort if far outside)
        _check_containment(xi_wm, shape, tol_frac, etype, fidx)

        # Interpolation row per fpt: u(x_wm_ref) = N(x_wm_ref) · u_upts
        interp = eles.basis.ubasis.nodal_basis_at(xi_wm)

        groups.append({
            'etype': etype, 'fidx': fidx, 'eidxs': eidxs, 'idx': idx,
            'interp': interp,
            'h_wm': h_wm.ravel(),
            'xi_wm': xi_wm,
            'fpt_offset': n_total,
            'n_fpts': n_fpts_grp,
        })
        n_total += n_fpts_grp

    return groups, n_total


def _invert_geomap(sbasis, spts, plocs, ktlocs, *, niters, rtol, etype, fidx):
    def nb_op(pts): return sbasis.nodal_basis_at(pts, clean=False)
    def jnb_op(pts): return sbasis.jac_nodal_basis_at(pts, clean=False)

    tol = rtol*np.linalg.norm(np.ptp(spts, axis=0), axis=-1)

    k = ktlocs.copy()
    kp = np.einsum('ij,jik->ik', nb_op(k), spts)

    for _ in range(niters):
        A = np.einsum('ijk,jkl->kli', jnb_op(k), spts)
        k -= np.linalg.solve(A, (kp - plocs)[..., None]).squeeze(axis=-1)
        kp = np.einsum('ij,jik->ik', nb_op(k), spts)

        if (np.linalg.norm(kp - plocs, axis=1) < tol).all():
            return k

    d = np.linalg.norm(kp - plocs, axis=1)
    n_fail = int((d >= tol).sum())
    raise RuntimeError(
        f'WMLES Newton inversion failed to converge for {n_fail} flux points '
        f'on etype={etype!r}, fidx={fidx} after {niters} iterations '
        f'(max residual {d.max():.3e}). '
        f'Mesh quality near the wall may be too poor for centroid-based '
        f'matching.'
    )


def _check_containment(xi_ref, shape, tol_frac, etype, fidx):
    # tol_abs is the reference-coordinate buffer added to each face
    # inequality inside shape.valid_spt(). For tensor-product shapes
    # (Hex/Quad) this corresponds cleanly to tol_frac of the half-extent;
    # for simplices/prisms it's a per-face buffer (still defensible at
    # this magnitude); for pyramids it's a fixed buffer that becomes
    # loose near the apex where the cross-section shrinks. Used here as
    # a tripwire for Newton divergence rather than a precise geometric
    # statement, so the topology dependence is acceptable.
    tol_abs = 2*tol_frac

    inside = shape.valid_spt(xi_ref, tol=tol_abs)
    if not inside.all():
        n_bad = int((~inside).sum())
        raise RuntimeError(
            f'WMLES matching location outside reference element for {n_bad} '
            f'flux points on etype={etype!r}, fidx={fidx} '
            f'(tolerance: {tol_frac*100:g}% of element extent). '
            f'Element shear or curvature is excessive for the centroid-based '
            f'matching strategy.'
        )


class WMLESMatchingMixin:
    """LES matching-point infrastructure shared by wall-model BCs.

    Provides per-fpt matching geometry, the wmles_matching kernel that
    populates a BC-local primitives buffer each RK stage, and the
    `u_match` / `h_wm` externs that the BC's flux kernel consumes.

    Contract — the concrete BC class must:
      * inherit this mixin BEFORE the BC base class
        (so super().__init__ wires self._be, self.c, self._tplargs, etc.)
      * call self._wmles_setup(be, lhs, elemap, cfgsect, cfg) from its
        __init__ after super().__init__
    After _wmles_setup, the following are populated:
      self._wmles_nprims, self._wmles_groups,
      self._wmles_match_buf, self._wmles_h_wm_mat
    and the following are wired:
      self.kernels['wmles_matching']  (factory taking uin, returns list)
      self.set_external('u_match', ...) and self.set_external('h_wm', ...)
    """

    def _wmles_setup(self, be, lhs, elemap, cfgsect, cfg, *,
                     n_warm_params=2):
        # c_p is used by BCs that derive T at the matching point via the
        # ideal-gas EOS (currently only eq-ode-wall). Look in the BC's
        # section first, then [constants]; fall back to air-at-STP
        # (1005 J/(kg·K)) for BCs that don't actually consume it.
        if cfg.hasopt(cfgsect, 'cp'):
            self.c['cp'] = cfg.getfloat(cfgsect, 'cp')
        elif cfg.hasopt('constants', 'cp'):
            self.c['cp'] = cfg.getfloat('constants', 'cp')
        else:
            self.c['cp'] = 1005.0

        # Per-fpt matching geometry and interpolation vectors
        self._wmles_groups, n_match = setup_matching_groups(lhs, elemap)
        self._wmles_nprims = self.ndims + 2

        if n_match != self.ninterfpts:
            raise RuntimeError(
                f'WMLES matching fpt count ({n_match}) disagrees with BC '
                f'interface fpt count ({self.ninterfpts}).'
            )

        # Per-fpt primitives buffer: T, velocity, p (T and p zeroed in
        # low-Mach mode).
        self._wmles_match_buf = be.matrix(
            (self._wmles_nprims, n_match), tags={'align'}
        )

        # Map each group's (etype, fidx)-local fpts to their BC global
        # iteration index so the matching kernel can scatter into the
        # right buffer positions.
        self._wmles_compute_global_fpt_idx(elemap)

        # Register the matching kernel template and bind a uin-taking
        # factory (the integrator constructs one per RK register at setup).
        be.pointwise.register(
            'pyfr.solvers.navstokes.kernels.bcs.wmles-matching'
        )
        self.kernels['wmles_matching'] = self._wmles_make_matching_factory(
            elemap
        )

        # Expose the matching buffer to the BC's flux kernel as an extern
        # so it can read per-fpt primitives by iteration index.
        self.set_external(
            'u_match', f'in fpdtype_t[{self._wmles_nprims}]',
            value=self._wmles_match_buf
        )

        # Per-fpt matching height in BC iteration order, also exposed via
        # extern. Reordered once from concat order at setup.
        h_wm_all = np.concatenate([g['h_wm'] for g in self._wmles_groups])
        h_wm_bc_order = h_wm_all[self._wmles_concat_to_bc_order]
        self._wmles_h_wm_mat = be.const_matrix(
            h_wm_bc_order[None, :], tags={'align'}
        )
        self.set_external(
            'h_wm', 'in fpdtype_t', value=self._wmles_h_wm_mat
        )

        # Warm-start buffer: holds the previous-call converged shooting
        # parameters (in BC iteration order, same as h_wm). Zero-initialised;
        # the BC kernel uses warm_state[0] > 0 as the "is initialised"
        # sentinel (tau_w > 0 always physically).
        self._wmles_warm_buf = be.matrix(
            (n_warm_params, n_match),
            initval=np.zeros((n_warm_params, n_match)),
            tags={'align'}
        )
        self.set_external(
            'warm_state', f'inout fpdtype_t[{n_warm_params}]',
            value=self._wmles_warm_buf
        )

    def _wmles_compute_global_fpt_idx(self, elemap):
        # Reproduce the (idx-sort then self._perm) ordering that
        # _get_inter_arrays applies, then invert it so we know where each
        # group's local fpt lands in BC iteration order.
        all_idx = np.concatenate([
            np.repeat(g['idx'], elemap[g['etype']].nfacefpts[g['fidx']])
            for g in self._wmles_groups
        ])
        natural_order = np.argsort(all_idx, kind='stable')
        bc_order = natural_order[self._perm]
        ro_inv = np.argsort(bc_order)

        # Keep the concat→BC mapping for any per-fpt scalar (e.g. h_wm)
        # we want to reorder once at setup.
        self._wmles_concat_to_bc_order = ro_inv

        offset = 0
        for g in self._wmles_groups:
            g['global_fpt_idx'] = ro_inv[offset:offset + g['n_fpts']].astype(
                np.int32
            )
            offset += g['n_fpts']

    def _wmles_make_matching_factory(self, elemap):
        be = self._be

        # Pre-build const matrices that don't depend on uin
        for g in self._wmles_groups:
            eles = elemap[g['etype']]
            nfp = eles.nfacefpts[g['fidx']]

            # Per-fpt interpolation row stored column-major (nupts × n_fpts)
            # so the kernel reads a length-nupts vector per fpt iteration.
            g['_interp_mat'] = be.const_matrix(
                np.ascontiguousarray(g['interp'].T), tags={'align'}
            )
            g['_eidx_per_fpt'] = np.repeat(g['eidxs'], nfp).astype(np.int32)

        def factory(uin):
            kerns = []
            for g in self._wmles_groups:
                eles = elemap[g['etype']]
                nupts = eles.nupts
                n_fpts = g['n_fpts']

                # Per-fpt view of the parent element's full volume DoF
                # block (nupts × nvars) in scal_upts[uin].
                scal = eles.scal_upts[uin]
                u_volume_view = be.view(
                    np.full(n_fpts, scal.mid),
                    np.zeros(n_fpts, dtype=np.int32),
                    g['_eidx_per_fpt'],
                    vshape=(nupts, self.nvars)
                )

                # Per-fpt output view that scatters writes into the BC
                # buffer at the correct (permuted) global index.
                u_match_view = be.view(
                    np.full(n_fpts, self._wmles_match_buf.mid),
                    np.zeros(n_fpts, dtype=np.int32),
                    g['global_fpt_idx'],
                    vshape=(self._wmles_nprims, 1)
                )

                tplargs = {
                    'nupts': nupts, 'nvars': self.nvars,
                    'ndims': self.ndims, 'nprims': self._wmles_nprims,
                    'c': self.c,
                }
                kerns.append(be.kernel(
                    'wmles_matching', tplargs=tplargs, dims=[n_fpts],
                    u_volume=u_volume_view, interp=g['_interp_mat'],
                    u_match=u_match_view
                ))

            return kerns

        return factory
