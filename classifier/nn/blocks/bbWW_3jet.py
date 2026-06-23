from .bbWW_models import *


class InputEmbed3jet(nn.Module):
    """
    Input embedding for the 3-jet HH->bbWW semileptonic region:
    exactly 2 b-jets + 1 non-b jet + lepton + regressed neutrino.

    The single non-b jet is treated as the surviving W_had daughter. There is
    no W_had dijet reconstruction. Partial hadronic-top candidates (b + single
    non-b jet) are formed for the ttbar-auxiliary branch; the H->WW reasoning
    leans on the leptonic side (regressed W_lep, lep+nu kinematics) plus
    angular information between the single jet and the leptonic W.
    """

    def __init__(
        self,
        dijetFeatures,
        ancillaryFeatures=["njets", "nsoftjets", "HT", "year"],
        layers=None,
        device="cuda",
        phase_symmetric=False,
    ):
        super(InputEmbed3jet, self).__init__()
        self.layers = layers
        self.debug = False
        self.dD = dijetFeatures
        self.dA = len(ancillaryFeatures)
        self.ancillaryFeatures = ancillaryFeatures
        self.device = device

        # Jet-count constants: 2 b-jets, 1 non-b jet, no dijet combinatorics
        self.bsl, self.wsl, self.qqsl = 2, 1, 0

        if self.dA:
            self.ancillaryEmbed = GhostBatchNorm1d(
                self.dA, features_out=self.dD, phase_symmetric=phase_symmetric,
                conv=True, bias=False, name="ancillary embedder",
            )
            self.layers.addLayer(self.ancillaryEmbed)

        # b-jet embedder: (pt, eta, phi, mass, btagScore) -- phi relative to bb
        self.bJetEmbed = GhostBatchNorm1d(
            5, features_out=self.dD, phase_symmetric=phase_symmetric,
            conv=True, name="jet embedder",
        )
        self.bJetConv = GhostBatchNorm1d(
            self.dD, phase_symmetric=phase_symmetric, conv=True, name="jet convolution",
        )

        # single non-b jet embedder: (pt, eta, phi, mass).
        self.nonbJetEmbed = GhostBatchNorm1d(
            4, features_out=self.dD, phase_symmetric=phase_symmetric,
            conv=True, name="nonb jet embedder",
        )
        self.nonbJetConv = GhostBatchNorm1d(
            self.dD, phase_symmetric=phase_symmetric, conv=True, name="nonb jet convolution",
        )

        # lepton embedder
        self.lepEmbed = GhostBatchNorm1d(
            6, features_out=self.dD, phase_symmetric=phase_symmetric,
            conv=True, name="lepton embedder",
        )
        self.lepConv = GhostBatchNorm1d(
            self.dD, phase_symmetric=phase_symmetric, conv=True, name="lepton convolution",
        )

        # Regressed neutrino 4-vector (pt, eta, phi, E)
        self.regressedNuEmbed = GhostBatchNorm1d(
            4, features_out=self.dD, phase_symmetric=phase_symmetric,
            conv=True, name="regressed neutrino embedder",
        )
        self.regressedNuConv = GhostBatchNorm1d(
            self.dD, phase_symmetric=phase_symmetric, conv=True,
            name="regressed neutrino convolution",
        )

        # Partial hadronic-top candidates: b + single non-b jet (2 candidates, one per b-jet)
        self.bWhadEmbed = GhostBatchNorm1d(
            4, features_out=self.dD, phase_symmetric=phase_symmetric,
            conv=True, name="partial hadronic top embedder",
        )
        self.bWhadConv = GhostBatchNorm1d(
            self.dD, phase_symmetric=phase_symmetric, conv=True,
            name="partial hadronic top convolution",
        )

        # Leptonic top candidates: b + regressed W_lep (2 candidates)
        self.bWlepEmbed = GhostBatchNorm1d(
            4, features_out=self.dD, phase_symmetric=phase_symmetric,
            conv=True, name="leptonic top embedder",
        )
        self.bWlepConv = GhostBatchNorm1d(
            self.dD, phase_symmetric=phase_symmetric, conv=True,
            name="leptonic top convolution",
        )

        # bb dijet
        self.bbDiJetEmbed = GhostBatchNorm1d(
            4, features_out=self.dD, phase_symmetric=phase_symmetric,
            conv=True, name="dijet embedder",
        )
        self.bbDiJetConv = GhostBatchNorm1d(
            self.dD, phase_symmetric=phase_symmetric, conv=True,
            name="dijet convolution",
        )

        # Regressed leptonic W 4-vector (pt, eta, phi, m)
        self.regWlepEmbed = GhostBatchNorm1d(
            4, features_out=self.dD, phase_symmetric=phase_symmetric,
            conv=True, name="regressed leptonic W embedder",
        )
        self.regWlepConv = GhostBatchNorm1d(
            self.dD, phase_symmetric=phase_symmetric, conv=True,
            name="regressed leptonic W convolution",
        )

        # MdR (mass + deltaR) embedders shared across relationship branches
        self.MdREmbed = GhostBatchNorm1d(
            2, features_out=self.dD, phase_symmetric=phase_symmetric,
            conv=True, name="M(a,b), dR(a,b) embedder",
        )
        self.MdRConv = GhostBatchNorm1d(
            self.dD, phase_symmetric=phase_symmetric, conv=True,
            name="M(a,b), dR(a,b) convolution",
        )

        # TT relationship matrices (partial bWhad + bWlep)
        self.MdRttEmbed = GhostBatchNorm1d(
            2, features_out=self.dD, phase_symmetric=phase_symmetric,
            conv=True, name="ttbar relationship embedder",
        )
        self.MdRttConv = GhostBatchNorm1d(
            self.dD, phase_symmetric=phase_symmetric, conv=True,
            name="ttbar relationship convolution",
        )

        # Derived kinematics: 7 features for 3-jet topology
        #   [mbb, lnu_mT, dphi_lep_met, pt_bb, dphi_bb_met, dR(nb, lepW), dR(nb, bb)]
        self.n_derived = 7
        self.derivedEmbed = GhostBatchNorm1d(
            self.n_derived, features_out=self.dD, phase_symmetric=phase_symmetric,
            conv=True, name="derived kinematics embedder",
        )
        self.derivedConv = GhostBatchNorm1d(
            self.dD, phase_symmetric=phase_symmetric, conv=True,
            name="derived kinematics convolution",
        )

        # Leptonic W mass scalar
        self.lepWMassEmbed = GhostBatchNorm1d(
            1, features_out=self.dD, phase_symmetric=phase_symmetric,
            conv=True, name="leptonic W mass embedder",
        )

        # bb mass scalar (replaces hadW_mass in 3-jet, feeds final_mass_head)
        self.bbMassEmbed = GhostBatchNorm1d(
            1, features_out=self.dD, phase_symmetric=phase_symmetric,
            conv=True, name="bb mass embedder",
        )

        # Register layers (embed first, then conv with embed as input dep)
        self.layers.addLayer(self.bJetEmbed)
        self.layers.addLayer(self.bbDiJetEmbed)
        self.layers.addLayer(self.nonbJetEmbed)
        self.layers.addLayer(self.MdREmbed)
        self.layers.addLayer(self.lepEmbed)
        self.layers.addLayer(self.regressedNuEmbed)
        self.layers.addLayer(self.bWhadEmbed)
        self.layers.addLayer(self.bWlepEmbed)
        self.layers.addLayer(self.regWlepEmbed)

        self.layers.addLayer(self.bJetConv, [self.bJetEmbed])
        self.layers.addLayer(self.bbDiJetConv, [self.bbDiJetEmbed])
        self.layers.addLayer(self.nonbJetConv, [self.nonbJetEmbed])
        self.layers.addLayer(self.MdRConv, [self.MdREmbed])
        self.layers.addLayer(self.lepConv, [self.lepEmbed])
        self.layers.addLayer(self.regressedNuConv, [self.regressedNuEmbed])
        self.layers.addLayer(self.bWhadConv, [self.bWhadEmbed])
        self.layers.addLayer(self.bWlepConv, [self.bWlepEmbed])
        self.layers.addLayer(self.regWlepConv, [self.regWlepEmbed])
        self.layers.addLayer(self.MdRttEmbed)
        self.layers.addLayer(self.MdRttConv, [self.MdRttEmbed])
        self.layers.addLayer(self.derivedEmbed)
        self.layers.addLayer(self.derivedConv, [self.derivedEmbed])
        self.layers.addLayer(self.lepWMassEmbed)
        self.layers.addLayer(self.bbMassEmbed)

        # Mask buffer for bb-self-pair MdR (diagonal is the "same jet" mask)
        self.register_buffer(
            "mask_bb_same",
            torch.zeros((1, self.bsl, self.bsl), dtype=torch.bool),
        )
        for i in range(self.bsl):
            self.mask_bb_same[:, i, i] = 1

    def dataPrep(self, b, nb, l, a, reg_nu=None):
        """
        Build physics quantities for the 3-jet region. Unlike the 4-jet lowpt
        version there is no qq combinatorics (wsl=1, qqsl=0). bWhad here is
        the *partial* hadronic top: b-jet + the single surviving non-b jet,
        2 candidates total.
        """
        device = b.get_device() if b.get_device() >= 0 else "cpu"

        n = b.shape[0]
        b = b.view(n, 5, 2)
        nb = nb.view(n, 5, 1)
        l = l.view(n, 6, 1)
        a = a.view(n, self.dA, 1)

        # Drop the regressor attn_score (feature 5): degenerate at 1.0 with wsl=1.
        nb = nb[:, :4, :]                      # (n, 4, 1): pt, eta, phi, mass

        raw_nb = nb.reshape(n, -1).clone()     # (n, 4) flat
        raw_lep = l.squeeze(-1).clone()        # (n, 6)

        # log(HT): look up HT's position dynamically since the ancillary
        # feature list is configured per-workflow (e.g. "njets" is dropped in
        # the 3-jet region because it's constant by selection).
        ht_idx = self.ancillaryFeatures.index("HT")
        a[:, ht_idx, :] = torch.log(torch.clamp(a[:, ht_idx, :], min=1e-6))

        # Leptonic W from regressed neutrino
        if reg_nu is not None:
            reg_nu = reg_nu.view(n, 4)
            nu_PxPyPzE = reg_nu

            lep_PxPyPzE = PxPyPzE(l[:, :4, 0])
            regW_PxPyPzE = lep_PxPyPzE + nu_PxPyPzE
            regW_lep = PtEtaPhiM(regW_PxPyPzE).unsqueeze(-1)  # (n, 4, 1)

            lepW_mass = calc_mW(l[:, :4, 0], reg_nu[:, :3]).unsqueeze(-1).unsqueeze(-1)  # (n, 1, 1)

            # Leptonic top: b + regW, 2 candidates (one per b).
            # Swap ordering (1,0) so index 0 uses b1 (paired w/ b0 hadronic) and vice versa.
            bWlep, bWlepPxPyPzE = addFourVectors(
                b[:, :, (1, 0)], regW_lep.expand(-1, -1, 2)
            )  # (n, 4, 2)

            reg_nu = reg_nu.unsqueeze(-1)
        else:
            reg_nu = torch.zeros(n, 4, 1, device=device)
            regW_lep = torch.zeros(n, 4, 1, device=device)
            regW_PxPyPzE = torch.zeros(n, 4, device=device)
            lepW_mass = torch.zeros(n, 1, 1, device=device)
            bWlep = torch.zeros(n, 4, 2, device=device)
            bWlepPxPyPzE = torch.zeros(n, 4, 2, device=device)

        # H->bb dijet
        bb, bbPxPyPzE = addFourVectors(b[:, :, 0], b[:, :, 1])

        # Partial hadronic top: b + single non-b jet, 2 candidates
        bWhad, bWhadPxPyPzE = addFourVectors(
            b,                           # (n, 4, 2)
            nb.expand(-1, -1, self.bsl)  # (n, 4, 2): same nb paired with each b
        )

        bb = bb.unsqueeze(2)
        bbPxPyPzE = bbPxPyPzE.unsqueeze(2)

        # Padded-jet mask: nb pt < 0 signals a missing jet (shouldn't happen in
        # 3-jet selection by construction, but kept for downstream MdR masking).
        mask = (nb[:, 0, :] < 0)  # (n, 1)

        bPxPyPzE = PxPyPzE(b)
        nbPxPyPzE = PxPyPzE(nb)
        lPxPyPzE = PxPyPzE(l)
        regWPxPyPzE = regW_PxPyPzE.unsqueeze(-1)

        # ---- MdR matrices ----
        # bb: diagonal masked (self-pairs)
        bbMdR = matrixMdR(b, b, v1PxPyPzE=bPxPyPzE, v2PxPyPzE=bPxPyPzE)
        mask_bbMdR = self.mask_bb_same.expand(n, self.bsl, self.bsl)

        # bb-dijet vs single non-b jet
        bbnMdR = matrixMdR(bb, nb, v1PxPyPzE=bbPxPyPzE, v2PxPyPzE=nbPxPyPzE)  # (n, 2, 1, 1)

        # Lepton-nu transverse mass
        reg_nu_polar = PtEtaPhiM(reg_nu.squeeze(-1)).unsqueeze(-1)  # (n, 4, 1)
        lnu_mT = transverse_mass(l, reg_nu_polar)

        # b-jet vs single nb (hadronic top MdR -- 2 candidates)
        bWhadMdR = matrixMdR(b, nb, v1PxPyPzE=bPxPyPzE, v2PxPyPzE=nbPxPyPzE)  # (n, 2, 2, 1)
        # Take diagonal: candidate i = b_i + nb
        bWhadMdR = bWhadMdR[:, :, torch.arange(self.bsl), :]  # (n, 2, 2, 1) -> pick matching
        # Clean reshape: for each b-jet i, the (b_i, nb) MdR entry
        bWhadMdR = torch.stack(
            [bWhadMdR[:, :, i, 0] for i in range(self.bsl)], dim=-1
        ).unsqueeze(-1)  # (n, 2, bsl=2, 1)
        mask_bWhad = torch.zeros(n, self.bsl, dtype=torch.bool, device=device)

        # b-jet vs regressed leptonic W (leptonic top): reorder to match bWlep (1,0)
        bWlepMdR = matrixMdR(b, regW_lep, v1PxPyPzE=bPxPyPzE, v2PxPyPzE=regWPxPyPzE)
        bWlepMdR = bWlepMdR[:, :, (1, 0), :]  # (n, 2, bsl, 1)
        mask_bWlep = torch.zeros(n, self.bsl, dtype=torch.bool, device=device)

        # regressed W_lep vs bb
        bbWlepMdR = matrixMdR(bb, regW_lep, v1PxPyPzE=bbPxPyPzE, v2PxPyPzE=regWPxPyPzE)  # (n, 2, 1, 1)

        # regressed W_lep vs single nb (angular correlations of the two W daughters)
        WlepNBMdR = matrixMdR(regW_lep, nb, v1PxPyPzE=regWPxPyPzE, v2PxPyPzE=nbPxPyPzE)  # (n, 2, 1, 1)

        # Derived kinematics (pre-log, 7 scalars)
        mbb = bb[:, 3:4, 0:1]
        dphi_lep_met = calcDeltaPhi(l, reg_nu_polar)
        pt_bb = bb[:, 0:1, 0:1]
        dphi_bb_met = calcDeltaPhi(bb, reg_nu_polar)
        dR_nb_lepW = calcDeltaR(nb, regW_lep)            # (n, 1, 1)
        dR_nb_bb = calcDeltaR(nb, bb.squeeze(2).unsqueeze(-1))  # (n, 1, 1)
        derived_kinematics = torch.cat([
            mbb, lnu_mT, dphi_lep_met, pt_bb, dphi_bb_met, dR_nb_lepW, dR_nb_bb,
        ], dim=1)  # (n, 7, 1)

        # Log transforms on pt/mass
        nb[:, (0, 3), :] = torch.log(1 + nb[:, (0, 3), :])
        nb[isinf(nb)] = -1
        b[:, (0, 3), :] = torch.log(1 + b[:, (0, 3), :])
        bb[:, (0, 3), :] = torch.log(1 + bb[:, (0, 3), :])

        # Permutation augmentation: only b-jets have something to permute (nb is singular)
        b = torch.cat([b, b[:, :, (1, 0)]], 2)
        # Replace absolute phi with relative phi to bb-dijet
        b[:, 2:3, :] = calcDeltaPhi(bb, b[:, :, :])

        # Scalar mbb for the final_mass_head (pre-log-transform of bb pt/mass happened above)
        mbb_scalar = bb[:, 3:4, 0:1]  # (n, 1, 1), log(1+m)-transformed

        return (b, bb, a, nb, l, lnu_mT, bWhad, bWlep,
                bbMdR, bbnMdR, bWhadMdR, bWlepMdR, bbWlepMdR, WlepNBMdR,
                mask, mask_bbMdR, mask_bWhad, mask_bWlep,
                derived_kinematics, raw_nb, raw_lep,
                reg_nu, regW_lep, lepW_mass, mbb_scalar)

    def updateMeanStd(self, b, nb, l, a, reg_nu=None):
        (b, bb, a, nb, l, lnu_mT, bWhad, bWlep,
         bbMdR, bbnMdR, bWhadMdR, bWlepMdR, bbWlepMdR, WlepNBMdR,
         mask, mask_bbMdR, mask_bWhad, mask_bWlep,
         derived_kinematics, raw_nb, raw_lep,
         reg_nu, regW_lep, lepW_mass, mbb_scalar) = self.dataPrep(
            b, nb, l, a, reg_nu)

        n = b.shape[0]
        # Flatten MdR entries for stats accumulation
        MdR = torch.cat((
            bbMdR.view(n, 2, -1),
            bbnMdR.view(n, 2, -1),
            bbWlepMdR.view(n, 2, -1),
            WlepNBMdR.view(n, 2, -1),
        ), dim=2)
        mask_MdR = torch.cat((
            mask_bbMdR.view(n, -1),
            mask.view(n, -1),                                           # bbn: single nb
            torch.zeros(n, 1, dtype=torch.bool, device=b.device),       # bbWlep: no mask
            mask.view(n, -1),                                           # WlepNB: single nb
        ), dim=1)

        MdRtt = torch.cat((bWhadMdR.view(n, 2, -1), bWlepMdR.view(n, 2, -1)), dim=2)
        mask_MdRtt = torch.cat((mask_bWhad, mask_bWlep), dim=1)

        self.ancillaryEmbed.updateMeanStd(a)
        self.bJetEmbed.updateMeanStd(b)
        self.bbDiJetEmbed.updateMeanStd(bb)
        self.nonbJetEmbed.updateMeanStd(nb)
        self.MdREmbed.updateMeanStd(MdR, mask_MdR)
        self.lepEmbed.updateMeanStd(l)
        self.regressedNuEmbed.updateMeanStd(reg_nu)
        self.bWhadEmbed.updateMeanStd(bWhad.view(n, 4, -1))
        self.bWlepEmbed.updateMeanStd(bWlep.view(n, 4, -1))
        self.regWlepEmbed.updateMeanStd(regW_lep)
        self.MdRttEmbed.updateMeanStd(MdRtt, mask_MdRtt)
        self.derivedEmbed.updateMeanStd(derived_kinematics)
        self.lepWMassEmbed.updateMeanStd(lepW_mass)
        self.bbMassEmbed.updateMeanStd(mbb_scalar)

    def initMeanStd(self):
        self.ancillaryEmbed.initMeanStd()
        self.bJetEmbed.initMeanStd()
        self.bbDiJetEmbed.initMeanStd()
        self.nonbJetEmbed.initMeanStd()
        self.MdREmbed.initMeanStd()
        self.MdRttEmbed.initMeanStd()
        self.lepEmbed.initMeanStd()
        self.regressedNuEmbed.initMeanStd()
        self.bWhadEmbed.initMeanStd()
        self.bWlepEmbed.initMeanStd()
        self.regWlepEmbed.initMeanStd()
        self.derivedEmbed.initMeanStd()
        self.lepWMassEmbed.initMeanStd()
        self.bbMassEmbed.initMeanStd()

    def setGhostBatches(self, nGhostBatches, subset=False):
        self.ancillaryEmbed.setGhostBatches(nGhostBatches)
        self.bJetEmbed.setGhostBatches(nGhostBatches)
        self.bbDiJetEmbed.setGhostBatches(nGhostBatches)
        self.nonbJetEmbed.setGhostBatches(nGhostBatches)
        self.MdREmbed.setGhostBatches(nGhostBatches)
        self.MdRttEmbed.setGhostBatches(nGhostBatches)
        self.lepEmbed.setGhostBatches(nGhostBatches)
        self.regressedNuEmbed.setGhostBatches(nGhostBatches)
        self.bWhadEmbed.setGhostBatches(nGhostBatches)
        self.bWlepEmbed.setGhostBatches(nGhostBatches)
        self.regWlepEmbed.setGhostBatches(nGhostBatches)
        self.derivedEmbed.setGhostBatches(nGhostBatches)
        self.lepWMassEmbed.setGhostBatches(nGhostBatches)
        self.bbMassEmbed.setGhostBatches(nGhostBatches)

        if subset:
            return

        self.bJetConv.setGhostBatches(nGhostBatches)
        self.bbDiJetConv.setGhostBatches(nGhostBatches)
        self.nonbJetConv.setGhostBatches(nGhostBatches)
        self.MdRConv.setGhostBatches(nGhostBatches)
        self.MdRttConv.setGhostBatches(nGhostBatches)
        self.lepConv.setGhostBatches(nGhostBatches)
        self.regressedNuConv.setGhostBatches(nGhostBatches)
        self.bWhadConv.setGhostBatches(nGhostBatches)
        self.bWlepConv.setGhostBatches(nGhostBatches)
        self.regWlepConv.setGhostBatches(nGhostBatches)
        self.derivedConv.setGhostBatches(nGhostBatches)

    def forward(self, b, nb, l, a, reg_nu=None):
        (b, bb, a, nb, l, lnu_mT, bWhad, bWlep,
         bbMdR, bbnMdR, bWhadMdR, bWlepMdR, bbWlepMdR, WlepNBMdR,
         mask, mask_bbMdR, mask_bWhad, mask_bWlep,
         derived_kinematics, raw_nb, raw_lep,
         reg_nu, regW_lep, lepW_mass, mbb_scalar) = self.dataPrep(
            b, nb, l, a, reg_nu)

        n = b.shape[0]

        a_emb = self.ancillaryEmbed(a)

        # non-b jet embedding (single jet): wsl=1, permutation-augment doubles to 2
        mask_nb = torch.cat([mask, mask], dim=1) if False else mask  # no permutation for wsl=1
        nb_emb = self.nonbJetEmbed(nb, mask)
        nb_emb = nb_emb + a_emb
        nb_emb = self.nonbJetConv(NonLU(nb_emb), mask)

        # MdR branches: flatten, shared embed/conv, unflatten
        bbMdR_flat = bbMdR.view(n, 2, self.bsl * self.bsl)     # (n, 2, 4)
        bbnMdR_flat = bbnMdR.view(n, 2, self.wsl)              # (n, 2, 1)
        bbWlepMdR_flat = bbWlepMdR.view(n, 2, -1)              # (n, 2, 1)
        WlepNBMdR_flat = WlepNBMdR.view(n, 2, self.wsl)        # (n, 2, 1)

        mask_bbMdR_flat = mask_bbMdR.view(n, -1)
        mask_bbn_flat = mask.view(n, -1)
        mask_bbWlep = torch.zeros(n, 1, dtype=torch.bool, device=b.device)
        mask_WlepNB = mask.clone()

        MdR = torch.cat((bbMdR_flat, bbnMdR_flat, bbWlepMdR_flat, WlepNBMdR_flat), dim=2)
        mask_MdR = torch.cat((mask_bbMdR_flat, mask_bbn_flat, mask_bbWlep, mask_WlepNB), dim=1)
        MdR = self.MdREmbed(MdR, mask_MdR)
        MdR = self.MdRConv(NonLU(MdR), mask_MdR)

        off = 0
        n_bb = self.bsl * self.bsl
        bbMdR = MdR[:, :, off:off + n_bb].view(n, self.dD, self.bsl, self.bsl); off += n_bb
        bbnMdR = MdR[:, :, off:off + self.wsl].view(n, self.dD, 1, self.wsl); off += self.wsl
        bbWlepMdR = MdR[:, :, off:off + 1].view(n, self.dD, 1, 1); off += 1
        WlepNBMdR = MdR[:, :, off:].view(n, self.dD, 1, self.wsl)

        # TT relationship MdR (partial bWhad + bWlep, both (n, 2, bsl, 1))
        bWhadMdR_flat = bWhadMdR.view(n, 2, -1)
        bWlepMdR_flat = bWlepMdR.view(n, 2, -1)
        MdRtt = torch.cat((bWhadMdR_flat, bWlepMdR_flat), dim=2)
        mask_MdRtt = torch.cat((mask_bWhad, mask_bWlep), dim=1)
        MdRtt = self.MdRttEmbed(MdRtt, mask_MdRtt)
        MdRtt = self.MdRttConv(NonLU(MdRtt), mask_MdRtt)
        bWhadMdR = MdRtt[:, :, :self.bsl].view(n, self.dD, self.bsl, 1)
        bWlepMdR = MdRtt[:, :, self.bsl:].view(n, self.dD, self.bsl, 1)

        # Physics object embeddings
        b_emb = self.bJetEmbed(b)
        bb_emb = self.bbDiJetEmbed(bb)
        b_emb = b_emb + a_emb
        b_emb = self.bJetConv(NonLU(b_emb))
        bb_emb = self.bbDiJetConv(NonLU(bb_emb))

        l_emb = self.lepEmbed(l)
        l_emb = self.lepConv(NonLU(l_emb))

        reg_nu_emb = self.regressedNuEmbed(reg_nu)
        reg_nu_emb = self.regressedNuConv(NonLU(reg_nu_emb))

        regW_emb = self.regWlepEmbed(regW_lep)
        regW_emb = self.regWlepConv(NonLU(regW_emb))

        bWhad_emb = self.bWhadEmbed(bWhad.view(n, 4, -1), mask_bWhad)
        bWhad_emb = self.bWhadConv(NonLU(bWhad_emb), mask_bWhad)
        bWlep_emb = self.bWlepEmbed(bWlep.view(n, 4, -1))
        bWlep_emb = self.bWlepConv(NonLU(bWlep_emb))

        derived_emb = self.derivedEmbed(derived_kinematics)
        derived_emb = self.derivedConv(NonLU(derived_emb))

        lepW_mass_emb = self.lepWMassEmbed(lepW_mass)
        bb_mass_emb = self.bbMassEmbed(mbb_scalar)

        return (b_emb, bb_emb, a_emb, nb_emb, l_emb, lnu_mT,
                bWhad_emb, bWlep_emb,
                bbMdR, bbnMdR, bWhadMdR, bWlepMdR, bbWlepMdR, WlepNBMdR,
                mask_bbn_flat, mask_bWhad, mask_bWlep,
                derived_emb, reg_nu_emb, regW_emb, lepW_mass_emb, bb_mass_emb,
                raw_nb, raw_lep)


class bbWW_3jet(nn.Module):
    """
    HH->bbWW classifier for the 3-jet region (exactly 2 b-jets + 1 non-b jet).

    Differences vs bbWW_lowpt:
      - No qq dijet reconstruction, no W_had dijet attention, no pair-bias.
      - Partial hadronic top (b + single nb jet) with 2 candidates feeds the TT branch.
      - TT attention operates on 2 bWhad x 2 bWlep = 4 pairings (2 valid after same-b mask).
      - H->WW reasoning uses regressed W_lep, the single nb jet, the (regW, nb) angular
        correlation, and the lepW scalar mass.
      - final_mass_head is on (m_bb, m_lepW) instead of (m_hadW, m_lepW), since there is
        no m_hadW in this region.
    """

    def __init__(
        self,
        dijetFeatures,
        ancillaryFeatures,
        device="cuda",
        nClasses=1,
        architecture="bbWW_3jet",
    ):
        super(bbWW_3jet, self).__init__()
        self.debug = False
        self.dA = len(ancillaryFeatures)
        self.dD = dijetFeatures
        self.device = device
        self.name = architecture + "_%d" % (dijetFeatures)
        self.nC = nClasses
        self.onnx = False
        self.nGhostBatches = 64
        self.phase_symmetric = True

        self.layers = layerOrganizer()

        self.inputEmbed = InputEmbed3jet(
            self.dD, ancillaryFeatures, layers=self.layers,
            device=self.device, phase_symmetric=self.phase_symmetric,
        )

        bsl = self.inputEmbed.bsl

        # ResNet blocks (same pattern as lowpt)
        self.bbDiJetResNetBlock = ResNetBlock(
            self.dD, prefix="", nLayers=2,
            phase_symmetric=self.phase_symmetric, device=self.device,
            layers=self.layers,
            inputLayers=[self.inputEmbed.bJetConv, self.inputEmbed.bbDiJetConv],
        )
        self.lepWResNetBlock = ResNetBlock(
            self.dD, prefix="leptonic W", nLayers=2,
            phase_symmetric=self.phase_symmetric, device=self.device,
            layers=self.layers,
            inputLayers=[self.inputEmbed.lepConv, self.inputEmbed.regressedNuConv],
        )
        self.bWhadResNetBlock = ResNetBlock(
            self.dD, prefix="partial hadronic top", nLayers=2,
            phase_symmetric=self.phase_symmetric, device=self.device,
            layers=self.layers,
            inputLayers=[self.inputEmbed.bWhadConv, self.inputEmbed.bJetConv, self.inputEmbed.nonbJetConv],
        )
        self.bWlepResNetBlock = ResNetBlock(
            self.dD, prefix="leptonic top", nLayers=2,
            phase_symmetric=self.phase_symmetric, device=self.device,
            layers=self.layers,
            inputLayers=[self.inputEmbed.bWlepConv, self.inputEmbed.bJetConv, self.inputEmbed.regWlepConv],
        )

        # TT attention: 2 partial-bWhad x 2 bWlep = 4 pairings.
        # Scalars fed to the attention: [lnu_mT] (1 scalar). Kept lightweight.
        self.attention_tt = MinimalAttention(
            self.dD, heads=2, phase_symmetric=self.phase_symmetric,
            scalar_dim=1,
            layers=self.layers,
            inputLayers=[self.bWhadResNetBlock.conv[-1], self.bWlepResNetBlock.conv[-1]],
            device=self.device,
        )
        self.layers.addLayer(self.attention_tt, self.attention_tt.inputLayers)

        # qv conditioning for TT attention: concat of bWhadMdR + bWlepMdR + bbnMdR_expanded + bbWlepMdR_expanded
        # Shapes per pairing: each (n, dD, 1); 4 concatenated -> (n, 4*dD, 4 pairings)
        self.qv_embed = GhostBatchNorm1d(
            self.dD * 4, features_out=self.dD, conv=True,
            name="tt pairing qv projector",
        )

        # TT selector -> pick the winning b-as-hadronic candidate (2 options)
        self.select_tt = GhostBatchNorm1d(
            self.dD, features_out=1, conv=True, bias=False,
            name="TT pairing selector",
        )
        self.layers.addLayer(self.select_tt, [self.attention_tt])

        self.out_tt = GhostBatchNorm1d(
            self.dD, features_out=self.nC, conv=True, bias=True,
            name="TT bar score",
        )
        self.layers.addLayer(self.out_tt, [self.select_tt])

        # H->WW block: (regW_emb, nb_emb, WlepNB_MdR, lepW_mass_emb)
        self.HWWBlock = HiggsBlock(
            self.dD, n_inputs=4, phase_symmetric=self.phase_symmetric,
        )

        # Final mass head on (m_bb, m_lepW) -- replaces (m_hadW, m_lepW) from lowpt
        self.final_mass_head = nn.Sequential(
            GhostBatchNorm1d(
                2 * self.dD, features_out=self.dD,
                phase_symmetric=self.phase_symmetric, conv=True,
                name="joint mass fc1",
            ),
            NonLUModule(),
            GhostBatchNorm1d(
                self.dD, features_out=self.dD,
                phase_symmetric=self.phase_symmetric, conv=True,
                name="joint mass fc2",
            ),
            NonLUModule(),
        )

        # Final classifier: 16 pooled + dD joint-mass
        self.final_linear_layer = linear(in_channels=16 + self.dD, out_channels=self.nC)
        self.layers.addLayer(self.final_linear_layer)

        self.HH_final_embed = GhostBatchNorm1d(
            self.dD, features_out=self.dD, conv=True, name="HH final embed",
        )
        self.layers.addLayer(
            self.HH_final_embed,
            [self.inputEmbed.bJetConv, self.inputEmbed.nonbJetConv],
        )

        # Pooled event-score head (identical pattern to lowpt)
        self.out = nn.Sequential(
            GhostBatchNorm1d(
                self.dD, features_out=16, conv=True, bias=False,
                name="final event score",
            ),
            NonLUModule(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.forwardCalls = 0

    def embedding_layers(self):
        return sorted(set(self.layers.layers).difference(self.output_layers()))

    def output_layers(self):
        return [self.final_linear_layer.index]

    def updateMeanStd(self, b, nb, l, a, reg_nu=None):
        self.inputEmbed.updateMeanStd(b, nb, l, a, reg_nu)

    def initMeanStd(self):
        self.inputEmbed.initMeanStd()

    def setGhostBatches(self, nGhostBatches, subset=False):
        self.inputEmbed.setGhostBatches(nGhostBatches)
        self.bbDiJetResNetBlock.setGhostBatches(nGhostBatches)
        self.lepWResNetBlock.setGhostBatches(nGhostBatches)
        self.bWhadResNetBlock.setGhostBatches(nGhostBatches)
        self.bWlepResNetBlock.setGhostBatches(nGhostBatches)
        self.attention_tt.setGhostBatches(nGhostBatches)
        self.qv_embed.setGhostBatches(nGhostBatches)
        self.select_tt.setGhostBatches(nGhostBatches)
        self.out_tt.setGhostBatches(nGhostBatches)
        self.HH_final_embed.setGhostBatches(nGhostBatches)
        self.out[0].setGhostBatches(nGhostBatches)
        self.HWWBlock.setGhostBatches(nGhostBatches)
        self.final_mass_head[0].setGhostBatches(nGhostBatches)
        self.final_mass_head[2].setGhostBatches(nGhostBatches)
        self.nGhostBatches = nGhostBatches

    def forward(self, b, nb, l, a, reg_nu=None):
        self.forwardCalls += 1
        (b, bb, a, nb, l, lnu_mT, bWhad, bWlep,
         bbMdR, bbnMdR, bWhadMdR, bWlepMdR, bbWlepMdR, WlepNBMdR,
         mask_bbn, mask_bWhad, mask_bWlep,
         derived, reg_nu_emb, regW_emb, lepW_mass_emb, bb_mass_emb,
         raw_nb, raw_lep) = self.inputEmbed(b, nb, l, a, reg_nu)

        n = b.shape[0]
        bsl = self.inputEmbed.bsl

        b0 = b.clone()
        bb0 = bb.clone()
        l0 = l.clone()
        bWhad0 = bWhad.clone()
        bWlep0 = bWlep.clone()
        regW0 = regW_emb.clone()

        b = NonLU(b)
        bb = NonLU(bb)
        nb = NonLU(nb)
        l = NonLU(l)
        lnu_mT = NonLU(lnu_mT)
        bWhad = NonLU(bWhad)
        bWlep = NonLU(bWlep)
        regW_emb = NonLU(regW_emb)

        # ResNet refinement
        bb, bb0 = self.bbDiJetResNetBlock(b, bb, b0, bb0, debug=self.debug)
        regW_emb, regW0 = self.lepWResNetBlock(l, regW_emb, l0, regW0, debug=self.debug)
        # bWhad context = nb (the single hadronic-side jet, already conv'd)
        bWhad, bWhad0 = self.bWhadResNetBlock(
            nb.expand(-1, -1, bsl), bWhad,
            nb.expand(-1, -1, bsl), bWhad0,
            debug=self.debug,
        )
        bWlep, bWlep0 = self.bWlepResNetBlock(regW_emb, bWlep, regW0, bWlep0, debug=self.debug)

        bbMdR = NonLU(bbMdR)
        bbnMdR = NonLU(bbnMdR)

        # ============================================================
        # TT attention: 2 bWhad x 2 bWlep = 4 pairings, same-b mask -> 2 valid
        # ============================================================
        scalars = lnu_mT.squeeze(1)  # (n, 1)

        n_bWhad = bsl          # 2
        n_bWlep = bsl          # 2
        n_tt = n_bWhad * n_bWlep  # 4

        # Expand bWhadMdR over bWlep dim and vice versa
        bWhad_exp = bWhadMdR.view(n, self.dD, n_bWhad).repeat_interleave(n_bWlep, dim=2)  # (n,dD,4)
        bWlep_exp = bWlepMdR.view(n, self.dD, n_bWlep).repeat(1, 1, n_bWhad)              # (n,dD,4)
        bbn_exp   = bbnMdR.view(n, self.dD, 1).expand(-1, -1, n_tt)                       # (n,dD,4)
        bbWlep_exp = bbWlepMdR.view(n, self.dD, 1).expand(-1, -1, n_tt)                   # (n,dD,4)

        qv_tt = torch.cat([bWhad_exp, bWlep_exp, bbn_exp, bbWlep_exp], dim=1)  # (n,4*dD,4)
        qv_tt = self.qv_embed(qv_tt)

        # Mask: prevent same b-jet on hadronic & leptonic sides
        # bWhad index i uses b_i (candidate i = b_i + nb)
        # bWlep index j uses b_{1-j} (from the (1,0) reorder in dataPrep)
        # Invalid when i == 1-j  =>  i + j == 1
        mask_tt = torch.zeros(n, n_bWhad, n_bWlep, dtype=torch.bool, device=self.device)
        mask_tt[:, 0, 1] = True   # bWhad uses b0, bWlep index 1 uses b0 -> invalid
        mask_tt[:, 1, 0] = True   # bWhad uses b1, bWlep index 0 uses b1 -> invalid

        TT, TT0, TT_weights = self.attention_tt(
            bWhad, bWlep, mask_tt, bWhad0, qv_tt, scalars, debug=self.debug,
        )

        # Select winning hadronic-top candidate (2 logits, one per b-as-hadronic hypothesis)
        TT_logits = self.select_tt(TT)
        TT_logits = TT_logits.view(n, n_bWhad)
        TT_score = F.softmax(TT_logits, dim=-1)
        TT_sel = torch.matmul(TT, TT_score.unsqueeze(-1))          # (n, dD, 1)
        TT_final = self.out_tt(TT_sel)
        self._last_tt_logits = TT_logits.detach()

        # Per-jet "attention weights" output: in 3-jet (single nb) any per-jet
        # attention weight collapses to 1.0 by construction, so emit a constant
        # 1.0 tensor of shape (n, heads, 1, wsl=1) to preserve the lowpt-shaped
        # WW_score1 eval column without pretending it carries information.
        heads = self.attention_tt.h
        self._jet_weights = torch.ones(
            n, heads, 1, 1, device=self.device, dtype=bb.dtype,
        ).detach()

        # ============================================================
        # H->WW block: (regW_emb, nb_emb, WlepNB_MdR, lepW_mass_emb)
        # Physics: leptonic W + the single surviving W daughter + their
        # angular correlation + leptonic W mass.
        # ============================================================
        HWW = self.HWWBlock(
            regW_emb,
            nb,                              # (n, dD, 1) single non-b jet
            WlepNBMdR.squeeze(2),            # (n, dD, 1) regW <-> nb MdR
            lepW_mass_emb,                   # (n, dD, 1)
        )  # (n, dD, 1)

        # ============================================================
        # Final HH concatenation
        # ============================================================
        HH = torch.cat([
            bb,                              # (n, dD, 1)
            HWW,                             # (n, dD, 1)
            nb,                              # (n, dD, 1) raw hadronic-side jet
            bbMdR[:, :, 0, 1:2],             # (n, dD, 1) b-b MdR off-diagonal
            bbnMdR.squeeze(2),               # (n, dD, wsl=1)
            derived,                         # (n, dD, 1)
            bbWlepMdR.squeeze(2),            # (n, dD, 1)
            WlepNBMdR.squeeze(2),            # (n, dD, wsl=1)
        ], dim=-1)
        HH_final = self.HH_final_embed(HH)

        HH_logits_pre = torch.cat([HH_final, TT_sel], dim=-1)     # (n, dD, ...)
        pooled = self.out(HH_logits_pre)                           # (n, 16)

        # Joint mass head on (m_bb, m_lepW) -- 3-jet analog of lowpt's (m_hadW, m_lepW)
        mass_joint_in = torch.cat([bb_mass_emb, lepW_mass_emb], dim=1)  # (n, 2*dD, 1)
        mass_feat = self.final_mass_head(mass_joint_in).squeeze(-1)      # (n, dD)

        HH_logits = self.final_linear_layer(
            torch.cat([pooled, mass_feat], dim=1)
        )

        # Return (HH_logits, TT_final, nb_hadronic_proxy) -- third slot is the
        # "WW" surrogate for compatibility with the training-loop signature.
        return HH_logits, TT_final, nb
