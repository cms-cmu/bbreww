from .bbWW_models import *
from .bbWW_models import _hadW_mass

class InputEmbed(nn.Module):
    def __init__(
        self,
        dijetFeatures,
        ancillaryFeatures=["njets", "nsoftjets", "HT", "year"],
        layers=None,
        device="cuda",
        phase_symmetric=False,
        store=None,
        storeData=None,
    ):
        super(InputEmbed, self).__init__()
        self.layers = layers
        self.debug = False
        self.dD = dijetFeatures
        self.dA = len(ancillaryFeatures)
        self.ancillaryFeatures = ancillaryFeatures
        self.device = device

        self.store = None
        self.storeData = None

        if self.dA:
            self.ancillaryEmbed = GhostBatchNorm1d(
                self.dA,
                features_out=self.dD,
                phase_symmetric=phase_symmetric,
                conv=True,
                bias=False,
                name="ancillary embedder",
            )
            self.layers.addLayer(self.ancillaryEmbed)

        # b-jet embedder: (pt, eta, phi, mass, btagScore) -- phi is relative to dijet
        self.bJetEmbed = GhostBatchNorm1d(
            5, features_out=self.dD, phase_symmetric=phase_symmetric,
            conv=True, name="jet embedder",
        )
        self.bJetConv = GhostBatchNorm1d(
            self.dD, phase_symmetric=phase_symmetric, conv=True, name="jet convolution"
        )

        # non-b jet embedder: (pt, eta, phi, label, pi)
        self.nonbJetEmbed = GhostBatchNorm1d(
            5, features_out=self.dD, phase_symmetric=phase_symmetric,
            conv=True, name="attention jet embedder",
        )
        self.nonbJetConv = GhostBatchNorm1d(
            self.dD, phase_symmetric=phase_symmetric, conv=True,
            name="attention jet convolution",
        )

        # lepton embedder
        self.lepEmbed = GhostBatchNorm1d(
            6, features_out=self.dD, phase_symmetric=phase_symmetric,
            conv=True, name="lepton embedder",
        )
        self.lepConv = GhostBatchNorm1d(
            self.dD, phase_symmetric=phase_symmetric, conv=True,
            name="lepton convolution",
        )

        # Regressed neutrino 4-vector (pt, eta, phi, E) embedder
        self.regressedNuEmbed = GhostBatchNorm1d(
            4, features_out=self.dD, phase_symmetric=phase_symmetric,
            conv=True, name="regressed neutrino embedder",
        )
        self.regressedNuConv = GhostBatchNorm1d(
            self.dD, phase_symmetric=phase_symmetric, conv=True,
            name="regressed neutrino convolution",
        )

        # Hadronic top candidates: b + W->qq
        self.bWhadEmbed = GhostBatchNorm1d(
            4, features_out=self.dD, phase_symmetric=phase_symmetric,
            conv=True, name="hadronic top embedder",
        )
        self.bWhadConv = GhostBatchNorm1d(
            self.dD, phase_symmetric=phase_symmetric, conv=True,
            name="hadronic top convolution",
        )

        # Leptonic top candidates: b + regressed W_lep
        self.bWlepEmbed = GhostBatchNorm1d(
            4, features_out=self.dD, phase_symmetric=phase_symmetric,
            conv=True, name="leptonic top embedder",
        )
        self.bWlepConv = GhostBatchNorm1d(
            self.dD, phase_symmetric=phase_symmetric, conv=True,
            name="leptonic top convolution",
        )

        # MdR relationship matrices (mass + deltaR)
        self.MdREmbed = GhostBatchNorm1d(
            2, features_out=self.dD, phase_symmetric=phase_symmetric,
            conv=True, name="M(a,b), dR(a,b) embedder",
        )
        self.MdRConv = GhostBatchNorm1d(
            self.dD, phase_symmetric=phase_symmetric, conv=True,
            name="M(a,b), dR(a,b) convolution",
        )

        # TT relationship matrices (top candidates MdR)
        self.MdRttEmbed = GhostBatchNorm1d(
            2, features_out=self.dD, phase_symmetric=phase_symmetric,
            conv=True, name="ttbar relationship embedder",
        )
        self.MdRttConv = GhostBatchNorm1d(
            self.dD, phase_symmetric=phase_symmetric, conv=True,
            name="ttbar relationship convolution",
        )

        # Jet counts and combinatorics
        self.bsl, self.wsl = 2, 4
        self.qqsl = self.wsl * (self.wsl - 1) // 2  # C(4,2) = 6

        self.register_buffer('mask_bb_same', torch.zeros((1, self.bsl, self.bsl), dtype=torch.bool))
        for i in range(self.bsl):
            self.mask_bb_same[:, i, i] = 1

        # Dijet embedders
        self.bbDiJetEmbed = GhostBatchNorm1d(
            4, features_out=self.dD, phase_symmetric=phase_symmetric,
            conv=True, name="dijet embedder",
        )
        self.nonbDiJetEmbed = GhostBatchNorm1d(
            4, features_out=self.dD, phase_symmetric=phase_symmetric,
            conv=True, name="W dijet embedder",
        )
        self.bbDiJetConv = GhostBatchNorm1d(
            self.dD, phase_symmetric=phase_symmetric, conv=True,
            name="dijet convolution",
        )
        self.nonbDiJetConv = GhostBatchNorm1d(
            self.dD, phase_symmetric=phase_symmetric, conv=True,
            name="W dijet convolution",
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

        # Derived kinematics: qqsl dijet masses + mbb + lnu_mT + dphi_lep_met + pt_bb + dphi_bb_met
        self.derivedEmbed = GhostBatchNorm1d(
            5 + self.qqsl, features_out=self.dD, phase_symmetric=phase_symmetric,
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

        # Register all layers
        self.layers.addLayer(self.bJetEmbed)
        self.layers.addLayer(self.bbDiJetEmbed)
        self.layers.addLayer(self.nonbJetEmbed)
        self.layers.addLayer(self.nonbDiJetEmbed)
        self.layers.addLayer(self.MdREmbed)
        self.layers.addLayer(self.lepEmbed)
        self.layers.addLayer(self.regressedNuEmbed)
        self.layers.addLayer(self.bWhadEmbed)
        self.layers.addLayer(self.bWlepEmbed)
        self.layers.addLayer(self.regWlepEmbed)

        self.layers.addLayer(self.bJetConv, [self.bJetEmbed])
        self.layers.addLayer(self.bbDiJetConv, [self.bbDiJetEmbed])
        self.layers.addLayer(self.nonbDiJetConv, [self.nonbDiJetEmbed])
        self.layers.addLayer(self.MdRConv, [self.MdREmbed])
        self.layers.addLayer(self.nonbJetConv, [self.nonbJetEmbed])
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


    def dataPrep(self, b, nb, l, a, reg_nu=None):
        device = b.get_device() if b.get_device() >= 0 else "cpu"

        n = b.shape[0]
        b = b.view(n, 5, 2)
        nb = nb.view(n, 5, -1)               # (n, 5, wsl): pt, eta, phi, mass, attn_score
        l = l.view(n, 6, 1)
        a = a.view(n, self.dA, 1)

        # Extract per-jet attention scores (5th feature) before kinematic processing
        nb_attn = nb[:, 4, :].clone()          # (n, wsl) per-jet attention scores
        nb = nb[:, :4, :]                      # (n, 4, wsl): pt, eta, phi, mass only

        # Save raw kinematics before any transforms (needed for attention bias + hadW_mass)
        raw_nb = nb.reshape(n, -1).clone()     # (n, 4*wsl) flat
        raw_lep = l.squeeze(-1).clone()        # (n, 6)

        a[:, 2, :] = torch.log(torch.clamp(a[:, 2, :], min=1e-6))  # log transform event HT

        # Build leptonic W from regressed neutrino (px, py, pz, E)
        if reg_nu is not None:
            reg_nu = reg_nu.view(n, 4)
            nu_PxPyPzE = reg_nu  # (n, 4): px, py, pz, E

            # Leptonic W = lepton + regressed neutrino (proper 4-vector addition)
            lep_PxPyPzE = PxPyPzE(l[:, :4, 0])  # (n, 4)
            regW_PxPyPzE = lep_PxPyPzE + nu_PxPyPzE
            regW_lep = PtEtaPhiM(regW_PxPyPzE).unsqueeze(-1)  # (n, 4, 1)

            # Scalar leptonic W mass
            lepW_mass = calc_mW(l[:, :4, 0], reg_nu[:, :3]).unsqueeze(-1).unsqueeze(-1)  # (n, 1, 1)

            # Leptonic top candidates: b + regressed W_lep (2 candidates, one per b-jet)
            bWlep, bWlepPxPyPzE = addFourVectors(
                b[:, :, (1, 0)], regW_lep.expand(-1, -1, 2)
            )  # (n, 4, 2)

            reg_nu = reg_nu.unsqueeze(-1)  # (n, 4, 1) for embedding
        else:
            reg_nu = torch.zeros(n, 4, 1, device=device)
            regW_lep = torch.zeros(n, 4, 1, device=device)
            regW_PxPyPzE = torch.zeros(n, 4, device=device)
            lepW_mass = torch.zeros(n, 1, 1, device=device)
            bWlep = torch.zeros(n, 4, 2, device=device)
            bWlepPxPyPzE = torch.zeros(n, 4, 2, device=device)

        ## bb: H->bb dijet candidates
        bb, bbPxPyPzE = addFourVectors(b[:, :, (0)], b[:, :, (1)])

        # C(4,2) = 6 dijet candidates from 4 non-b jets
        qq, qqPxPyPzE = addFourVectors(
            nb[:, :, (0, 0, 0, 1, 1, 2)], nb[:, :, (1, 2, 3, 2, 3, 3)]
        )

        ## Hadronic top: 2 b-jets x qqsl dijet candidates = 2*6 = 12 candidates
        bWhad, bWhadPxPyPzE = addFourVectors(
            b[:, :, (0, 1)].unsqueeze(3),  # [batch, 4, 2, 1]
            qq.unsqueeze(2)                # [batch, 4, 1, qqsl]
        )

        bb = bb.unsqueeze(2)  # add dim for MdR matrix calculation
        bbPxPyPzE = bbPxPyPzE.unsqueeze(2)

        # Detect padded jets BEFORE appending label row
        mask = (nb[:, 0, :] < 0)  # (n, wsl): True for padded jets
        nb = torch.cat(
            [nb, torch.ones((n, 1, self.wsl), dtype=torch.float, device=device)], 1
        )
        nb[:, -1, :][mask] = -1

        # C(4,2) = 6 dijet mask
        mask_qq = torch.stack([
            mask[:, 0] | mask[:, 1],
            mask[:, 0] | mask[:, 2],
            mask[:, 0] | mask[:, 3],
            mask[:, 1] | mask[:, 2],
            mask[:, 1] | mask[:, 3],
            mask[:, 2] | mask[:, 3],
        ], dim=1)

        bPxPyPzE = PxPyPzE(b)
        nbPxPyPzE = PxPyPzE(nb)
        lPxPyPzE = PxPyPzE(l)
        regWPxPyPzE = regW_PxPyPzE.unsqueeze(-1)  # (n, 4, 1)

        # ---- MdR matrices ----
        # b-jet pair
        bbMdR = matrixMdR(b, b, v1PxPyPzE=bPxPyPzE, v2PxPyPzE=bPxPyPzE)
        mask_bbMdR = self.mask_bb_same.expand(n, self.bsl, self.bsl)

        # bb-dijet vs individual non-b jets
        bbnMdR = matrixMdR(bb, nb, v1PxPyPzE=bbPxPyPzE, v2PxPyPzE=nbPxPyPzE)

        # bb-dijet vs qq-dijets
        bbqqMdR = matrixMdR(bb, qq, v1PxPyPzE=bbPxPyPzE, v2PxPyPzE=qqPxPyPzE)

        lepQQdR = calcDeltaR(l, qq)
        mask_bbn = mask.view(n, 1, self.wsl)

        # non-b jet pair mass and deltaR matrix
        qqMdR = matrixMdR(nb, nb, v1PxPyPzE=nbPxPyPzE, v2PxPyPzE=nbPxPyPzE)

        # Lepton-regressed-nu transverse mass
        # reg_nu is (px, py, pz, E) — convert to (pt, eta, phi, M) for transverse_mass
        reg_nu_polar = PtEtaPhiM(reg_nu.squeeze(-1)).unsqueeze(-1)  # (n, 4, 1)
        lnu_mT = transverse_mass(l, reg_nu_polar)

        mask_qqMdR = mask.view(n, 1, self.wsl) | mask.view(n, self.wsl, 1)

        # b-jet vs W candidates (hadronic top)
        bWhadMdR = matrixMdR(b, qq, v1PxPyPzE=bPxPyPzE, v2PxPyPzE=qqPxPyPzE)
        mask_bWhad = mask_qq.repeat_interleave(self.bsl, dim=1)  # (n, 2*qqsl)

        # b-jet vs regressed leptonic W (leptonic top): (n, 2, bsl, 1)
        bWlepMdR = matrixMdR(b, regW_lep, v1PxPyPzE=bPxPyPzE, v2PxPyPzE=regWPxPyPzE)
        bWlepMdR = bWlepMdR[:, :, (1, 0), :]  # reorder: b1+W, b0+W -> match bWlep order
        mask_bWlep = torch.zeros(n, self.bsl, dtype=torch.bool, device=device)

        # regressed W_lep vs bb-dijet (HH topology)
        bbWlepMdR = matrixMdR(bb, regW_lep, v1PxPyPzE=bbPxPyPzE, v2PxPyPzE=regWPxPyPzE)  # (n, 2, 1, 1)

        # regressed W_lep vs individual non-b jets (angular correlations)
        WlepNBMdR = matrixMdR(regW_lep, nb, v1PxPyPzE=regWPxPyPzE, v2PxPyPzE=nbPxPyPzE)  # (n, 2, 1, wsl)

        # Derived kinematics (computed before log-transforms)
        mjj_all = qq[:, 3:4, :]
        mbb = bb[:, 3:4, 0:1]
        dphi_lep_met = calcDeltaPhi(l, reg_nu_polar)
        pt_bb = bb[:, 0:1, 0:1]
        dphi_bb_met = calcDeltaPhi(bb, reg_nu_polar)
        derived_kinematics = torch.cat([
            mjj_all.transpose(1, 2),  # (batch, qqsl, 1)
            mbb, lnu_mT, dphi_lep_met, pt_bb, dphi_bb_met,
        ], dim=1)  # (batch, 5+qqsl, 1)

        # Log transforms
        nb[:, (0, 3), :] = torch.log(1 + nb[:, (0, 3), :])
        nb[isinf(nb)] = -1

        b[:, (0, 3), :] = torch.log(1 + b[:, (0, 3), :])
        bb[:, (0, 3), :] = torch.log(1 + bb[:, (0, 3), :])
        qq[:, (0, 3), :] = torch.log(1 + qq[:, (0, 3), :])

        # Permutation invariance augmentation
        b = torch.cat([b, b[:, :, (1,0)]], 2)
        nb = torch.cat([nb, nb[:, :, (3,2,1,0)]], 2)

        # Replace absolute phi with relative phi to bb-dijet
        b[:, 2:3, :] = calcDeltaPhi(bb, b[:, :, :])

        return (b, bb, qq, a, nb, l, lnu_mT, bWhad, bWlep, lepQQdR,
                bbMdR, qqMdR, bbnMdR, bbqqMdR, bWhadMdR, bWlepMdR,
                bbWlepMdR, WlepNBMdR,
                mask, mask_bbMdR, mask_qqMdR, mask_bbn, mask_qq, mask_bWhad, mask_bWlep,
                derived_kinematics, raw_nb, raw_lep,
                reg_nu, regW_lep, lepW_mass, nb_attn)

    def updateMeanStd(self, b, nb, l, a, reg_nu=None):
        (b, bb, qq, a, nb, l, lnu_mT, bWhad, bWlep, lepQQdR,
         bbMdR, qqMdR, bbnMdR, bbqqMdR, bWhadMdR, bWlepMdR,
         bbWlepMdR, WlepNBMdR,
         mask, mask_bbMdR, mask_qqMdR, mask_bbn, mask_qq, mask_bWhad, mask_bWlep,
         derived_kinematics, raw_nb, raw_lep,
         reg_nu, regW_lep, lepW_mass, nb_attn) = self.dataPrep(
            b, nb, l, a, reg_nu)

        n = b.shape[0]
        qqsl = self.qqsl
        MdR = torch.cat((
            bbMdR.view(n, 2, -1),
            qqMdR.view(n, 2, -1),
            bbnMdR.view(n, 2, -1),
            bbqqMdR.view(n, 2, -1),
            bbWlepMdR.view(n, 2, -1),
            WlepNBMdR.view(n, 2, -1),
        ), dim=2)
        mask_MdR = torch.cat((
            mask_bbMdR.view(n, -1),
            mask_qqMdR.view(n, -1),
            mask_bbn.view(n, -1),
            mask_qq.view(n, -1),
            torch.zeros(n, 1, dtype=torch.bool, device=b.device),  # bbWlep: no mask
            mask_bbn.view(n, -1),  # WlepNB: same mask as bbn
        ), dim=1)

        MdRtt = torch.cat((bWhadMdR.view(n, 2, -1), bWlepMdR.view(n, 2, -1)), dim=2)
        mask_MdRtt = torch.cat((mask_bWhad, mask_bWlep.view(n, -1)), dim=1)

        bWhad = bWhad.view(n, 4, -1)
        bWlep = bWlep.view(n, 4, -1)

        self.ancillaryEmbed.updateMeanStd(a)
        self.bJetEmbed.updateMeanStd(b)
        self.bbDiJetEmbed.updateMeanStd(bb)
        self.nonbJetEmbed.updateMeanStd(nb)
        self.nonbDiJetEmbed.updateMeanStd(qq)
        self.MdREmbed.updateMeanStd(MdR, mask_MdR)
        self.lepEmbed.updateMeanStd(l)
        self.regressedNuEmbed.updateMeanStd(reg_nu)
        self.bWhadEmbed.updateMeanStd(bWhad)
        self.bWlepEmbed.updateMeanStd(bWlep)
        self.regWlepEmbed.updateMeanStd(regW_lep)
        self.MdRttEmbed.updateMeanStd(MdRtt, mask_MdRtt)
        self.derivedEmbed.updateMeanStd(derived_kinematics)
        self.lepWMassEmbed.updateMeanStd(lepW_mass)

    def initMeanStd(self):
        self.ancillaryEmbed.initMeanStd()
        self.bJetEmbed.initMeanStd()
        self.bbDiJetEmbed.initMeanStd()
        self.nonbJetEmbed.initMeanStd()
        self.nonbDiJetEmbed.initMeanStd()
        self.MdREmbed.initMeanStd()
        self.MdRttEmbed.initMeanStd()
        self.lepEmbed.initMeanStd()
        self.regressedNuEmbed.initMeanStd()
        self.bWhadEmbed.initMeanStd()
        self.bWlepEmbed.initMeanStd()
        self.regWlepEmbed.initMeanStd()
        self.derivedEmbed.initMeanStd()
        self.lepWMassEmbed.initMeanStd()

    def setGhostBatches(self, nGhostBatches, subset=False):
        self.ancillaryEmbed.setGhostBatches(nGhostBatches)
        self.bJetEmbed.setGhostBatches(nGhostBatches)
        self.bbDiJetEmbed.setGhostBatches(nGhostBatches)
        self.nonbJetEmbed.setGhostBatches(nGhostBatches)
        self.nonbDiJetEmbed.setGhostBatches(nGhostBatches)
        self.MdREmbed.setGhostBatches(nGhostBatches)
        self.MdRttEmbed.setGhostBatches(nGhostBatches)
        self.lepEmbed.setGhostBatches(nGhostBatches)
        self.regressedNuEmbed.setGhostBatches(nGhostBatches)
        self.bWhadEmbed.setGhostBatches(nGhostBatches)
        self.bWlepEmbed.setGhostBatches(nGhostBatches)
        self.regWlepEmbed.setGhostBatches(nGhostBatches)
        self.derivedEmbed.setGhostBatches(nGhostBatches)
        self.lepWMassEmbed.setGhostBatches(nGhostBatches)

        if subset:
            return

        self.bJetConv.setGhostBatches(nGhostBatches)
        self.bbDiJetConv.setGhostBatches(nGhostBatches)
        self.nonbJetConv.setGhostBatches(nGhostBatches)
        self.nonbDiJetConv.setGhostBatches(nGhostBatches)
        self.MdRConv.setGhostBatches(nGhostBatches)
        self.MdRttConv.setGhostBatches(nGhostBatches)
        self.lepConv.setGhostBatches(nGhostBatches)
        self.regressedNuConv.setGhostBatches(nGhostBatches)
        self.bWhadConv.setGhostBatches(nGhostBatches)
        self.bWlepConv.setGhostBatches(nGhostBatches)
        self.regWlepConv.setGhostBatches(nGhostBatches)
        self.derivedConv.setGhostBatches(nGhostBatches)

    def forward(self, b, nb, l, a, reg_nu=None):
        (b, bb, qq, a, nb, l, lnu_mT, bWhad, bWlep, lepQQdR,
         bbMdR, qqMdR, bbnMdR, bbqqMdR, bWhadMdR, bWlepMdR,
         bbWlepMdR, WlepNBMdR,
         mask, mask_bbMdR, mask_qqMdR, mask_bbn, mask_qq, mask_bWhad, mask_bWlep,
         derived_kinematics, raw_nb, raw_lep,
         reg_nu, regW_lep, lepW_mass, nb_attn) = self.dataPrep(
            b, nb, l, a, reg_nu)

        a = self.ancillaryEmbed(a)
        mask_nb = torch.cat([mask, mask[:, list(reversed(range(self.wsl)))]], 1)
        nb = self.nonbJetEmbed(nb, mask_nb)
        qq = self.nonbDiJetEmbed(qq)
        nb = nb + a
        nb = self.nonbJetConv(NonLU(nb), mask_nb)

        n = bb.shape[0]
        qqsl = self.qqsl

        # Flatten and embed MdR matrices (including new regW relationships)
        bbMdR_flat = bbMdR.view(n, 2, self.bsl * self.bsl)
        qqMdR_flat = qqMdR.view(n, 2, self.wsl * self.wsl)
        bbnMdR_flat = bbnMdR.view(n, 2, self.wsl)
        bbqqMdR_flat = bbqqMdR.view(n, 2, qqsl)
        bbWlepMdR_flat = bbWlepMdR.view(n, 2, -1)       # (n, 2, 1)
        WlepNBMdR_flat = WlepNBMdR.view(n, 2, self.wsl)  # (n, 2, wsl)

        mask_bbMdR_flat = mask_bbMdR.view(n, -1)
        mask_qqMdR_flat = mask_qqMdR.view(n, -1)
        mask_bbn_flat = mask_bbn.view(n, -1)
        mask_bbWlep = torch.zeros(n, 1, dtype=torch.bool, device=b.device)
        mask_WlepNB = mask.clone()

        MdR = torch.cat((bbMdR_flat, qqMdR_flat, bbnMdR_flat, bbqqMdR_flat, bbWlepMdR_flat, WlepNBMdR_flat), dim=2)
        mask_MdR = torch.cat((mask_bbMdR_flat, mask_qqMdR_flat, mask_bbn_flat, mask_qq, mask_bbWlep, mask_WlepNB), dim=1)
        MdR = self.MdREmbed(MdR, mask_MdR)
        MdR = self.MdRConv(NonLU(MdR), mask_MdR)

        # Unflatten
        off = 0
        n_bb = self.bsl * self.bsl
        bbMdR = MdR[:, :, off:off + n_bb].view(n, self.dD, self.bsl, self.bsl); off += n_bb
        n_qq = self.wsl * self.wsl
        qqMdR = MdR[:, :, off:off + n_qq].view(n, self.dD, self.wsl, self.wsl); off += n_qq
        bbnMdR = MdR[:, :, off:off + self.wsl].view(n, self.dD, 1, self.wsl); off += self.wsl
        bbqqMdR = MdR[:, :, off:off + qqsl].view(n, self.dD, 1, qqsl); off += qqsl
        bbWlepMdR = MdR[:, :, off:off + 1].view(n, self.dD, 1, 1); off += 1
        WlepNBMdR = MdR[:, :, off:].view(n, self.dD, 1, self.wsl)

        # TT relationship matrices (bWhad + bWlep)
        bWhadMdR = bWhadMdR.view(n, 2, -1)
        bWlepMdR = bWlepMdR.view(n, 2, -1)
        MdRtt = torch.cat((bWhadMdR, bWlepMdR), dim=2)
        mask_MdRtt = torch.cat((mask_bWhad, mask_bWlep.view(n, -1)), dim=1)
        MdRtt = self.MdRttEmbed(MdRtt, mask_MdRtt)
        MdRtt = self.MdRttConv(NonLU(MdRtt), mask_MdRtt)

        bWhadMdR = MdRtt[:, :, :self.bsl * qqsl].view(n, self.dD, self.bsl, qqsl)
        bWlepMdR = MdRtt[:, :, self.bsl * qqsl:].view(n, self.dD, self.bsl, 1)

        # Embed physics objects
        b = self.bJetEmbed(b)
        bb = self.bbDiJetEmbed(bb)
        b = b + a
        b = self.bJetConv(NonLU(b))
        bb = self.bbDiJetConv(NonLU(bb))

        l = self.lepEmbed(l)
        l = self.lepConv(NonLU(l))

        # Regressed neutrino embedding (4-vector: pt, eta, phi, m=0)
        reg_nu_emb = self.regressedNuEmbed(reg_nu)
        reg_nu_emb = self.regressedNuConv(NonLU(reg_nu_emb))

        # Regressed leptonic W embedding (4-vector: pt, eta, phi, m)
        regW_emb = self.regWlepEmbed(regW_lep)
        regW_emb = self.regWlepConv(NonLU(regW_emb))

        # Top reconstruction embeddings
        bWhad = self.bWhadEmbed(bWhad.view(n, 4, -1), mask_bWhad)
        bWhad = self.bWhadConv(NonLU(bWhad), mask_bWhad)
        bWlep = self.bWlepEmbed(bWlep.view(n, 4, -1))  # b + regressed W_lep (2 candidates)
        bWlep = self.bWlepConv(NonLU(bWlep))

        # Derived kinematics
        derived = self.derivedEmbed(derived_kinematics)
        derived = self.derivedConv(NonLU(derived))

        lepW_mass_emb = self.lepWMassEmbed(lepW_mass)

        return (b, bb, qq, a, nb, l, lnu_mT, bWhad, bWlep, lepQQdR,
                bbMdR, qqMdR, bbnMdR, bbqqMdR, bWhadMdR, bWlepMdR,
                bbWlepMdR, WlepNBMdR,
                mask_bbn, mask_qq, mask_bWhad, mask_bWlep,
                derived, reg_nu_emb, regW_emb, lepW_mass_emb,
                raw_nb, raw_lep, nb_attn)


class HCR_lowpt(nn.Module):
    def __init__(
        self,
        dijetFeatures,
        ancillaryFeatures,
        device="cuda",
        nClasses=1,
        architecture="HCR",
    ):
        super(HCR_lowpt, self).__init__()
        self.debug = False
        self.dA = len(ancillaryFeatures)
        self.dD = dijetFeatures
        self.device = device
        self.name = architecture + "_%d" % (dijetFeatures)
        self.nC = nClasses
        self.store = None
        self.storeData = {}
        self.onnx = False
        self.nGhostBatches = 64
        self.phase_symmetric = True

        self.layers = layerOrganizer()

        self.inputEmbed = InputEmbed(
            self.dD,
            ancillaryFeatures,
            layers=self.layers,
            device=self.device,
            phase_symmetric=self.phase_symmetric,
        )

        # ResNet blocks for feature refinement
        self.bbDiJetResNetBlock = ResNetBlock(
            self.dD, prefix="", nLayers=2,
            phase_symmetric=self.phase_symmetric, device=self.device,
            layers=self.layers,
            inputLayers=[self.inputEmbed.bJetConv, self.inputEmbed.bbDiJetConv],
        )
        self.nonbDiJetResNetBlock = ResNetBlock(
            self.dD, prefix="", nLayers=2,
            phase_symmetric=self.phase_symmetric, device=self.device,
            layers=self.layers,
            inputLayers=[self.inputEmbed.nonbJetConv, self.inputEmbed.nonbDiJetConv],
        )
        # Leptonic W ResNet: reinforces regressed W embedding using lepton and regressed nu
        self.lepWResNetBlock = ResNetBlock(
            self.dD, prefix="leptonic W", nLayers=2,
            phase_symmetric=self.phase_symmetric, device=self.device,
            layers=self.layers,
            inputLayers=[self.inputEmbed.lepConv, self.inputEmbed.regressedNuConv],
        )
        self.bWhadResNetBlock = ResNetBlock(
            self.dD, prefix="hadronic top", nLayers=2,
            phase_symmetric=self.phase_symmetric, device=self.device,
            layers=self.layers,
            inputLayers=[self.inputEmbed.bWhadConv, self.inputEmbed.bJetConv, self.inputEmbed.nonbDiJetConv],
        )
        self.bWlepResNetBlock = ResNetBlock(
            self.dD, prefix="leptonic top", nLayers=2,
            phase_symmetric=self.phase_symmetric, device=self.device,
            layers=self.layers,
            inputLayers=[self.inputEmbed.bWlepConv, self.inputEmbed.bJetConv, self.inputEmbed.regWlepConv],
        )

        # Single-jet WW attention: regressed leptonic W queries individual non-b jets
        qqsl = self.inputEmbed.qqsl
        wsl = self.inputEmbed.wsl
        bsl = self.inputEmbed.bsl
        self.attention_WW = MinimalAttention(
            self.dD, heads=2, phase_symmetric=self.phase_symmetric,
            scalar_dim=qqsl + 1,  # 6 lepQQdR + 1 lnu_mT = 7
            layers=self.layers,
            inputLayers=[self.lepWResNetBlock.conv[-1], self.inputEmbed.nonbJetConv],
            device=self.device,
        )
        self.layers.addLayer(self.attention_WW, self.attention_WW.inputLayers)

        # TT attention: hadronic tops query leptonic tops
        # All 6 qq pairs used -> 2*6=12 hadronic x 2 leptonic = 24 pairings
        self.attention_tt = MinimalAttention(
            self.dD, heads=2, phase_symmetric=self.phase_symmetric,
            scalar_dim=qqsl + 1,  # 6 lepQQdR + 1 lnu_mT = 7
            layers=self.layers,
            inputLayers=[self.bWhadResNetBlock.conv[-1], self.bWlepResNetBlock.conv[-1]],
            device=self.device,
        )
        self.layers.addLayer(self.attention_tt, self.attention_tt.inputLayers)

        self.scalars_embed = GhostBatchNorm1d(
            qqsl + 1, features_out=self.dD, conv=True,
            name="scalar physics relationships embed"
        )

        self.qv_embed = GhostBatchNorm1d(
            self.dD * 5, features_out=self.dD, conv=True,
            name="qv physics relationships projector"
        )

        # Single-jet attention bias modules
        self.jet_dR_embed = GhostBatchNorm1d(
            1, features_out=self.dD, conv=True, name="jet deltaR embedder"
        )
        self.jet_mjj_embed = GhostBatchNorm1d(
            1, features_out=self.dD, conv=True, name="jet dijet mass embedder"
        )
        self.jet_attn_embed = GhostBatchNorm1d(
            1, features_out=self.dD, conv=True, name="external jet attn score embedder"
        )
        self.qv_combine = GhostBatchNorm1d(
            3 * self.dD, features_out=self.dD, conv=True, name="qv combine deltaR+mjj+attn"
        )

        # Hadronic W mass from attention-selected jets
        self.hadW_mass_embed = GhostBatchNorm1d(
            1, features_out=self.dD, conv=True, name="hadronic W mass embedder"
        )

        self.select_tt = GhostBatchNorm1d(
            self.dD, features_out=1, conv=True, bias=False,
            name="TT pairing selector"
        )
        self.layers.addLayer(self.select_tt, [self.attention_tt])

        self.select_WW = GhostBatchNorm1d(
            self.dD, features_out=1, conv=True, bias=False,
            name="non-bjet selector"
        )
        self.layers.addLayer(self.select_WW, [self.attention_WW])

        self.none_WW_score = GhostBatchNorm1d(
            self.dD, features_out=1, conv=True,
            name="WW rejection scorer"
        )
        self.layers.addLayer(self.none_WW_score, [self.attention_WW])

        self.out_tt = GhostBatchNorm1d(
            self.dD, features_out=self.nC, conv=True, bias=True,
            name="TT bar score"
        )
        self.layers.addLayer(self.out_tt, [self.select_tt])

        # H->WW block: 2-layer MLP with residual, full interaction between
        # hadronic W (from attention) and leptonic W (regressed) + both masses
        # Learns conditional patterns like off-shell lepW + on-shell hadW = signal
        self.HWWBlock = HiggsBlock(
            self.dD, n_inputs=4, phase_symmetric=self.phase_symmetric,
        )

        self.final_linear_layer = linear(in_channels=16, out_channels=self.nC)
        self.layers.addLayer(self.final_linear_layer)

        self.HH_final_embed = GhostBatchNorm1d(
            self.dD, features_out=self.dD, conv=True, name="HH final embed"
        )
        self.layers.addLayer(self.HH_final_embed, [self.inputEmbed.bJetConv, self.select_WW])

        self.out = nn.Sequential(
            GhostBatchNorm1d(
                self.dD, features_out=16, conv=True, bias=False,
                name="final event score"
            ),
            NonLUModule(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            self.final_linear_layer
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
        self.nonbDiJetResNetBlock.setGhostBatches(nGhostBatches)
        self.lepWResNetBlock.setGhostBatches(nGhostBatches)
        self.bWhadResNetBlock.setGhostBatches(nGhostBatches)
        self.bWlepResNetBlock.setGhostBatches(nGhostBatches)
        self.attention_WW.setGhostBatches(nGhostBatches)
        self.attention_tt.setGhostBatches(nGhostBatches)
        self.scalars_embed.setGhostBatches(nGhostBatches)
        self.qv_embed.setGhostBatches(nGhostBatches)
        self.select_tt.setGhostBatches(nGhostBatches)
        self.select_WW.setGhostBatches(nGhostBatches)
        self.none_WW_score.setGhostBatches(nGhostBatches)
        self.out_tt.setGhostBatches(nGhostBatches)
        self.HH_final_embed.setGhostBatches(nGhostBatches)
        self.out[0].setGhostBatches(nGhostBatches)
        self.jet_dR_embed.setGhostBatches(nGhostBatches)
        self.jet_mjj_embed.setGhostBatches(nGhostBatches)
        self.jet_attn_embed.setGhostBatches(nGhostBatches)
        self.qv_combine.setGhostBatches(nGhostBatches)
        self.hadW_mass_embed.setGhostBatches(nGhostBatches)
        self.HWWBlock.setGhostBatches(nGhostBatches)
        self.nGhostBatches = nGhostBatches

    def forward(self, b, nb, l, a, reg_nu=None):
        self.forwardCalls += 1
        (b, bb, qq, a, nb, l, lnu_mT, bWhad, bWlep, lepQQdR,
         bbMdR, qqMdR, bbnMdR, bbqqMdR, bWhadMdR, bWlepMdR,
         bbWlepMdR, WlepNBMdR,
         mask_bbn, mask_qq, mask_bWhad, mask_bWlep,
         derived, reg_nu_emb, regW_emb, lepW_mass_emb,
         raw_nb, raw_lep, nb_attn) = self.inputEmbed(
            b, nb, l, a, reg_nu
        )

        n = b.shape[0]
        wsl = self.inputEmbed.wsl     # 4
        qqsl = self.inputEmbed.qqsl   # 6
        bsl = self.inputEmbed.bsl     # 2

        # Save pre-activation copies for residual connections
        b0 = b.clone()
        bb0 = bb.clone()
        nb0 = nb.clone()
        qq0 = qq.clone()
        l0 = l.clone()
        bWhad0 = bWhad.clone()
        bWlep0 = bWlep.clone()
        regW0 = regW_emb.clone()

        b = NonLU(b)
        bb = NonLU(bb)
        nb = NonLU(nb)
        qq = NonLU(qq)
        l = NonLU(l)
        lnu_mT = NonLU(lnu_mT)
        bWhad = NonLU(bWhad)
        bWlep = NonLU(bWlep)
        regW_emb = NonLU(regW_emb)

        # ResNet blocks: refine features
        bb, bb0 = self.bbDiJetResNetBlock(b, bb, b0, bb0, debug=self.debug)
        qq, qq0 = self.nonbDiJetResNetBlock(nb, qq, nb0, qq0, debug=self.debug)

        # Leptonic W: refine regressed W embedding using lepton and regressed nu context
        regW_emb, regW0 = self.lepWResNetBlock(l, regW_emb, l0, regW0, debug=self.debug)

        bWhad, bWhad0 = self.bWhadResNetBlock(
            qq.repeat_interleave(2, dim=2),
            bWhad,
            qq0.repeat_interleave(2, dim=2),
            bWhad0,
            debug=self.debug)
        # Leptonic top: b + regressed W_lep, reinforced with regressed W context
        bWlep, bWlep0 = self.bWlepResNetBlock(regW_emb, bWlep, regW0, bWlep0, debug=self.debug)

        bbMdR = NonLU(bbMdR)
        qqMdR = NonLU(qqMdR)
        bbnMdR = NonLU(bbnMdR)
        bbqqMdR = NonLU(bbqqMdR)

        # ============================================================
        # TT attention: all 6 qq pairs, 2*6=12 hadronic x 2 leptonic = 24 pairings
        # ============================================================
        scalars = torch.cat([lepQQdR, lnu_mT], dim=-1).squeeze(1)

        n_bWhad = bsl * qqsl     # 12
        n_bWlep = bsl            # 2 (one per b-jet)
        n_tt = n_bWhad * n_bWlep # 24

        bWhad_exp = bWhadMdR.reshape(n, -1, n_bWhad).repeat_interleave(n_bWlep, dim=2)
        bWlep_exp = bWlepMdR.squeeze(-1).repeat(1, 1, n_bWhad)  # (n, dD, 24)

        # Map each qq pair to its constituent non-b jets for bbn relationship
        bbn_flat = bbnMdR.squeeze(2)  # (n, dD, wsl=4)
        qq_idx = []
        for i in range(wsl):
            for j in range(i + 1, wsl):
                qq_idx.append((i, j))
        bbn_qq = torch.cat([
            torch.cat([bbn_flat[:, :, i:i+1], bbn_flat[:, :, j:j+1]], dim=1)
            for i, j in qq_idx
        ], dim=2)  # (n, 2*dD, qqsl=6)
        bbn_exp = bbn_qq.repeat_interleave(n_bWlep, dim=2).repeat(1, 1, bsl)  # (n, 2*dD, 24)

        bbqq_exp = bbqqMdR.squeeze(2)  # (n, dD, qqsl=6)
        bbqq_exp = bbqq_exp.repeat_interleave(n_bWlep, dim=2).repeat(1, 1, bsl)  # (n, dD, 24)

        qv_tt = torch.cat([bWhad_exp, bWlep_exp, bbn_exp, bbqq_exp], dim=1)
        qv_tt = self.qv_embed(qv_tt)

        # Mask: prevent same b-jet in both hadronic and leptonic top
        # bWhad: indices [0:qqsl] use b0, [qqsl:2*qqsl] use b1
        # bWlep: index 0 uses b1, index 1 uses b0
        mask_tt = torch.zeros(n, n_bWhad, n_bWlep, dtype=torch.bool, device=self.device)
        mask_tt[:, :qqsl, 0] = True       # b0 hadronic × b1 leptonic -> wait, bWlep[0] = b1+W, so b0 had x b1 lep is VALID
        # bWlep order is (b1+W, b0+W) from dataPrep: b[:, :, (1, 0)]
        # So bWlep[0] has b1, bWlep[1] has b0
        # bWhad[0:qqsl] has b0, bWhad[qqsl:] has b1
        # Invalid: b0_had x b0_lep = bWhad[0:qqsl] x bWlep[1]
        #          b1_had x b1_lep = bWhad[qqsl:]  x bWlep[0]
        mask_tt[:, :qqsl, 1] = True   # b0 hadronic × b0 leptonic (invalid)
        mask_tt[:, qqsl:, 0] = True   # b1 hadronic × b1 leptonic (invalid)

        TT, TT0, TT_weights = self.attention_tt(
            bWhad, bWlep, mask_tt, bWhad0, qv_tt, scalars, debug=self.debug
        )

        # TTbar pairing selection
        TT_logits = self.select_tt(TT)
        TT_logits = TT_logits.view(n, n_bWhad)
        TT_score = F.softmax(TT_logits, dim=-1)
        TT_sel = torch.matmul(TT, TT_score.unsqueeze(-1))
        TT_final = self.out_tt(TT_sel)
        self._last_tt_logits = TT_logits.detach()

        # ============================================================
        # Single-jet WW attention: regressed leptonic W queries individual jets
        # ============================================================
        nb_jets = nb[:, :, :wsl]  # (n, dD, 4)
        jet_mask = mask_bbn.view(n, wsl)

        # Physics-aware attention bias
        nb_raw_4 = raw_nb.view(n, 4, -1)[:, :, :wsl]
        lep_raw_6 = raw_lep.view(n, 6, 1)
        lepNBdR = calcDeltaR(lep_raw_6, nb_raw_4)
        jet_dR = self.jet_dR_embed(lepNBdR, jet_mask)

        jet_mjj = compute_mjj(raw_nb, wsl)
        jet_mjj = self.jet_mjj_embed(jet_mjj, jet_mask)

        # External attention scores as per-jet bias (packed in nonbJetCand)
        ext_attn = nb_attn.view(n, 1, wsl)
        jet_attn = self.jet_attn_embed(ext_attn, jet_mask)

        jet_qv = self.qv_combine(torch.cat([jet_dR, jet_mjj, jet_attn], dim=1), jet_mask)

        WW, WW0, WW_weights = self.attention_WW(
            regW_emb,                 # q: (n, dD, 1) regressed leptonic W
            nb_jets,                  # v: (n, dD, wsl=4) individual jets
            jet_mask.unsqueeze(1),    # mask: (n, 1, wsl)
            regW0,                    # q0: residual
            jet_qv,                   # qv: physics-aware bias
            scalars,                  # 6 lepQQdR + 1 lnu_mT = 7
            self.debug
        )
        self._jet_weights = WW_weights.detach()

        # Hadronic W mass from attention-selected jets
        hadW_mass = _hadW_mass(raw_nb, WW_weights.detach())
        hadW_mass_emb = self.hadW_mass_embed(hadW_mass)

        # ============================================================
        # H->WW block: full interaction between hadronic and leptonic W
        # ============================================================
        # main=regW_emb (residual connection), context=WW, hadW_mass, lepW_mass
        HWW = self.HWWBlock(regW_emb, WW, hadW_mass_emb, lepW_mass_emb)  # (n, dD, 1)

        # ============================================================
        # Final HH concatenation (convolution over all features)
        # HWW output replaces raw individual W features -- it already encodes their interactions
        # ============================================================
        scalars_emb = self.scalars_embed(scalars.unsqueeze(-1))
        HH = torch.cat([
            bb,                           # (n, dD, 1) -- H->bb
            HWW,                          # (n, dD, 1) -- H->WW (learned W-W interactions)
            WW,                           # (n, dD, 1) -- raw hadronic W from attention
            bbMdR[:, :, 0, 1:2],          # (n, dD, 1) -- bb mass/dR
            bbnMdR.squeeze(2),            # (n, dD, wsl=4)
            qqMdR.view(n, self.dD, -1),   # (n, dD, wsl*wsl=16)
            scalars_emb,                  # (n, dD, 1)
            derived,                      # (n, dD, 1) -- derived kinematics
            bbWlepMdR.squeeze(2),         # (n, dD, 1) -- bb vs regW MdR
            WlepNBMdR.squeeze(2),         # (n, dD, wsl=4) -- regW vs non-b jets MdR
        ], dim=-1)
        HH_final = self.HH_final_embed(HH)

        HH_logits = torch.cat([HH_final, TT_sel], dim=-1)
        HH_logits = self.out(HH_logits)

        return HH_logits, TT_final, WW

    def setStore(self, store):
        self.store = store
        self.inputEmbed.store = store
        self.inputEmbed.storeData = self.storeData

    def writeStore(self):
        print(self.store)
        np.save(self.store, self.storeData)
