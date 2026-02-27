from .bbWW_models import *

# helper: signed-log W mass quadratic discriminant for classifier
def _w_mass_discriminant(lep_pt, lep_eta, lep_phi, lep_mass, nu_px, nu_py):
    lep_px = lep_pt * torch.cos(lep_phi)
    lep_py = lep_pt * torch.sin(lep_phi)
    lep_pz = lep_pt * torch.sinh(lep_eta)
    lep_E = torch.sqrt(lep_px**2 + lep_py**2 + lep_pz**2 + lep_mass**2)
    A = (lep_px * nu_px + lep_py * nu_py) + (80.379**2 - lep_E**2 + lep_px**2 + lep_py**2 + lep_pz**2) / 2
    C = lep_E**2 - lep_pz**2
    met_pt_sq = nu_px**2 + nu_py**2
    disc = (2 * A * lep_pz)**2 - 4 * (lep_E**2 * met_pt_sq - A**2) * C
    return (torch.sign(disc) * torch.log1p(torch.abs(disc))).unsqueeze(1)  # (n, 1, 1)

# helper function: per-event Cholesky factors for both hypotheses
def _build_cholesky(chol_raw):
    n = chol_raw.shape[0]
    L = torch.zeros(n, 3, 3, device=chol_raw.device, dtype=chol_raw.dtype)
    L[:, 0, 0] = F.softplus(chol_raw[:, 0]).clamp(min= 1e-6, max=200.0)
    L[:, 1, 0] = chol_raw[:, 1]
    L[:, 1, 1] = F.softplus(chol_raw[:, 2]).clamp(min= 1e-6, max=200.0)
    L[:, 2, 0] = chol_raw[:, 3]
    L[:, 2, 1] = chol_raw[:, 4]
    L[:, 2, 2] = F.softplus(chol_raw[:, 5]).clamp(min= 1e-6, max=200.0)
    return L

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
                self.dA, # + 1 test feature: uncomment in next iteration
                features_out=self.dD,
                phase_symmetric=phase_symmetric,
                conv=True,
                bias=False,
                name="ancillary embedder",
            )
            self.layers.addLayer(self.ancillaryEmbed)
            # self.ancillaryConv  = GhostBatchNorm1d(self.dD, phase_symmetric=phase_symmetric, conv=True, name='Ancillary Convolution')
            # self.layers.addLayer(self.ancillaryConv, [self.ancillaryEmbed])
        ## to do section
        # embed inputs to dijetResNetBlock in target feature space
        self.bJetEmbed = GhostBatchNorm1d(
            6,
            features_out=self.dD,
            phase_symmetric=phase_symmetric,
            conv=True,
            name="jet embedder",
        )  # phi is relative to dijet

        self.bJetConv = GhostBatchNorm1d(
            self.dD, 
            phase_symmetric=phase_symmetric, 
            conv=True, 
            name="jet convolution"
        )
        self.nonbJetEmbed = GhostBatchNorm1d(
            5,
            features_out=self.dD,
            phase_symmetric=phase_symmetric,
            conv=True,
            name="attention jet embedder",
        ) 
        self.nonbJetConv = GhostBatchNorm1d(
            self.dD,
            phase_symmetric=phase_symmetric,
            conv=True,
            name="attention jet convolution",
        )
        self.lepEmbed = GhostBatchNorm1d(
            6,
            features_out=self.dD,
            phase_symmetric=phase_symmetric,
            conv=True,
            name="lepton embedder",
        )
        self.lepConv = GhostBatchNorm1d(
            self.dD,
            phase_symmetric=phase_symmetric,
            conv=True,
            name="lepton convolution",
        )
        self.nuEmbed = GhostBatchNorm1d(
            2,
            features_out=self.dD,
            phase_symmetric=phase_symmetric,
            conv=True,
            name="MET embedder",
        )
        self.nuConv = GhostBatchNorm1d(
            self.dD,
            phase_symmetric=phase_symmetric,
            conv=True,
            name="MET convolution",
        )
        self.bWhadEmbed = GhostBatchNorm1d(
            4,
            features_out = self.dD,
            phase_symmetric=phase_symmetric,
            conv=True,
            name="hadronic top embedder",
        )
        self.bWhadConv = GhostBatchNorm1d(
            self.dD,
            phase_symmetric=phase_symmetric,
            conv=True,
            name="hadronic top convolution",
        )
        self.bWlepEmbed = GhostBatchNorm1d(
            4,
            features_out = self.dD,
            phase_symmetric=phase_symmetric,
            conv=True,
            name="leptonic top embedder",
        )
        self.bWlepConv = GhostBatchNorm1d(
            self.dD,
            phase_symmetric=phase_symmetric,
            conv=True,
            name="leptonic top convolution",
        )
        self.MdRttEmbed = GhostBatchNorm1d(
            2,
            features_out = self.dD,
            phase_symmetric=phase_symmetric,
            conv=True,
            name="ttbar relationship embedder",
        )
        self.MdRttConv = GhostBatchNorm1d(
            self.dD,
            phase_symmetric=phase_symmetric,
            conv=True,
            name="ttbar relationship convolution",
        )
        self.MdREmbed = GhostBatchNorm1d(
            2,
            features_out=self.dD,
            phase_symmetric=phase_symmetric,
            conv=True,
            name="M(a,b), dR(a,b) embedder",
        )
        self.MdRConv = GhostBatchNorm1d(
            self.dD,
            phase_symmetric=phase_symmetric,
            conv=True,
            name="M(a,b), dR(a,b) convolution",
        )

        self.bsl, self.wsl = 2, 3

        self.register_buffer('mask_bb_same', torch.zeros((1, self.bsl, self.bsl), dtype=torch.bool))
        for i in range(self.bsl):
            self.mask_bb_same[:, i, i] = (
                1  # mask diagonal, don't want mass, dR of jet with itself. (we do want duplicates for i,j and j,i because query and value are treated differently in attention block)
            )

        self.bbDiJetEmbed = GhostBatchNorm1d(
            4,
            features_out=self.dD,
            phase_symmetric=phase_symmetric,
            conv=True,
            name="dijet embedder",
        )  # phi is relative do dijet

        self.nonbDiJetEmbed = GhostBatchNorm1d(
            4,
            features_out=self.dD,
            phase_symmetric=phase_symmetric,
            conv=True,
            name="W dijet embedder",
        )
        self.bbDiJetConv = GhostBatchNorm1d(
            self.dD,
            phase_symmetric=phase_symmetric,
            conv=True,
            name="dijet convolution",
        )
        self.nonbDiJetConv = GhostBatchNorm1d(
            self.dD,
            phase_symmetric=phase_symmetric,
            conv=True,
            name="W dijet convolution",
        )

        self.solnEmbed = GhostBatchNorm1d(
            6,  # pz1, pz2, oss80, pz3, pz4, oss40
            features_out=self.dD,
            phase_symmetric=phase_symmetric,
            conv=True,
            name="kinematic solutions embedder",
        )
        self.solnConv = GhostBatchNorm1d(
            self.dD,
            phase_symmetric=phase_symmetric,
            conv=True,
            name="kinematic solutions convolution",
        )

        self.derivedEmbed = GhostBatchNorm1d(
            8,
            features_out=self.dD,
            phase_symmetric=phase_symmetric,
            conv=True,
            name="derived kinematics embedder",
        )
        self.derivedConv = GhostBatchNorm1d(
            self.dD,
            phase_symmetric=phase_symmetric,
            conv=True,
            name="derived kinematics convolution",
        )

        self.layers.addLayer(self.bJetEmbed)
        self.layers.addLayer(self.bbDiJetEmbed)
        self.layers.addLayer(self.nonbJetEmbed)
        self.layers.addLayer(self.nonbDiJetEmbed)
        self.layers.addLayer(self.MdREmbed)
        self.layers.addLayer(self.lepEmbed)
        self.layers.addLayer(self.nuEmbed)
        self.layers.addLayer(self.bWhadEmbed)
        self.layers.addLayer(self.bWlepEmbed)

        self.layers.addLayer(self.bJetConv, [self.bJetEmbed])
        self.layers.addLayer(self.bbDiJetConv, [self.bbDiJetEmbed])
        self.layers.addLayer(self.nonbDiJetConv, [self.nonbDiJetEmbed])
        self.layers.addLayer(self.MdRConv, [self.MdREmbed])
        self.layers.addLayer(self.nonbJetConv, [self.nonbJetEmbed])
        self.layers.addLayer(self.lepConv, [self.lepEmbed])
        self.layers.addLayer(self.nuConv, [self.nuEmbed])
        self.layers.addLayer(self.bWhadConv, [self.bWhadEmbed])
        self.layers.addLayer(self.bWlepConv, [self.bWlepEmbed])
        self.layers.addLayer(self.MdRttEmbed)
        self.layers.addLayer(self.MdRttConv, [self.MdRttEmbed])
        self.layers.addLayer(self.solnEmbed)
        self.layers.addLayer(self.solnConv, [self.solnEmbed])
        self.layers.addLayer(self.derivedEmbed)
        self.layers.addLayer(self.derivedConv, [self.derivedEmbed])


    def dataPrep(self, b, nb, l, nu, a):  # , device='cuda'):
        device = b.get_device() if b.get_device() >= 0 else "cpu"
        # # if device=='cpu': # prevent overwritting data from dataloader when doing operations directly from RAM rather than copying to VRAM
        # j = j.clone()
        # o = o.clone()
        # a = a.clone()

        n = b.shape[0]
        b = b.view(n, 5, 2)
        nb = nb.view(n, 4, -1)
        l = l.view(n, 6, 1)
        nu = nu.view(n, 2, 1)
        a = a.view(n, self.dA, 1)

        a[:, 2, :] = torch.log(torch.clamp(a[:, 2, :], min=1e-6))  # avoid log(0)

        #reconstruct leptonic W by solving MET pz with W mass constraint
        W_lep1, W_lep2, oss_80, pz1_80, pz2_80 = get_lepW(l[:, :4], nu)
        W_lep = torch.cat([W_lep1, W_lep2], dim=2)
        Ws_lep1, Ws_lep2, oss_40, pz1_40, pz2_40 = get_lepW(l[:, :4], nu, mW = 40.0)
        Ws_lep = torch.cat([Ws_lep1, Ws_lep2], dim=2)

        # kinematic solutions: (batch, 6, 1)
        kinematic_solutions = torch.cat([
            pz1_80, pz2_80, oss_80,
            pz1_40, pz2_40, oss_40,
        ], dim=1)  # each is (n, 1, 1) → cat along dim=1 → (n, 6, 1)

        ## bb: H->bb dijet candidates, qq: W->qq dijet candidates"
        bb, bbPxPyPzE = addFourVectors(
            b[:, :, (0)], b[:, :, (1)]
        )

        qq, qqPxPyPzE = addFourVectors(
            nb[:, :, (0, 0, 1)], nb[:, :, (1, 2, 2)]
        )

        ## top reconstruction
        bWhad, bWhadPxPyPzE = addFourVectors(
            b[:, :, (0, 1)].unsqueeze(3),  # [batch, 4, 2, 1]
            qq.unsqueeze(2)                # [batch, 4, 1, 3]
        )
        bWlep, bWlepPxPyPzE = addFourVectors(
            b[:, :, (1, 1, 0, 0)],
            W_lep[:, :, (0, 1, 0, 1)] 
        )

        bb = bb.unsqueeze(2) # add a dimension to calculating MdR matrix symmetrically later
        bbPxPyPzE = bbPxPyPzE.unsqueeze(2)

        mask, bbMdR, qqMdR, bbnMdR, mask_bbMdR, mask_qqMdR, mask_bbn = None, None, None, None, None, None, None
        b = torch.cat(
            [b, 2 * torch.ones((n, 1, 2), dtype=torch.float, device=device)], 1
        )  # label bJets with 2 (-1 for mask, 0 for not preselected, 1 for preselected jet)
        nb = torch.cat(
            [nb, torch.ones((n, 1, 3), dtype=torch.float, device=device)], 1
        ) 
        mask = (nb[:, 3, :] == -1)
        mask_qq = torch.stack([
            mask[:, 0] | mask[:, 1],  # qq[0] = nb[0] + nb[1]
            mask[:, 0] | mask[:, 2],  # qq[1] = nb[0] + nb[2]
            mask[:, 1] | mask[:, 2],  # qq[2] = nb[1] + nb[2]
        ], dim=1) # mask for di-jet candidates involving padded entries

        bPxPyPzE = PxPyPzE(b)
        nbPxPyPzE = PxPyPzE(nb)
        lPxPyPzE = PxPyPzE(l)

        # For b-jets: compute matrix of dijet masses and opening angles between other jets
        n = bb.shape[0]
        bbMdR = matrixMdR(b, b, v1PxPyPzE=bPxPyPzE, v2PxPyPzE=bPxPyPzE)
        mask_bbMdR = self.mask_bb_same.expand(n, self.bsl, self.bsl)

        # compute matrix of trijet masses and opening angles between b-dijets and non-bjets
        bbnMdR = matrixMdR(bb, nb, v1PxPyPzE=bbPxPyPzE, v2PxPyPzE=nbPxPyPzE)
     
        # compute matrix of quadjet masses and opening angles between b-dijets and qq-dijets
        bbqqMdR = matrixMdR(bb, qq, v1PxPyPzE=bbPxPyPzE, v2PxPyPzE=qqPxPyPzE)

        lepQQdR = calcDeltaR(l, qq)
        mask_bbn = mask.view(n, 1, self.wsl)

        # For nonb-jets: compute matrix of dijet masses and opening angles between other jets
        n = qq.shape[0]
        qqMdR = matrixMdR(nb, nb, v1PxPyPzE=nbPxPyPzE, v2PxPyPzE=nbPxPyPzE)

        # For lepton and MET, compute transverse mass
        lnu_mT = transverse_mass(l, nu)

        mask_qqMdR = mask.view(n, 1, self.wsl) | mask.view(
            n, self.wsl, 1
        )  # mask of 2d matrix of nonb-jets (i,j) is True if mask[i] | mask[j]

        # compute matrix of masses and opening angles between b-jets and W candidates (top)
        bWhadMdR = matrixMdR(b, qq, v1PxPyPzE=bPxPyPzE, v2PxPyPzE=qqPxPyPzE)
        mask_bWhad = mask_qq.repeat_interleave(self.bsl, dim=1)  # shape: (n, 6)

        bWlepMdR = matrixMdR(b, l.unsqueeze(2), v1PxPyPzE=bPxPyPzE, v2PxPyPzE=lPxPyPzE) # l needs an extra dimension for concat later
        bWlepMdR = bWlepMdR[:, :, (1, 1, 0, 0), :]  # Expand from 2 to 4 candidates
        mask_bWlep = torch.zeros(n, self.bsl * 2, dtype=torch.bool, device=device) # nothing to mask 

        nb[:, (0, 3), :] = torch.log(1 + nb[:, (0, 3), :])
        nb[isinf(nb)] = -1  # isinf not supported by ONNX

        b[:, (0, 3), :] = torch.log(1 + b[:, (0, 3), :])
        bb[:, (0, 3), :] = torch.log(1 + bb[:, (0, 3), :])
        qq[:, (0, 3), :] = torch.log(1 + qq[:, (0, 3), :])

        b = torch.cat([b, b[:, :, (1,0)]] , 2) # create permutation invariance by augmenting opposite order of same jets
        nb = torch.cat([nb, nb[:, :, (2,1,0)]] , 2)

        # only keep relative angular information so that learned features are invariant under global phi rotations and eta/phi flips
        b[:, 2:3, :] = calcDeltaPhi(bb, b[:, :, :]) # replace jet phi with deltaPhi between dijet and jet

        # Extract derived kinematics as a (batch, 8, 1) tensor
        mjj_01 = qq[:, 3:4, 0:1]  # third W candidate mass (jets 0,1)
        mjj_02 = qq[:, 3:4, 1:2]  # second W candidate mass (jets 0,2)
        mjj_12 = qq[:, 3:4, 2:3]  # first W candidate mass (jets 1,2)
        mbb = bb[:, 3:4, 0:1]     # bb mass

        dphi_lep_met = calcDeltaPhi(l, nu)
        pt_bb = bb[:, 0:1, 0:1]
        dphi_bb_met = calcDeltaPhi(bb, nu)

        derived_kinematics = torch.cat([
            mjj_01, mjj_02, mjj_12,  # 3 W mass candidates
            mbb,                      # Higgs mass candidate
            lnu_mT,                   # transverse mass
            dphi_lep_met,             # angular separation lep-MET
            pt_bb,                    # pT of bb system
            dphi_bb_met               # angular separation bb-MET
        ], dim=1)  # Shape: (batch, 8, 1)

        return b, bb, qq, a, nb , l, nu, lnu_mT, bWhad, bWlep, lepQQdR, bbMdR, qqMdR, bbnMdR, bbqqMdR, bWhadMdR, bWlepMdR, mask, mask_bbMdR, mask_qqMdR, mask_bbn, mask_qq, mask_bWhad, mask_bWlep, derived_kinematics, kinematic_solutions

    def updateMeanStd(self,  b, nb, l, nu, a):
        if b.shape[0] == 0: # guard against empty batches from random initialization
            return
        (b, bb, qq, a, nb , l, nu, lnu_mT, bWhad, bWlep, lepQQdR, bbMdR, qqMdR, bbnMdR, bbqqMdR, bWhadMdR, bWlepMdR,
        mask, mask_bbMdR, mask_qqMdR, mask_bbn, mask_qq, mask_bWhad, mask_bWlep, derived_kinematics, kinematic_solutions) = self.dataPrep(
                                                                        b, nb, l, nu, a)

        n, self.bsl, self.wsl = b.shape[0], b.shape[2] // 2, nb.shape[2] // 2 # need to half the third dimension because we repeated all the jets
        MdR = torch.cat(
            (
                bbMdR.view(n, 2, -1),
                qqMdR.view(n, 2, -1),
                bbnMdR.view(n, 2, -1),
                bbqqMdR.view(n, 2, -1)
            ),
            dim=2,
        )
        mask_MdR = torch.cat(
            (
                mask_bbMdR.view(n, -1),
                mask_qqMdR.view(n, -1),
                mask_bbn.view(n, -1),
                mask_qq.view(n, -1) # mask_qq works for bbqqMdR
            ),
            dim=1,
        )
        
        MdRtt = torch.cat(
            (
                bWhadMdR.view(n, 2, -1),
                bWlepMdR.view(n, 2, -1),
            ),
            dim=2,
        )

        mask_MdRtt = torch.cat(
            (
                mask_bWhad,  # (n, 6)
                mask_bWlep.view(n, -1)  # (n, 2)
            ),
            dim=1
        )  # Result: (n, 8)


        bWhad = bWhad.view(n, 4, -1)  # (n, 4, 2, 3) -> (n, 4, 6)
        bWlep = bWlep.view(n, 4, -1)  # (n, 4, 2, 1) -> (n, 4, 2)
        
        self.ancillaryEmbed.updateMeanStd(a)
        self.bJetEmbed.updateMeanStd(b)
        self.bbDiJetEmbed.updateMeanStd(bb)
        self.nonbJetEmbed.updateMeanStd(nb)
        self.nonbDiJetEmbed.updateMeanStd(qq)
        self.MdREmbed.updateMeanStd(MdR, mask_MdR)
        self.lepEmbed.updateMeanStd(l)
        self.nuEmbed.updateMeanStd(nu)
        self.bWlepEmbed.updateMeanStd(bWlep)
        self.bWhadEmbed.updateMeanStd(bWhad)
        self.MdRttEmbed.updateMeanStd(MdRtt, mask_MdRtt)
        self.solnEmbed.updateMeanStd(kinematic_solutions)
        self.derivedEmbed.updateMeanStd(derived_kinematics)

    def initMeanStd(self):
        self.ancillaryEmbed.initMeanStd()
        self.bJetEmbed.initMeanStd()
        self.bbDiJetEmbed.initMeanStd()
        self.nonbJetEmbed.initMeanStd()
        self.nonbDiJetEmbed.initMeanStd()
        self.MdREmbed.initMeanStd()
        self.MdRttEmbed.initMeanStd()
        self.lepEmbed.initMeanStd()
        self.nuEmbed.initMeanStd()
        self.bWhadEmbed.initMeanStd()
        self.bWlepEmbed.initMeanStd()
        self.solnEmbed.initMeanStd()
        self.derivedEmbed.initMeanStd()

    def setGhostBatches(self, nGhostBatches, subset=False):
        self.ancillaryEmbed.setGhostBatches(nGhostBatches)
        self.bJetEmbed.setGhostBatches(nGhostBatches)
        self.bbDiJetEmbed.setGhostBatches(nGhostBatches)
        self.nonbJetEmbed.setGhostBatches(nGhostBatches)
        self.nonbDiJetEmbed.setGhostBatches(nGhostBatches)
        self.MdREmbed.setGhostBatches(nGhostBatches)
        self.MdRttEmbed.setGhostBatches(nGhostBatches)
        self.lepEmbed.setGhostBatches(nGhostBatches)
        self.nuEmbed.setGhostBatches(nGhostBatches)
        self.bWhadEmbed.setGhostBatches(nGhostBatches)
        self.bWlepEmbed.setGhostBatches(nGhostBatches)
        self.solnEmbed.setGhostBatches(nGhostBatches)
        self.derivedEmbed.setGhostBatches(nGhostBatches)

        if subset:
            return

        self.bJetConv.setGhostBatches(nGhostBatches)
        self.bbDiJetConv.setGhostBatches(nGhostBatches)
        self.nonbJetConv.setGhostBatches(nGhostBatches)
        self.nonbDiJetConv.setGhostBatches(nGhostBatches)
        self.MdRConv.setGhostBatches(nGhostBatches)
        self.MdRttConv.setGhostBatches(nGhostBatches)
        self.lepConv.setGhostBatches(nGhostBatches)
        self.nuConv.setGhostBatches(nGhostBatches)
        self.bWhadConv.setGhostBatches(nGhostBatches)
        self.bWlepConv.setGhostBatches(nGhostBatches)
        self.solnConv.setGhostBatches(nGhostBatches)
        self.derivedConv.setGhostBatches(nGhostBatches)

    def forward(self, b, nb, l, nu, a):
        (b, bb, qq, a, nb , l, nu, lnu_mT, bWhad, bWlep, lepQQdR, bbMdR, qqMdR, bbnMdR, bbqqMdR, bWhadMdR, bWlepMdR,
        mask, mask_bbMdR, mask_qqMdR, mask_bbn, mask_qq, mask_bWhad, mask_bWlep, derived_kinematics, kinematic_solutions) = self.dataPrep(b, nb, l, nu, a)

        a = self.ancillaryEmbed(a)
        # a = self.ancillaryConv(NonLU(a))
        mask_nb =  torch.cat([mask, mask[:, [2,1,0]]], 1) # augment mask from 2 to 4, matching pattern for jets
        nb = self.nonbJetEmbed(nb, mask_nb)
        qq = self.nonbDiJetEmbed(qq)
        nb = nb + a
        nb = self.nonbJetConv(NonLU(nb), mask_nb)
        # print('o after conv a\n',o[0])
        # o = o+o0

        n = bb.shape[0]

        # bbMdR is (n, 3, bsl, wsl)
        # flatten the matrices for passing through convolution
        bbMdR = bbMdR.view(n, 2, self.bsl*self.bsl)
        qqMdR = qqMdR.view(n, 2, self.wsl*self.wsl)
        bbnMdR = bbnMdR.view(n, 2, self.wsl)
        bbqqMdR = bbqqMdR.view(n, 2, self.wsl)        
        mask_bbMdR = mask_bbMdR.view(n, -1)
        mask_qqMdR = mask_qqMdR.view(n, -1)
        mask_bbn = mask_bbn.view(n, -1)
        MdR = torch.cat((bbMdR, qqMdR, bbnMdR, bbqqMdR), dim=2)
        mask_MdR = torch.cat((mask_bbMdR, mask_qqMdR, mask_bbn, mask_qq), dim=1) # Higgs masses and dijets information
        # MdPhi is (n, 3, osl*osl + dsl*osl)
        MdR = self.MdREmbed(MdR, mask_MdR)
        MdR = self.MdRConv(NonLU(MdR), mask_MdR)

        # get back original shape (equivalent to unflatten)
        bbMdR = MdR[:, :, : self.bsl * self.bsl].view(
            n, self.dD, self.bsl, self.bsl
        )
        qqMdR = MdR[:, :, self.bsl * self.bsl : self.bsl * self.bsl + self.wsl * self.wsl ].view(
            n, self.dD, self.wsl, self.wsl
        )
        bbnMdR = MdR[:, :, self.bsl * self.bsl + self.wsl * self.wsl : self.bsl * self.bsl + self.wsl * self.wsl + self.wsl].view(
            n, self.dD, 1, self.wsl
        )
        bbqqMdR = MdR[:, :,  self.bsl * self.bsl + self.wsl * self.wsl + self.wsl :].view(
            n, self.dD, 1, self.wsl
        )

        
        bWhadMdR = bWhadMdR.view(n, 2, -1)
        bWlepMdR = bWlepMdR.view(n, 2, -1)
        MdRtt = torch.cat((bWhadMdR, bWlepMdR), dim=2)
        mask_MdRtt = torch.cat(
            (
                mask_bWhad,
                mask_bWlep.view(n, -1)
            ),
            dim=1
        )

        MdRtt = self.MdRttEmbed(MdRtt, mask_MdRtt)
        MdRtt = self.MdRttConv(NonLU(MdRtt), mask_MdRtt)

        bWhadMdR = MdRtt[:, :, :self.bsl* self.wsl].view(
            n, self.dD, self.bsl, self.wsl
        )
        bWlepMdR = MdRtt[:, :, self.bsl*self.wsl:].view(
            n, self.dD, self.bsl * 2, 1
        )

        b = self.bJetEmbed(b)
        bb = self.bbDiJetEmbed(bb)
        b = b + a
        b = self.bJetConv(NonLU(b))
        bb = self.bbDiJetConv(NonLU(bb))

        l = self.lepEmbed(l)
        nu = self.nuEmbed(nu)
        l = self.lepConv(NonLU(l))
        nu = self.nuConv(NonLU(nu))

        # top reconstruction
        bWhad = self.bWhadEmbed(bWhad.view(n, 4, -1), mask_bWhad)
        bWlep = self.bWlepEmbed(bWlep.view(n, 4, -1))
        bWhad = self.bWhadConv(NonLU(bWhad), mask_bWhad)
        bWlep = self.bWlepConv(NonLU(bWlep))

        derived = self.derivedEmbed(derived_kinematics)
        derived = self.derivedConv(NonLU(derived))

        soln = self.solnEmbed(kinematic_solutions)
        soln = self.solnConv(NonLU(soln))

        return b, bb, qq, a, nb , l, nu, lnu_mT, bWhad, bWlep, lepQQdR, bbMdR, qqMdR, bbnMdR, bbqqMdR, bWhadMdR, bWlepMdR, mask_bbMdR, mask_qqMdR, mask_bbn, mask_qq, mask_bWhad, mask_bWlep, derived, soln, kinematic_solutions
  

class OnShellClassifier(nn.Module):
    """Binary classifier for on-shell vs off-shell leptonic W.

    Routes events to the appropriate neutrino regressor at inference time.
    Trained with BCE on isLepW labels.

    Input: onshell_input + oss_80 + oss_40 + disc_feat = 2*dD + 12
    """

    def __init__(self, dD):
        super().__init__()
        # Project enriched input (2*dD + 12) down to dD
        self.input_embed = GhostBatchNorm1d(2*dD + 12, features_out=dD, conv=True)
        # Two residual blocks
        self.block1 = nn.Sequential(
            GhostBatchNorm1d(dD, features_out=dD, conv=True),
            NonLUModule(),
            GhostBatchNorm1d(dD, features_out=dD, conv=True),
            NonLUModule(),
        )
        self.block2 = nn.Sequential(
            GhostBatchNorm1d(dD, features_out=dD, conv=True),
            NonLUModule(),
            GhostBatchNorm1d(dD, features_out=dD, conv=True),
            NonLUModule(),
        )
        # Binary classifier: p(on-shell)
        self.classifier = nn.Sequential(
            GhostBatchNorm1d(dD, features_out=dD, conv=True),
            NonLUModule(),
            GhostBatchNorm1d(dD, features_out=1, conv=True),
        )

    def forward(self, classifier_input):
        # classifier_input: (n, 2*dD+12, 1)
        x = self.input_embed(classifier_input)
        x = x + self.block1(x)
        x = x + self.block2(x)
        logit_onshell = self.classifier(x).squeeze(-1).squeeze(-1)  # (n,): raw logit
        return logit_onshell
      
class METRegressor(nn.Module):
    def __init__(
        self,
        dijetFeatures,
        ancillaryFeatures,
        device="cuda",
        architecture="HCR",
    ):
        super(METRegressor, self).__init__()
        self.debug = False
        self.dA = len(ancillaryFeatures)
        self.dD = dijetFeatures  # dimension of embeded   dijet feature space
        self.device = device
        self.name = (
            architecture
            + "_%d" % (dijetFeatures)
        )
        self.store = None
        self.storeData = {}
        self.onnx = False
        self.nGhostBatches = 64
        self.phase_symmetric = True

        self.layers = layerOrganizer()

        # this module handles input shifting scaling and learns the optimal scale and shift for the appropriate inputs
        self.inputEmbed = InputEmbed(
            self.dD,
            ancillaryFeatures,
            layers=self.layers,
            device=self.device,
            phase_symmetric=self.phase_symmetric,
        )

        # Stride=3 Kernel=3 reinforce dijet features, in parallel update jet features for next reinforce layer
        # |1|2|1,2|3|4|3,4|1|3|1,3|2|4|2,4|1|4|1,4|2|3|2,3|
        #     |1,2|   |3,4|   |1,3|   |2,4|   |1,4|   |2,3|
        self.bbDiJetResNetBlock = ResNetBlock(
            self.dD,
            prefix="",
            nLayers=2,
            phase_symmetric=self.phase_symmetric,
            device=self.device,
            layers=self.layers,
            inputLayers=[self.inputEmbed.bJetConv, self.inputEmbed.bbDiJetConv],
        )
        self.nonbDiJetResNetBlock = ResNetBlock(
            self.dD,
            prefix="",
            nLayers=2,
            phase_symmetric=self.phase_symmetric,
            device=self.device,
            layers=self.layers,
            inputLayers=[self.inputEmbed.nonbJetConv, self.inputEmbed.nonbDiJetConv],
        )
        self.lepWResNetBlock = ResNetBlock(
            self.dD,
            prefix="leptonic W",
            nLayers=2,
            phase_symmetric=self.phase_symmetric,
            device=self.device,
            layers=self.layers,
            inputLayers=[self.inputEmbed.lepConv, self.inputEmbed.nuConv],
        )
        self.bWhadResNetBlock = ResNetBlock(
            self.dD,
            prefix="hadronic top",
            nLayers=2,
            phase_symmetric=self.phase_symmetric,
            device=self.device,
            layers=self.layers,
            inputLayers=[self.inputEmbed.bWhadConv, self.inputEmbed.bJetConv, self.inputEmbed.nonbDiJetConv],
        )
        self.bWlepResNetBlock = ResNetBlock(
            self.dD,
            prefix="leptonic top",
            nLayers=2,
            phase_symmetric=self.phase_symmetric,
            device=self.device,
            layers=self.layers,
            inputLayers=[self.inputEmbed.bWlepConv, self.inputEmbed.bJetConv, self.inputEmbed.lepConv],
        )

        self.attention_WW = MinimalAttention(
            self.dD,
            heads=2,
            phase_symmetric=self.phase_symmetric,
            scalar_dim = 4,
            layers=self.layers,
            inputLayers=[self.lepWResNetBlock.conv[-1], self.inputEmbed.nonbJetConv],
            device=self.device,
        )
        self.layers.addLayer(self.attention_WW, self.attention_WW.inputLayers)

        self.attention_tt = MinimalAttention(
            self.dD,
            heads=2,
            phase_symmetric=self.phase_symmetric,
            layers=self.layers,
            scalar_dim = 4,
            inputLayers=[self.bWhadResNetBlock.conv[-1], self.bWlepResNetBlock.conv[-1]],
            device=self.device,
        )
        self.layers.addLayer(self.attention_tt, self.attention_tt.inputLayers)

        self.scalars_embed = GhostBatchNorm1d(
            4, 
            features_out=self.dD,
            conv=True, 
            name="scalar physics relationships embed"
        )

        self.qv_embed = GhostBatchNorm1d(
            self.dD*5,  # Input: full feature dim (40)
            features_out=8,  # Output: heads * head_dim = 2 * 4
            conv=True,
            name="qv physics relationships projector"
        )
        self.select_tt = GhostBatchNorm1d(
            self.dD,
            features_out=1,  # Single score per candidate
            conv=True,
            bias=False,  # No bias because softmax is translation invariant
            name="TT pairing selector"
        )
        self.layers.addLayer(self.select_tt, [self.attention_tt])

        # Embed lepton-jet deltaR for attention bias (qv)
        self.jet_dR_embed = GhostBatchNorm1d(
            1,
            features_out=self.dD,
            conv=True,
            name="jet deltaR embedder",
        )

        self.onshell_classifier = OnShellClassifier(self.dD)

        dH = self.dD * 4  # wider hidden dim for regressor heads

        # On-shell neutrino regressor: outputs (dpx, dpy)
        # +dD: lep_W0, +2: init_px/py, +1: lnu_mT, +6: jet_weights = 2*dD+9
        self.nu_regressor_onshell = nn.Sequential(
            GhostBatchNorm1d(2*self.dD + 9, features_out=dH, conv=True),   # expand
            NonLUModule(),
            GhostBatchNorm1d(dH, features_out=dH, conv=True),
            NonLUModule(),
            GhostBatchNorm1d(dH, features_out=dH, conv=True),
            NonLUModule(),
            GhostBatchNorm1d(dH, features_out=dH, conv=True),
            NonLUModule(),
            GhostBatchNorm1d(dH, features_out=dH, conv=True),
            NonLUModule(),
            GhostBatchNorm1d(dH, features_out=2, conv=True),         # dpx, dpy
        )
        # pz solution selector: onshell_input + rapidity gaps + |eta_nu| + oss_corrected
        # Runs after px/py correction so features use corrected pz solutions
        # +dD: lep_W0, +2: init_px/py, +1: lnu_mT, +6: jet_weights,
        # +2: deta_sol1/2, +2: |eta_nu_sol1/2|, +1: log1p(oss_corrected) = 2*dD+14
        self.pz_selector = nn.Sequential(
            GhostBatchNorm1d(2*self.dD + 14, features_out=dH, conv=True),
            NonLUModule(),
            GhostBatchNorm1d(dH, features_out=dH, conv=True),
            NonLUModule(),
            GhostBatchNorm1d(dH, features_out=1, conv=True),         # logit_sol
        )
        # Off-shell neutrino regressor: context + lep_W0 + 4 pz solutions + lnu_mT + jet_weights
        # +dD: lep_W0, +4: pz solutions, +1: lnu_mT, +6: jet_weights (2 heads * 3 jets) = 2*dD+11
        # Extra hidden layer vs on-shell head: unconstrained 3D regression is harder
        self.nu_regressor_offshell = nn.Sequential(
            GhostBatchNorm1d(2*self.dD + 11, features_out=dH, conv=True),  # expand
            NonLUModule(),
            GhostBatchNorm1d(dH, features_out=dH, conv=True),
            NonLUModule(),
            GhostBatchNorm1d(dH, features_out=dH, conv=True),
            NonLUModule(),
            GhostBatchNorm1d(dH, features_out=dH, conv=True),
            NonLUModule(),
            GhostBatchNorm1d(dH, features_out=dH, conv=True),
            NonLUModule(),
            GhostBatchNorm1d(dH, features_out=dH, conv=True),
            NonLUModule(),
            GhostBatchNorm1d(dH, features_out=3, conv=True),           # dpx, dpy, dpz
        )

        # Per-event Cholesky factor heads for full 3x3 covariance of (px, py, pz)
        # Outputs 6 parameters: L11, L21, L22, L31, L32, L33 (lower-triangular)
        self.nu_cholesky_onshell = nn.Sequential(
            GhostBatchNorm1d(2*self.dD + 9, features_out=dH, conv=True),
            NonLUModule(),
            GhostBatchNorm1d(dH, features_out=dH, conv=True),
            NonLUModule(),
            GhostBatchNorm1d(dH, features_out=6, conv=True),
        )
        self.nu_cholesky_offshell = nn.Sequential(
            GhostBatchNorm1d(2*self.dD + 11, features_out=dH, conv=True),
            NonLUModule(),
            GhostBatchNorm1d(dH, features_out=dH, conv=True),
            NonLUModule(),
            GhostBatchNorm1d(dH, features_out=6, conv=True),
        )

        # Post-hoc logistic regression for on-shell vs off-shell selection at inference.
        # Input: (p_onshell, sigma_pz_on, sigma_pz_off) → logit.
        # Fitted on validation data after training completes; replaces hard cuts.
        self.selector_gate = nn.Linear(3, 1)
        # Initialize to approximate the old hard cuts as a starting point
        with torch.no_grad():
            self.selector_gate.weight.copy_(torch.tensor([[2.0, 0.5, -0.5]]))
            self.selector_gate.bias.copy_(torch.tensor([-1.0]))

        self.forwardCalls = 0

    def embedding_layers(self):
        return sorted(self.layers.layers)

    def updateMeanStd(self,  b, nb, l, nu, a):
        self.inputEmbed.updateMeanStd( b, nb, l, nu, a)

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
        self.jet_dR_embed.setGhostBatches(nGhostBatches)
        # Output heads: iterate GBN layers in sequential modules and classifier
        self.onshell_classifier.input_embed.setGhostBatches(nGhostBatches)
        for module in (self.nu_regressor_onshell, self.pz_selector,
                       self.nu_regressor_offshell,
                       self.nu_cholesky_onshell, self.nu_cholesky_offshell,
                       self.onshell_classifier.block1, self.onshell_classifier.block2,
                       self.onshell_classifier.classifier):
            for layer in module:
                if hasattr(layer, "setGhostBatches"):
                    layer.setGhostBatches(nGhostBatches)
        self.nGhostBatches = nGhostBatches


    def forward(self, b, nb, l, nu, a):
        self.forwardCalls += 1
        # Save raw inputs before embedding overwrites them
        raw_met = nu.clone()  # (n, 2): [pt, phi]
        raw_lep = l.clone()   # (n, 6): [pt, eta, phi, mass, isE, isM]
        raw_nb  = nb.clone()  # (n, 4*nj): non-b jets for deltaR computation
        (b, bb, qq, a, nb , l, nu, lnu_mT, bWhad, bWlep, lepQQdR, bbMdR, qqMdR, bbnMdR, bbqqMdR,
        bWhadMdR, bWlepMdR, mask_bbMdR, mask_qqMdR, mask_bbn, mask_qq, mask_bWhad, mask_bWlep,
        derived, soln, kinematic_solutions) = self.inputEmbed(
            b, nb, l, nu, a
        )
        n = b.shape[0]

        b0 = b.clone()
        bb0 = bb.clone()
        nb0 = nb.clone()
        qq0 = qq.clone()
        l0 = l.clone()
        bWhad0 = bWhad.clone()
        bWlep0 = bWlep.clone()

        b = NonLU(b)
        bb = NonLU(bb)
        nb = NonLU(nb)
        qq = NonLU(qq)
        l = NonLU(l)
        nu = NonLU(nu)
        lnu_mT = NonLU(lnu_mT)
        bWhad = NonLU(bWhad)
        bWlep = NonLU(bWlep)

        # bb: H -> bb candidates, qq: W -> qq candidates
        bb, bb0 = self.bbDiJetResNetBlock(b, bb, b0, bb0, debug=self.debug)
        qq, qq0 = self.nonbDiJetResNetBlock(nb, qq, nb0, qq0, debug=self.debug)

        # Create unified leptonic W candidate
        lep_W = l + nu
        lep_W0 = lep_W.clone()
        lep_W = NonLU(lep_W)

        bWhad, bWhad0 = self.bWhadResNetBlock(
            qq.repeat_interleave(2, dim=2),
            bWhad,
            qq0.repeat_interleave(2, dim=2),
            bWhad0,
            debug=self.debug)
        bWlep, bWlep0 = self.bWlepResNetBlock(l, bWlep, l0, bWlep0, debug=self.debug)

        bbMdR = NonLU(bbMdR)
        qqMdR = NonLU(qqMdR)
        bbnMdR = NonLU(bbnMdR)
        bbqqMdR = NonLU(bbqqMdR)
        scalars = torch.cat([lepQQdR, lnu_mT], dim=-1).squeeze(1)

        bWhad_exp = bWhadMdR.reshape(n, -1, 6).repeat_interleave(4, dim=2)
        bWlep_exp = bWlepMdR.squeeze(-1).repeat(1, 1, 6)

        bbn_flat = bbnMdR.squeeze(2)
        bbn_w0 = torch.cat([bbn_flat[:, :, 0:1], bbn_flat[:, :, 1:2]], dim=1)
        bbn_w1 = torch.cat([bbn_flat[:, :, 0:1], bbn_flat[:, :, 2:3]], dim=1)
        bbn_w2 = torch.cat([bbn_flat[:, :, 1:2], bbn_flat[:, :, 2:3]], dim=1)

        bbn_exp = torch.cat([bbn_w0, bbn_w1, bbn_w2], dim=2)
        bbn_exp = bbn_exp.repeat_interleave(4, dim=2).repeat(1, 1, 2)
        bbqq_exp = bbqqMdR.squeeze(2)
        bbqq_exp = bbqq_exp.repeat_interleave(4, dim=2).repeat(1, 1, 2)

        qv_tt = torch.cat([bWhad_exp, bWlep_exp, bbn_exp, bbqq_exp], dim=1)
        qv_tt = self.qv_embed(qv_tt)

        mask_tt = torch.zeros(n, 6, 4, dtype=torch.bool, device=self.device)
        mask_tt[:, 0:3, 2:4] = True
        mask_tt[:, 3:6, 0:2] = True

        # TTbar pairing selection
        TT, TT0, TT_weights = self.attention_tt(
            bWhad, bWlep, mask_tt, bWhad0, qv_tt, scalars, debug=self.debug
        )
        TT_logits = self.select_tt(TT)  # Shape: (n, 6, 1)
        TT_logits = TT_logits.view(n, 6)  # Shape: (n, 6)
        TT_score = F.softmax(TT_logits, dim=-1)  # Shape: (n, 6)
        TT_context = torch.matmul(TT, TT_score.unsqueeze(-1))

        # Individual jet attention: leptonic W queries individual non-b jets
        nb_jets = nb[:, :, :3]          # (n, dD, 3) original jets (drop augmented permutations)
        jet_mask = mask_bbn.view(n, 3)  # (n, 3) per-jet padding mask

        # Compute deltaR between lepton and individual jets from raw kinematics
        nb_raw = raw_nb.view(n, 4, -1)[:, :, :3]  # (n, 4, 3) original raw jets
        lep_raw = raw_lep.view(n, 6, 1)
        lepNBdR = calcDeltaR(lep_raw, nb_raw)      # (n, 1, 3)
        jet_dR = self.jet_dR_embed(lepNBdR, jet_mask)  # (n, dD, 3) embedded deltaR

        WW, WW0, WW_weights = self.attention_WW(
            lep_W,                     # q:  (n, dD, 1) single leptonic W query
            nb_jets,                   # v:  (n, dD, 3) individual jets
            jet_mask.unsqueeze(1),     # mask: (n, 1, 3)
            lep_W0,                    # q0: (n, dD, 1) residual
            jet_dR,                    # qv: (n, dD, 3) deltaR attention bias
            scalars,
            self.debug
        )
        # WW is (n, dD, 1) - enriched leptonic W after attending to jets
        WW_sel = WW

        # Per-jet attention weights (attached for gradient flow)
        # Concatenate heads to preserve per-head information: (n, h, 1, 3) -> (n, h*3)
        jet_weights = WW_weights.squeeze(2).reshape(n, -1)  # (n, h*3=6)
        self._jet_weights = jet_weights.detach()  # for monitoring

        # build context from all other objects in the event
        leptonic_query = lep_W + soln + derived  # all (n, dD, 1)
        full_context = leptonic_query + WW_sel + TT_context + bb  # (n, dD, 1)

        # Initial estimate: px, py from MET; pz from kinematic solutions
        met_pt = raw_met[:, 0:1]   # (n, 1)
        met_phi = raw_met[:, 1:2]  # (n, 1)
        init_px = met_pt * torch.cos(met_phi)  # (n, 1)
        init_py = met_pt * torch.sin(met_phi)  # (n, 1)

        # Raw leptonic transverse mass from MET and lepton (before embedding)
        lep_pt_raw = raw_lep[:, 0:1]   # (n, 1)
        dphi_lnu = raw_lep[:, 2:3] - raw_met[:, 1:2]  # lep phi - MET phi
        lnu_mT_raw = torch.sqrt(
            (2 * lep_pt_raw * met_pt * (1 - torch.cos(dphi_lnu))).clamp(min=1e-6)
        )  # (n, 1)

        # Shared enriched input: full_context + lep_W0 + init_px + init_py + lnu_mT + jet_weights
        # Used by on-shell px/py regressor and on-shell Cholesky (2*dD + 9)
        jet_weights_feat = jet_weights.unsqueeze(-1)  # (n, 6, 1)
        lnu_mT_feat = lnu_mT_raw.unsqueeze(1)         # (n, 1, 1)
        onshell_input = torch.cat([
            full_context, lep_W0,
            init_px.unsqueeze(1), init_py.unsqueeze(1),
            lnu_mT_feat, jet_weights_feat,
        ], dim=1)  # (n, 2*dD+9, 1)

        # Classifier gets additional discriminant features:
        # oss_80, oss_40 (compressed off-shell scores) + W mass discriminant (continuous)
        # log1p compresses the bimodal distribution (0 for real solutions, huge for complex).
        # Add 1e-3 floor so all-on-shell ghost batches still have nonzero variance in GBN.
        oss_80 = torch.log1p(kinematic_solutions[:, 2:3, :] + 1e-3)  # (n, 1, 1)
        oss_40 = torch.log1p(kinematic_solutions[:, 5:6, :] + 1e-3)  # (n, 1, 1)

        disc_feat = _w_mass_discriminant(lep_pt_raw, raw_lep[:, 1:2], raw_lep[:, 2:3], raw_lep[:, 3:4], init_px, init_py)

        classifier_input = torch.cat([
            onshell_input, oss_80, oss_40, disc_feat,
        ], dim=1)  # (n, 2*dD+12, 1)

        # Classify on-shell vs off-shell
        logit_onshell = self.onshell_classifier(classifier_input)

        # initial pz: average of 80/40 GeV W mass constraint solutions for on/off shell
        init_pz_on = 0.5 * (kinematic_solutions[:, 0, :] + kinematic_solutions[:, 1, :])  # (n, 1)
        init_pz_off = 0.5 * (kinematic_solutions[:, 3, :] + kinematic_solutions[:, 4, :])  # (n, 1)

        nu_init_off = torch.cat([init_px, init_py, init_pz_off], dim=1)  # (n, 3)
        delta_on = self.nu_regressor_onshell(onshell_input).squeeze(-1)  # (n, 2): dpx, dpy
        nu_px_on = init_px.squeeze(1) + delta_on[:, 0]
        nu_py_on = init_py.squeeze(1) + delta_on[:, 1]

        # Solve W mass quadratic with corrected (px, py), mW = 80.379 GeV
        pz_sol1, pz_sol2, _, oss_corrected = get_nu_pz_cartesian(
            raw_lep[:, 0], raw_lep[:, 1], raw_lep[:, 2], raw_lep[:, 3],
            nu_px_on, nu_py_on, mW=80.379,
        )

        # Lepton-neutrino rapidity for both pz solutions (from corrected MET)
        lep_eta = raw_lep[:, 1]  # (n,)
        nu_E_sol1 = torch.sqrt(nu_px_on**2 + nu_py_on**2 + pz_sol1**2 + 1e-8)
        nu_E_sol2 = torch.sqrt(nu_px_on**2 + nu_py_on**2 + pz_sol2**2 + 1e-8)
        eta_nu_sol1 = torch.atanh((pz_sol1 / nu_E_sol1).clamp(-1 + 1e-6, 1 - 1e-6))
        eta_nu_sol2 = torch.atanh((pz_sol2 / nu_E_sol2).clamp(-1 + 1e-6, 1 - 1e-6))
        deta_sol1 = eta_nu_sol1 - lep_eta
        deta_sol2 = eta_nu_sol2 - lep_eta

        # pz selector: onshell context + rapidity features + corrected off-shell score
        selector_input = torch.cat([
            onshell_input,
            deta_sol1.unsqueeze(-1).unsqueeze(-1),              # (n, 1, 1)
            deta_sol2.unsqueeze(-1).unsqueeze(-1),              # (n, 1, 1)
            eta_nu_sol1.unsqueeze(-1).unsqueeze(-1),             # (n, 1, 1)
            eta_nu_sol2.unsqueeze(-1).unsqueeze(-1),             # (n, 1, 1)
            torch.log1p(oss_corrected + 1e-3).unsqueeze(-1).unsqueeze(-1),  # (n, 1, 1)
        ], dim=1)  # (n, 2*dD+14, 1)
        logit_sol = self.pz_selector(selector_input).squeeze(-1).squeeze(-1)  # (n,)

        # Binary select: sigmoid(logit_sol) > 0.5 → use sol1, else sol2 for analytic nu_pz
        use_sol1 = logit_sol > 0.0  # equivalent to sigmoid > 0.5
        nu_pz_on = torch.where(use_sol1, pz_sol1, pz_sol2)

        nu_pred_on = torch.stack([nu_px_on, nu_py_on, nu_pz_on], dim=1)  # (n, 3)
        logit_sol_on = logit_sol

        # --- Off-shell neutrino: regress all 3 components ---
        pz_solutions = kinematic_solutions[:, [0, 1, 3, 4], :]  # (n, 4, 1): pz1_80, pz2_80, pz1_40, pz2_40
        offshell_input = torch.cat(
            [full_context, lep_W0, pz_solutions, lnu_mT_feat, jet_weights_feat], dim=1
            )  # (n, 2*dD+11, 1)
        delta_off = self.nu_regressor_offshell(offshell_input).squeeze(-1)  # (n, 3)
        nu_pred_off = nu_init_off + delta_off

        L_on = _build_cholesky(self.nu_cholesky_onshell(onshell_input).squeeze(-1))
        L_off = _build_cholesky(self.nu_cholesky_offshell(offshell_input).squeeze(-1))

        return nu_pred_on, L_on, nu_pred_off, L_off, (logit_onshell, logit_sol_on)

    def setStore(self, store):
        self.store = store
        self.inputEmbed.store = store
        self.inputEmbed.storeData = self.storeData

    def writeStore(self):
        # print(self.storeData)
        print(self.store)
        np.save(self.store, self.storeData)