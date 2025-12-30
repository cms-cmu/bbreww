from .bbWW_models import *

class InputEmbed(nn.Module):
    def __init__(
        self,
        dijetFeatures,
        ancillaryFeatures=["HT", "njets", "nsoftjets"],
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
            name="lepton convolution",
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
            3,
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
            4,
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

        self.register_buffer('mask_qq_same', torch.zeros((1, self.wsl, self.wsl), dtype=torch.bool))
        for i in range(self.wsl):
            self.mask_qq_same[:, i, i] = 1  # mask diagonal

        self.register_buffer('mask_bW_same', torch.zeros((1, self.bsl, self.wsl), dtype=torch.bool))
        for i in range(self.wsl):
            self.mask_bW_same[:, i, i] = 1  # mask diagonal

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
        )  # phi is removed
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


    def dataPrep(self, b, nb, l, nu, a):  # , device='cuda'):
        device = b.get_device() if b.get_device() >= 0 else "cpu"
        # # if device=='cpu': # prevent overwritting data from dataloader when doing operations directly from RAM rather than copying to VRAM
        # j = j.clone()
        # o = o.clone()
        # a = a.clone()

        n = b.shape[0]
        b = b.view(n, 4, 2)
        nb = nb.view(n, 4, -1)
        l = l.view(n, 6, 1)
        nu = nu.view(n, 2, 1)
        a = a.view(n, self.dA, 1)

        a[:, 1, :] = torch.log(
            a[:, 1, :] - 3
        )  # TODO: find index based on the feature name, check if relevant

        ## bb: H->bb dijet candidates, qq: W->qq dijet candidates"
        bb, bbPxPyPzE = addFourVectors(
            b[:, :, (0)], b[:, :, (1)]
        )

        qq, qqPxPyPzE = addFourVectors(
            nb[:, :, (0, 0, 1)], nb[:, :, (1, 2, 2)]
        )

        bb = bb.unsqueeze(2) # add a dimension to calculating MdR matrix symmetrically later
        bbPxPyPzE = bbPxPyPzE.unsqueeze(2)
        qq = qq.unsqueeze(2)
        qqPxPyPzE = qqPxPyPzE.unsqueeze(2)

        ## top reconstruction
        bWhad, bWhadPxPyPzE = addFourVectors(
            b[:, :, (0, 1)], qq # hadronic top candidate
        )
        bWlep, bWlepPxPyPzE = addFourVectors(
            b[:, :, (0, 1)], l[:, :, (0, 0)] # leptonic top candidate (only add b + l because MET is not a four vector)
        )

        mask, bbMdR, qqMdR, bbnMdR, mask_bbMdR, mask_qqMdR, mask_bbn = None, None, None, None, None, None, None
        b = torch.cat(
            [b, 2 * torch.ones((n, 1, 2), dtype=torch.float, device=device)], 1
        )  # label bJets with 2 (-1 for mask, 0 for not preselected, 1 for preselected jet)
        nb = torch.cat(
            [nb, torch.ones((n, 1, 3), dtype=torch.float, device=device)], 1
        ) 
        mask = (nb[:, 2, :] == -1)
        bPxPyPzE = PxPyPzE(b)
        nbPxPyPzE = PxPyPzE(nb)
        lPxPyPzE = PxPyPzE(l)

        # For b-jets: compute matrix of dijet masses and opening angles between other jets
        n = bb.shape[0]
        bbMdR = matrixMdR(b, b, v1PxPyPzE=bPxPyPzE, v2PxPyPzE=bPxPyPzE)
        bbMdR = torch.cat(
            [
                bbMdR,
                torch.zeros((n, 2, self.bsl, self.bsl), dtype=torch.float, device=device)
            ],
            1,
        )  # flag with zeros to signify dijet quantities

        mask_bbMdR = mask.view(n, 1, self.bsl) | mask.view(
            n, self.bsl, 1
        )  # mask of 2d matrix of b-jets (i,j) is True if mask[i] | mask[j]
        mask_bbMdR = mask_bbMdR.masked_fill(self.mask_bb_same, 1)

        # compute matrix of trijet masses and opening angles between b-dijets and non-bjets
        bbnMdR = matrixMdR(bb, nb, v1PxPyPzE=bbPxPyPzE, v2PxPyPzE=nbPxPyPzE)
        bbnMdR = torch.cat(
            [
                bbnMdR,
                torch.ones((n, 2, 1, self.wsl), dtype=torch.float, device=device)
            ],
            1,
        )  # flag with ones to signify trijet quantities
        lepQQdR = calcDeltaR(l, qq)
        mask_bbn = mask.view(n, 1, self.bsl)

        # For nonb-jets: compute matrix of dijet masses and opening angles between other jets
        n = qq.shape[0]
        qqMdR = matrixMdR(nb, nb, v1PxPyPzE=nbPxPyPzE, v2PxPyPzE=nbPxPyPzE)
        qqMdR = torch.cat(
            [
                qqMdR,
                torch.zeros((n, 2, self.wsl, self.wsl), dtype=torch.float, device=device)
            ],
            1,
        )  # flag with zeros to signify dijet quantities

        # For lepton and MET, compute transverse mass
        lnu_mT = transverse_mass(l, nu)

        mask_qqMdR = mask.view(n, 1, self.wsl) | mask.view(
            n, self.wsl, 1
        )  # mask of 2d matrix of nonb-jets (i,j) is True if mask[i] | mask[j]
        mask_qqMdR = mask_qqMdR.masked_fill(self.mask_qq_same, 1)

        # compute matrix of masses and opening angles between b-jets and W candidates (top)
        bWhadMdR = matrixMdR(b, qq, v1PxPyPzE=bPxPyPzE, v2PxPyPzE=qqPxPyPzE)
        bWhadMdR = torch.cat(
            [
                bWhadMdR,
                torch.zeros((n, self.bsl, self.wsl, 1), dtype=torch.float, device=device)
            ],
            1,
        )  # flag with zeros to signify calculated quantities (b+W)

        mask_bWhad = mask.view(n, 1, self.bsl) | mask.view(
            n, self.wsl, 1
        )  # mask of 2d matrix of bW (i,j) is True if mask[i] | mask[j]
        mask_bWhad = mask_bWhad.masked_fill(self.mask_bW_same, 1)

        bWlepMdR = matrixMdR(b, l.unsqueeze(2), v1PxPyPzE=bPxPyPzE, v2PxPyPzE=lPxPyPzE) # l needs an extra dimension for concat later
        bWlepMdR = torch.cat(
            [
                bWlepMdR,
                torch.ones((n, 1, self.bsl, 1), dtype=torch.float, device=device)
            ],
            1,
        )  # flag with zeros to signify calculated quantities (b+W)


        mask_bWlep = mask.view(n, 1, self.bsl) | mask.view(
            n, self.bsl, 1
        )  # mask of 2d matrix of bW (i,j) is True if mask[i] | mask[j]
        mask_bWlep = mask_bWlep.masked_fill(self.mask_bW_same, 1) # to do: create self.mask_bW_same above

        nb[:, (0, 3), :] = torch.log(1 + nb[:, (0, 3), :])
        nb[isinf(nb)] = -1  # isinf not supported by ONNX

        b[:, (0, 3), :] = torch.log(1 + b[:, (0, 3), :])
        bb[:, (0, 3), :] = torch.log(1 + bb[:, (0, 3), :])
        qq[:, (0, 3), :] = torch.log(1 + qq[:, (0, 3), :])

        b = torch.cat([b, b[:, :, (1,0)]] , 2) # create permutation invariance by augmenting opposite order of same jets
        nb = torch.cat([nb, nb[:, :, (2,1,0)]] , 2)

        # only keep relative angular information so that learned features are invariant under global phi rotations and eta/phi flips
        b[:, 2:3, :] = calcDeltaPhi(bb, b[:, :, :]) # replace jet phi with deltaPhi between dijet and jet
        bb[:, 2:3, :] = calcDeltaPhi(qq, bb[:, :, :])
        nb[:, 2:3, :] = calcDeltaPhi(qq, nb[:, :, :]) # replace jet phi with deltaPhi between dijet and jet
        qq[:, 2:3, :] = calcDeltaPhi(bb, qq[:, :, :])

        return b, bb, qq, a, nb , l, nu, lnu_mT, bWhad, bWlep, lepQQdR, bbMdR, qqMdR, bbnMdR, bWhadMdR, bWlepMdR, mask, mask_bbMdR, mask_qqMdR, mask_bbn, mask_bWhad, mask_bWlep

    def updateMeanStd(self,  b, nb, l, nu, a):
        (b, bb, qq, a, nb , l, nu, lnu_mT, bWhad, bWlep, lepQQdR, bbMdR, qqMdR, bbnMdR, bWhadMdR, bWlepMdR, 
        mask, mask_bbMdR, mask_qqMdR, mask_bbn, mask_bWhad, mask_bWlep) = self.dataPrep(
                                                                        b, nb, l, nu, a)
                                                                         # , device='cpu')


        n, self.bsl, self.wsl = b.shape[0], 2, 2 #hard code these values if only using 2 b and 2 nonbjets
        MdR = torch.cat(
            (
                bbMdR.view(n, 4, -1),
                qqMdR.view(n, 4, -1),
                bbnMdR.view(n, 4, -1)
            ),
            dim=2,
        )
        mask_MdR = torch.cat(
            (
                mask_bbMdR.view(n, -1),
                mask_qqMdR.view(n, -1),
                mask_bbn.view(n, -1),
            ),
            dim=1,
        )
        
        MdRtt = torch.cat(
            (
                bWhadMdR.view(n, 3, -1),
                bWlepMdR.view(n, 3,- 1),
            ),
            dim=2,
        )

        mask_MdRtt =  mask_bWhad.view(n, -1) # mask is same for had and lep tt
        
        # self. diMdPhi_embed.setMeanStd(ooMdPhi.view(n, 2, self.bsl*self.bsl), mask_oo.view(n, self.bsl*self.bsl))
        # self.triMdPhi_embed.setMeanStd(doMdPhi.view(n, 2, self.wsl*self.bsl), mask_do.view(n, self.wsl*self.bsl))

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

    def forward(self, b, nb, l, nu, a):
        (b, bb, qq, a, nb , l, nu, lnu_mT, bWhad, bWlep, lepQQdR, bbMdR, qqMdR, bbnMdR, bWhadMdR, bWlepMdR, 
        mask, mask_bbMdR, mask_qqMdR, mask_bbn, mask_bWhad, mask_bWlep) = self.dataPrep(b, nb, l, nu, a)

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
        bbMdR = bbMdR.view(n, 4, self.bsl*self.bsl)
        qqMdR = qqMdR.view(n, 4, self.wsl*self.wsl)
        bbnMdR = bbnMdR.view(n, 4, self.wsl)
        mask_bbMdR = mask_bbMdR.view(n, -1)
        mask_qqMdR = mask_qqMdR.view(n, -1)
        mask_bbn = mask_bbn.view(n, -1)
        MdR = torch.cat((bbMdR, qqMdR, bbnMdR), dim=2)
        mask_MdR = torch.cat((mask_bbMdR, mask_qqMdR, mask_bbn), dim=1) # Higgs masses and dijets information
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
        bbnMdR = MdR[:, :, self.bsl * self.bsl + self.wsl * self.wsl :].view(
            n, self.dD, 1, self.wsl
        )

        bWhadMdR = bWhadMdR.view(n, self.bsl*self.wsl + 1, -1)
        bWlepMdR = bWlepMdR.view(n, self.bsl + 1, -1)
        MdRtt = torch.cat((bWhadMdR, bWlepMdR), dim=2)
        mask_MdRtt =  mask_bWhad.view(n, -1) # mask is same for had and lep tt
        MdRtt = self.MdRttEmbed(MdRtt) # nothing to mask for nominal case (2b, 2 nonbjets)
        MdRtt = self.MdRttConv(NonLU(MdRtt))

        bWhadMdR = MdRtt[:, :, :self.bsl* self.wsl].view(
            n, self.dD, self.bsl, self.wsl
        )
        bWlepMdR = MdRtt[:, :, self.bsl:].view(
            n, self.dD, self.bsl, 1
        )

        b = self.bJetEmbed(b)
        bb = self.bbDiJetEmbed(bb)
        b = b + a
        b = self.bJetConv(NonLU(b))
        bb = self.bJetConv(NonLU(bb))

        l = self.lepEmbed(l)
        nu = self.nuEmbed(nu)
        l = self.lepConv(NonLU(l))
        nu = self.nuConv(NonLU(nu))

        # top reconstruction
        bWhad = self.bWhadEmbed(bWhad)
        bWlep = self.bWlepEmbed(bWlep)
        bWhad = self.bWhadConv(NonLU(bWhad))
        bWlep = self.bWlepConv(NonLU(bWlep))


        return b, bb, qq, a, nb , l, nu, lnu_mT, bWhad, bWlep, lepQQdR, bbMdR, qqMdR, bbnMdR, bWhadMdR, bWlepMdR, mask_bbMdR, mask_qqMdR, mask_bbn, mask_bWhad, mask_bWlep
    
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
        self.dD = dijetFeatures  # dimension of embeded   dijet feature space
        self.device = device
        dijetBottleneck = None
        self.name = (
            architecture
            + "_%d" % (dijetFeatures)
        )
        self.nC = nClasses
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
            inputLayers=[self.inputEmbed.lepConv, self.inputEmbed.nuEmbed],
        )
        self.bWhadResNetBlock = ResNetBlock(
            self.dD,
            prefix="leptonic W",
            nLayers=2,
            phase_symmetric=self.phase_symmetric,
            device=self.device,
            layers=self.layers,
            inputLayers=[self.inputEmbed.bWhadConv, self.inputEmbed.bJetConv, self.inputEmbed.nonbDiJetConv],
        )
        self.bWlepResNetBlock = ResNetBlock(
            self.dD,
            prefix="leptonic W",
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
            scalar_dim = 2,
            layers=self.layers,
            inputLayers=[self.lepWResNetBlock.conv[-1], self.nonbDiJetResNetBlock.reinforce[-1]],
            device=self.device,
        )
        self.layers.addLayer(self.attention_WW, self.attention_WW.inputLayers)

        self.attention_hh = MinimalAttention(
            self.dD,
            heads=2,
            phase_symmetric=self.phase_symmetric,
            scalar_dim = 2,
            layers=self.layers,
            inputLayers=[self.inputEmbed.bJetConv, self.attention_WW],
            device=self.device,
        )
        self.layers.addLayer(self.attention_hh, self.attention_hh.inputLayers)

        self.attention_tt = MinimalAttention(
            self.dD,
            heads=2,
            phase_symmetric=self.phase_symmetric,
            layers=self.layers,
            scalar_dim = 2,
            inputLayers=[self.bWhadResNetBlock.conv[-1], self.bWlepResNetBlock.conv[-1]],
            device=self.device,
        )

        self.scalars_embed = GhostBatchNorm1d(
            1, 
            features_out=self.dD,
            conv=True, 
            name="scalar physics relationships embed"
        )

        self.qv_embed = GhostBatchNorm1d(
            self.dD*3,  # Input: full feature dim (24)
            features_out=8,  # Output: heads * head_dim = 2 * 4
            conv=True,
            name="qv physics relationships projector"
        )

        self.layers.addLayer(self.attention_tt, self.attention_tt.inputLayers)

        # Embed enhanced HH representation  
        self.HH_final_embed = GhostBatchNorm1d(
            self.dD,
            features_out=self.dD, 
            conv=True,
            name="HH final embed"
        )

        self.TT_final_embed = GhostBatchNorm1d(
            self.dD,
            features_out=self.dD, 
            conv=True,
            name="TT final embed"
        )

        self.layers.addLayer(self.WW_final_embed, [self.attention_WW])
        self.layers.addLayer(self.HH_final_embed, [self.attention_hh])
        self.layers.addLayer(self.TT_final_embed, [self.attention_tt])

        self.final_combine = GhostBatchNorm1d(
            self.dD *2,  # Input from concatenated WW + HH 
            features_out=self.nC, 
            conv=True, 
            name="combine WW and HH and TT"
        )
        # self.layers.addLayer(self.dijetEmbedInQuadjetSpace, [previousLayer])
        self.layers.addLayer(self.final_combine, [self.WW_final_embed, self.HH_final_embed])

        self.select_tt = GhostBatchNorm1d(
            self.dD, 
            features_out=1,  # Single score per candidate
            conv=True, 
            bias=False,  # No bias because softmax is translation invariant
            name="TT pairing selector"
        )

        self.select_WW = GhostBatchNorm1d(
            self.dD, 
            features_out=1,  # Single score per candidate
            conv=True, 
            bias=False,  # No bias because softmax is translation invariant
            name="TT pairing selector"
        )

        self.out_tt = GhostBatchNorm1d(
            self.dD, 
            features_out=self.nC,  # final tt bar score
            conv=True, 
            bias=True,
            name="TT bar score"
        )

        self.final_linear_layer = linear(in_channels=16, out_channels=self.nC)
        self.layers.addLayer(self.final_linear_layer)

        self.out = nn.Sequential(
            GhostBatchNorm1d(
                self.dD, 
                features_out=16, 
                conv=True, 
                bias=False,
                name="final event score"
            ),
            NonLUModule(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            self.final_linear_layer
        )

        self.layers.addLayer(self.select_tt, [self.attention_tt])
        self.layers.addLayer(self.select_WW, [self.attention_WW])
        self.layers.addLayer(self.out_tt, [self.select_tt])  
        self.forwardCalls = 0

    def embedding_layers(self):
        return sorted(set(self.layers.layers).difference(self.output_layers()))

    def output_layers(self):
        return [self.out.index]

    def updateMeanStd(self,  b, nb, l, nu, a):
        self.inputEmbed.updateMeanStd( b, nb, l, nu, a)

    def initMeanStd(self):
        self.inputEmbed.initMeanStd()

    def setGhostBatches(self, nGhostBatches, subset=False):
        self.inputEmbed.setGhostBatches(nGhostBatches)
        self.WW_final_embed.setGhostBatches(nGhostBatches)
        self.HH_final_embed.setGhostBatches(nGhostBatches)
        self.final_combine.setGhostBatches(nGhostBatches)
        self.out.setGhostBatches(nGhostBatches)
        self.nGhostBatches = nGhostBatches

    def forward(self, b, nb, l, nu, a):
        self.forwardCalls += 1
        # print('\n-------------------------------\n')
        (b, bb, qq, a, nb , l, nu, lnu_mT, bWhad, bWlep, lepQQdR, bbMdR, qqMdR, bbnMdR, 
        bWhadMdR, bWlepMdR, mask_bbMdR, mask_qqMdR, mask_bbn, mask_bWhad, mask_bWlep)  = self.inputEmbed(
            b, nb, l, nu, a
        )  # format inputs to array of objects and apply scalers and GBNs
        # print('o after inputEmbed\n',o[0])
        n = b.shape[0]
        #
        # Build up dijet pixels with jet pixels and initial dijet pixels
        #

        # Embed the jet 4-vectors and dijet ancillary features into the target feature space
        b0 = b.clone()
        bb0 = bb.clone()
        nb0 = nb.clone()
        qq0 = qq.clone()
        l0 = l.clone()
        nu0 = nu.clone()
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

        # Create unified W candidate with all information available
        lep_W = l + nu  # can add them because inputs have been embedded
        lep_W0 = lep_W.clone()
        lep_W = NonLU(lep_W)

        bWhad, bWhad0 = self.bWhadResNetBlock(qq, bWhad, qq0, bWhad0, debug=self.debug)
        bWlep, bWlep0 = self.bWlepResNetBlock(l, bWlep, l0, bWlep0, debug=self.debug)

        bbMdR = NonLU(bbMdR)
        qqMdR = NonLU(qqMdR)
        bbnMdR = NonLU(bbnMdR)
        scalars = torch.cat([lepQQdR, lnu_mT], dim= -1)

        qv_tt = torch.stack([
            # Pair (had0, lep0): b[0]+qq with b[0]+l
            torch.cat([bWhadMdR[:, :, 0, 0], bWlepMdR[:, :, 0, 0], bbnMdR[:, :, 0, 0]], dim=-1),
            # Pair (had0, lep1): b[0]+qq with b[1]+l  
            torch.cat([bWhadMdR[:, :, 0, 0], bWlepMdR[:, :, 1, 0], bbnMdR[:, :, 0, 1]], dim=-1),
            # Pair (had1, lep0): b[1]+qq with b[0]+l
            torch.cat([bWhadMdR[:, :, 1, 0], bWlepMdR[:, :, 0, 0], bbnMdR[:, :, 0, 0]], dim=-1),
            # Pair (had1, lep1): b[1]+qq with b[1]+l
            torch.cat([bWhadMdR[:, :, 1, 0], bWlepMdR[:, :, 1, 0], bbnMdR[:, :, 0, 1]], dim=-1),
        ], dim=-1)
        qv_tt = self.qv_embed(qv_tt)

        TT, TT0, TT_weights = self.attention_tt(
            bWhad,    # queries: hadronic top candidate
            bWlep,    # values: leptonic top candidate
            None,     # mask: None
            bWhad0,   # residual for hadronic top
            qv_tt,       # physics relationships (delta R and mass between b-jets and nonb-jets)
            scalars.squeeze(1),  # scalar physics relationships (dR (lep, qq) and transverse_mass(lep, nu))
            debug=self.debug
        )

        # TTbar pairing selection
        TT_logits = self.select_tt(TT)  # Shape: (n, 2, 1)
        TT_logits = TT_logits.view(n, 2)  # Shape: (n, 2)
        TT_score = F.softmax(TT_logits, dim=-1)  # Shape: (n, 2)

        TT_sel = torch.matmul(TT, TT_score.unsqueeze(-1))
        TT_final = self.out_tt(TT_sel)  # Shape: (n, nC)
        #TT_final = TT_logits.squeeze(-1)
        self._last_tt_logits = TT_logits.detach() # save TTbar candidates scores

        scalars = self.scalars_embed(scalars)
        qv = torch.cat([
                bbnMdR,
                qqMdR
            ], dim=-1)
        
        WW, WW0, WW_weights = self.attention_WW(
            lep_W,        # queries: leptonic W candidate
            qq,           # values: hadronic W candidate (non-bjet dijets) 
            mask_qqMdR,         # mask: None as we have exactly one lep_W and qq candidate right now
            lep_W0,       # residual for leptonic W
            qv,
            scalars,       # scalar physics relationships (dR (lep, qq) and transverse_mass(lep, nu))
            self.debug
        )

        WW_logits = self.select_WW(WW)  # Shape: (n, 3, 1)
        WW_logits = F.softmax(WW_logits.view(n, 3), dim=-1)
        self._WW_logits = WW_logits.detach()
        WW = torch.matmul(WW, WW_logits.unsqueeze(-1))

        HH = torch.cat([
            bb,                                  # (n, dD, 1)
            WW,                                  # (n, dD, 1)
            bbMdR[:, :, 0, 1:2].squeeze(2),      # (n, dD, 1) - squeeze out dimension 2
            bbnMdR.squeeze(2),                   # (n, dD, wsl) - squeeze out dimension 2
            qqMdR.view(n, self.dD, -1),          # (n, dD, wsl*wsl) - flatten last two dims
            scalars
        ], dim=-1) # Result shape: (n, features, 4)
        HH_final = self.HH_final_embed(HH)

        HH_logits = torch.cat([HH_final, TT_sel], dim=-1) # combine HH and H-> WW scores
        HH_logits = self.out(HH_logits)

        return HH_logits, TT_final, WW

    def setStore(self, store):
        self.store = store
        self.inputEmbed.store = store
        self.inputEmbed.storeData = self.storeData

    def writeStore(self):
        # print(self.storeData)
        print(self.store)
        np.save(self.store, self.storeData)