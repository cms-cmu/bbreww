from .bbWW_models import *

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
        b = b.view(n, 5, 2)
        nb = nb.view(n, 4, -1)
        l = l.view(n, 6, 1)
        nu = nu.view(n, 2, 1)
        a = a.view(n, self.dA, 1)

        a[:, 2, :] = torch.log(a[:, 2, :])  # log transform event HT

        #reconstruct leptonic W by solving MET pz with W mass constraint
        W_lep1, W_lep2, off_shell_score = get_lepW(l[:, :4], nu)
        W_lep = torch.cat([W_lep1, W_lep2], dim=2)
        
        #a = torch.cat([a, off_shell_score.view(n, 1, 1)], dim=1)

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

        return b, bb, qq, a, nb , l, nu, lnu_mT, bWhad, bWlep, lepQQdR, bbMdR, qqMdR, bbnMdR, bbqqMdR, bWhadMdR, bWlepMdR, mask, mask_bbMdR, mask_qqMdR, mask_bbn, mask_qq, mask_bWhad, mask_bWlep

    def updateMeanStd(self,  b, nb, l, nu, a):
        (b, bb, qq, a, nb , l, nu, lnu_mT, bWhad, bWlep, lepQQdR, bbMdR, qqMdR, bbnMdR, bbqqMdR, bWhadMdR, bWlepMdR, 
        mask, mask_bbMdR, mask_qqMdR, mask_bbn, mask_qq, mask_bWhad, mask_bWlep) = self.dataPrep(
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
        (b, bb, qq, a, nb , l, nu, lnu_mT, bWhad, bWlep, lepQQdR, bbMdR, qqMdR, bbnMdR, bbqqMdR, bWhadMdR, bWlepMdR, 
        mask, mask_bbMdR, mask_qqMdR, mask_bbn, mask_qq, mask_bWhad, mask_bWlep) = self.dataPrep(b, nb, l, nu, a)

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

        return b, bb, qq, a, nb , l, nu, lnu_mT, bWhad, bWlep, lepQQdR, bbMdR, qqMdR, bbnMdR, bbqqMdR, bWhadMdR, bWlepMdR, mask_bbMdR, mask_qqMdR, mask_bbn, mask_qq, mask_bWhad, mask_bWlep
    
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
            inputLayers=[self.lepWResNetBlock.conv[-1], self.nonbDiJetResNetBlock.reinforce[-1]],
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

        self.select_WW = GhostBatchNorm1d(
            self.dD,
            features_out=1,  # Single score per candidate
            conv=True,
            bias=False,  # No bias because softmax is translation invariant
            name="non-bjet pairing selector"
        )
        self.layers.addLayer(self.select_WW, [self.attention_WW])

        self.none_WW_score = GhostBatchNorm1d(
            self.dD,
            features_out=1,
            conv=True,
            name="WW rejection scorer"
        )
        self.layers.addLayer(self.none_WW_score, [self.attention_WW])

        self.out_tt = GhostBatchNorm1d(
            self.dD,
            features_out=self.nC,  # final tt bar score
            conv=True,
            bias=True,
            name="TT bar score"
        )
        self.layers.addLayer(self.out_tt, [self.select_tt]) 

        self.final_linear_layer = linear(in_channels=16, out_channels=self.nC)
        self.layers.addLayer(self.final_linear_layer)

        self.HH_final_embed = GhostBatchNorm1d(
            self.dD,
            features_out=self.dD, 
            conv=True,
            name="HH final embed"
        )
        self.layers.addLayer(self.HH_final_embed, [self.inputEmbed.bJetConv, self.select_WW])

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
        self.forwardCalls = 0

    def embedding_layers(self):
        return sorted(set(self.layers.layers).difference(self.output_layers()))

    def output_layers(self):
        return [self.final_linear_layer.index]

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
        self.select_WW.setGhostBatches(nGhostBatches)
        self.none_WW_score.setGhostBatches(nGhostBatches)
        self.out_tt.setGhostBatches(nGhostBatches)
        self.HH_final_embed.setGhostBatches(nGhostBatches)
        self.out[0].setGhostBatches(nGhostBatches)
        self.nGhostBatches = nGhostBatches


    def forward(self, b, nb, l, nu, a):
        self.forwardCalls += 1
        # print('\n-------------------------------\n')
        (b, bb, qq, a, nb , l, nu, lnu_mT, bWhad, bWlep, lepQQdR, bbMdR, qqMdR, bbnMdR, bbqqMdR, 
        bWhadMdR, bWlepMdR, mask_bbMdR, mask_qqMdR, mask_bbn, mask_qq, mask_bWhad, mask_bWlep)  = self.inputEmbed(
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
        scalars = torch.cat([lepQQdR, lnu_mT], dim= -1).squeeze(1) # remove middle dimension for attention mechanism broadcasting

        # create 6x4 features for attention mechanism
        bWhad_exp = bWhadMdR.reshape(n, -1, 6).repeat_interleave(4, dim=2)  # (n, d, 24)
        bWlep_exp = bWlepMdR.squeeze(-1).repeat(1, 1, 6)  # (n, d, 24)

        # there are two non-bjets for each bb-dijet, so take average of two
        bbn_flat = bbnMdR.squeeze(2)  # (n, d, 3)
        bbn_w0 = torch.cat([bbn_flat[:, :, 0:1], bbn_flat[:, :, 1:2]], dim=1)  # (n, 2d, 1) - W0 uses nb0+nb1
        bbn_w1 = torch.cat([bbn_flat[:, :, 0:1], bbn_flat[:, :, 2:3]], dim=1)  # (n, 2d, 1) - W1 uses nb0+nb2
        bbn_w2 = torch.cat([bbn_flat[:, :, 1:2], bbn_flat[:, :, 2:3]], dim=1)  # (n, 2d, 1) - W2 uses nb1+nb2

        bbn_exp = torch.cat([bbn_w0, bbn_w1, bbn_w2], dim=2)  # (n, 2d, 3)
        bbn_exp = bbn_exp.repeat_interleave(4, dim=2).repeat(1, 1, 2)  # (n, 2d, 24)
        bbqq_exp = bbqqMdR.squeeze(2)  # (n, d, 3) - one per W candidate
        bbqq_exp = bbqq_exp.repeat_interleave(4, dim=2).repeat(1, 1, 2)  # (n, d, 24)

        
        # Concatenate all relationship features
        qv_tt = torch.cat([bWhad_exp, bWlep_exp, bbn_exp, bbqq_exp], dim=1)  # (n, 3*d, 12)
        qv_tt = self.qv_embed(qv_tt)

        # block invalid pairings (same b-jet in both tops) with a mask
        mask_tt = torch.zeros(n, 6, 4, dtype=torch.bool, device=self.device)
        mask_tt[:, 0:3, 2:4] = True  # b0 hadronic × b0 leptonic (invalid)
        mask_tt[:, 3:6, 0:2] = True  # b1 hadronic × b1 leptonic (invalid)

        TT, TT0, TT_weights = self.attention_tt(
            bWhad,    # queries: hadronic top candidate
            bWlep,    # values: leptonic top candidate
            mask_tt,  # masks out invalid pairings with the same b-jet
            bWhad0,   # residual for hadronic top
            qv_tt,    # physics relationships (delta R and mass between b-jets and nonb-jets)
            scalars,  # scalar physics relationships (dR (lep, qq) and transverse_mass(lep, nu))
            debug=self.debug
        )

        # TTbar pairing selection
        TT_logits = self.select_tt(TT)  # Shape: (n, 6, 1)
        TT_logits = TT_logits.view(n, 6)  # Shape: (n, 6)
        TT_score = F.softmax(TT_logits, dim=-1)  # Shape: (n, 6)

        TT_sel = torch.matmul(TT, TT_score.unsqueeze(-1))
        TT_final = self.out_tt(TT_sel)  # Shape: (n, nC)
        #TT_final = TT_logits.squeeze(-1)
        self._last_tt_logits = TT_logits.detach() # save TTbar candidates scores

        WW, WW0, WW_weights = self.attention_WW(
            lep_W.expand(-1, -1, 3),    # queries: leptonic W candidate
            qq,           # values: hadronic W candidate (non-bjet dijets) 
            mask_qq.unsqueeze(1).expand(-1, 3, -1),  # mask invalid dijets for all queries
            lep_W0.expand(-1, -1, 3), # residual for leptonic W
            qqMdR,
            scalars,       # scalar physics relationships (dR (lep, qq) and transverse_mass(lep, nu))
            self.debug
        )
        WW_logits = self.select_WW(WW)  # Shape: (n, 3, 1)
        WW_logits = F.softmax(WW_logits.view(n, 3), dim=-1)
        self._WW_logits = WW_logits.detach()
        WW = torch.matmul(WW, WW_logits.unsqueeze(-1))

        scalars = self.scalars_embed(scalars.unsqueeze(-1)) # match second dimensions before concatenating
        HH = torch.cat([
            bb,                           # (n, dD, 1)
            WW,                           # (n, dD, 1)
            bbMdR[:, :, 0, 1:2],          # (n, dD, 1) 
            bbnMdR.squeeze(2),            # (n, dD, wsl) - squeeze out dimension 2
            qqMdR.view(n, self.dD, -1),   # (n, dD, wsl*wsl) - flatten last two dims
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
