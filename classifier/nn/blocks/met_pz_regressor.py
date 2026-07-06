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

def calc_mW(lep, nu):
    ''' function takes lepton in polar and nu in cartesian coordinates'''
    ''' returns leptonic W mass'''
    lep_px = lep[:, 0] * torch.cos(lep[:, 2])
    lep_py = lep[:, 0] * torch.sin(lep[:, 2])
    lep_pz = lep[:, 0] * torch.sinh(lep[:, 1])
    lep_E = torch.sqrt(lep_px**2 + lep_py**2 + lep_pz**2 + lep[:, 3]**2)
    nu_E = torch.sqrt(nu[:, 0]**2 + nu[:, 1]**2 + nu[:, 2]**2)

    mW_sq = (lep_E + nu_E)**2 - (lep_px + nu[:, 0])**2 \
            - (lep_py + nu[:, 1])**2 - (lep_pz + nu[:, 2])**2
    mW = torch.sqrt(F.softplus(mW_sq, beta=1.0, threshold=20.0).clamp(min=1.0))

    return mW

def _nu_bjet_dR(nu_px, nu_py, pz, raw_b):
    """Compute deltaR between neutrino (given pz) and each b-jet.

    Args:
        nu_px, nu_py: (n,) neutrino px, py
        pz: (n,) neutrino pz
        raw_b: (n, 10) raw b-jet features [pt, eta, phi, mass, btag] x 2
    Returns:
        dR_b1, dR_b2: (n,) deltaR to each b-jet
    """
    nu_E = torch.sqrt(nu_px**2 + nu_py**2 + pz**2 + 1e-8)
    nu_eta = torch.atanh((pz / nu_E).clamp(-1 + 1e-6, 1 - 1e-6))
    nu_phi = torch.atan2(nu_py, nu_px)
    b = raw_b.view(-1, 5, 2)  # (n, 5, 2)
    b1_eta, b1_phi = b[:, 1, 0], b[:, 2, 0]
    b2_eta, b2_phi = b[:, 1, 1], b[:, 2, 1]
    dphi1 = torch.remainder(nu_phi - b1_phi + math.pi, 2 * math.pi) - math.pi
    dphi2 = torch.remainder(nu_phi - b2_phi + math.pi, 2 * math.pi) - math.pi
    dR_b1 = torch.sqrt((nu_eta - b1_eta)**2 + dphi1**2 + 1e-8)
    dR_b2 = torch.sqrt((nu_eta - b2_eta)**2 + dphi2**2 + 1e-8)
    return dR_b1, dR_b2

def _deta_solutions(raw_lep, nu_px, nu_py, pz_sol1, pz_sol2):
    """Compute delta-eta between lepton and neutrino for both pz solutions.

    Args:
        raw_lep: (n, 6) raw lepton features [pt, eta, phi, mass, isE, isM]
        nu_px, nu_py: (n,) corrected neutrino px, py
        pz_sol1, pz_sol2: (n,) the two pz solutions from the W mass constraint
    Returns:
        deta_sol1, deta_sol2: (n,) delta-eta for each solution
    """
    lep_eta = raw_lep[:, 1]
    nu_E_sol1 = torch.sqrt(nu_px**2 + nu_py**2 + pz_sol1**2 + 1e-8)
    nu_E_sol2 = torch.sqrt(nu_px**2 + nu_py**2 + pz_sol2**2 + 1e-8)
    eta_nu_sol1 = torch.atanh((pz_sol1 / nu_E_sol1).clamp(-1 + 1e-6, 1 - 1e-6))
    eta_nu_sol2 = torch.atanh((pz_sol2 / nu_E_sol2).clamp(-1 + 1e-6, 1 - 1e-6))
    return eta_nu_sol1 - lep_eta, eta_nu_sol2 - lep_eta

def _hadW_mass(raw_nb, ww_weights):
    """Compute hadronic W mass from the two highest-attention non-b jets.

    Args:
        raw_nb: (n, 4*nj) flat raw non-b jet features [pt, eta, phi, mass] per jet
        ww_weights: (n, heads, 1, nj) detached attention weights over nj jets
    Returns:
        (n, 1, 1) hadronic W candidate mass
    """
    n = raw_nb.shape[0]
    nb = raw_nb.view(n, 4, -1)  # (n, 4, nj)
    nj = nb.shape[2]
    if nj < 2:
        return torch.zeros(n, 1, 1, device=raw_nb.device)

    # Average attention across heads: (n, heads, 1, nj) → (n, nj)
    attn = ww_weights.squeeze(2).mean(dim=1)  # (n, nj)

    # Zero out attention for padded jets (pt == -1)
    padded = (nb[:, 0, :] < 0)  # (n, nj)
    attn = attn.masked_fill(padded, 0.0)

    _, top2 = attn.topk(2, dim=1)  # (n, 2) # Select top-2 jets by attention weight per event
    top2_exp = top2.unsqueeze(1).expand(-1, 4, -1)  # (n, 4, 2)
    sel_jets = torch.gather(nb, 2, top2_exp)  # (n, 4, 2)

    # Compute PxPyPzE for each selected jet and get dijet mass
    j1 = PxPyPzE(sel_jets[:, :, 0])  # (n, 4)
    j2 = PxPyPzE(sel_jets[:, :, 1])  # (n, 4)
    mass = diObjectMass(j1, j2)  # (n, 1)
    
    return mass.unsqueeze(-1)  # (n, 1, 1)


def compute_mjj(raw_nb, nj):
    """For each non-b jet, compute min m_jj over its pairings with other jets.

    Args:
        raw_nb: (n, 4*nj) flat raw non-b jet features [pt, eta, phi, mass] per jet
        nj: number of jets to consider
    Returns:
        (n, 1, nj) per-jet m_jj (large value for padded jets)
    """
    n = raw_nb.shape[0]
    nb = raw_nb.view(n, 4, -1)[:, :, :nj]  # (n, 4, nj)
    device = raw_nb.device

    # Build all C(nj,2) pair indices
    idx_i, idx_j = [], []
    for i in range(nj):
        for j in range(i + 1, nj):
            idx_i.append(i)
            idx_j.append(j)

    # Compute dijet masses for all pairs
    v1 = PxPyPzE(nb[:, :, idx_i])  # (n, 4, n_pairs)
    v2 = PxPyPzE(nb[:, :, idx_j])  # (n, 4, n_pairs)
    mjj = diObjectMass(v1, v2)     # (n, 1, n_pairs)

    # Map pair masses back to individual masses (to pass to attention mechanism)
    BIG = 999.0
    per_jet = torch.full((n, 1, nj), BIG, device=device)
    for p, (i, j) in enumerate(zip(idx_i, idx_j)):
        per_jet[:, :, i] = torch.min(per_jet[:, :, i], mjj[:, :, p])
        per_jet[:, :, j] = torch.min(per_jet[:, :, j], mjj[:, :, p])

    # Mask padded jets (pt < 0)
    padded = (nb[:, 0, :] < 0).unsqueeze(1)  # (n, 1, nj)
    per_jet = per_jet.masked_fill(padded, BIG)

    return per_jet  # (n, 1, nj)


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
            5,
            features_out=self.dD,
            phase_symmetric=phase_symmetric,
            conv=True,
            name="jet embedder",
        )  # (pt, eta, phi, mass, btagScore)

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

        self.bsl, self.wsl = 2, 4
        self.qqsl = self.wsl * (self.wsl - 1) // 2  # C(wsl, 2) = 6 dijet pairs
        self.wsl_tt = 3  # Use only first 3 jets for TT attention (reduces n_tt from 48 to 24)
        self.qqsl_tt = self.wsl_tt * (self.wsl_tt - 1) // 2  # C(3, 2) = 3 dijet pairs for TT

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
            5 + self.qqsl,  # qqsl dijet masses + mbb + lnu_mT + dphi_lep_met + pt_bb + dphi_bb_met
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
            nb[:, :, (0, 0, 0, 1, 1, 2)], nb[:, :, (1, 2, 3, 2, 3, 3)]
        )

        ## top reconstruction
        bWhad, bWhadPxPyPzE = addFourVectors(
            b[:, :, (0, 1)].unsqueeze(3),  # [batch, 4, 2, 1]
            qq.unsqueeze(2)                # [batch, 4, 1, qqsl]
        )
        bWlep, bWlepPxPyPzE = addFourVectors(
            b[:, :, (1, 1, 0, 0)],
            W_lep[:, :, (0, 1, 0, 1)] 
        )

        bb = bb.unsqueeze(2) # add a dimension to calculating MdR matrix symmetrically later
        bbPxPyPzE = bbPxPyPzE.unsqueeze(2)

        mask, bbMdR, qqMdR, bbnMdR, mask_bbMdR, mask_qqMdR, mask_bbn = None, None, None, None, None, None, None
        # Detect padded jets BEFORE appending label row (padded jets have pt == -1)
        mask = (nb[:, 0, :] < 0)  # (n, nj): True for padded jets
        nb = torch.cat(
            [nb, torch.ones((n, 1, 4), dtype=torch.float, device=device)], 1
        )
        # Set label to -1 for padded jets so downstream code stays consistent
        nb[:, -1, :][mask] = -1
        mask_qq = torch.stack([
            mask[:, 0] | mask[:, 1],  # qq[0] = nb[0] + nb[1]
            mask[:, 0] | mask[:, 2],  # qq[1] = nb[0] + nb[2]
            mask[:, 0] | mask[:, 3],  # qq[1] = nb[0] + nb[3]
            mask[:, 1] | mask[:, 2],  # qq[2] = nb[1] + nb[2]
            mask[:, 1] | mask[:, 3],  # qq[2] = nb[1] + nb[3]
            mask[:, 2] | mask[:, 3],  # qq[2] = nb[2] + nb[3]
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
        nb = torch.cat([nb, nb[:, :, (3,2,1,0)]] , 2)

        # only keep relative angular information so that learned features are invariant under global phi rotations and eta/phi flips
        b[:, 2:3, :] = calcDeltaPhi(bb, b[:, :, :]) # replace jet phi with deltaPhi between dijet and jet

        # Extract derived kinematics as a (batch, 5+qqsl, 1) tensor
        mjj_all = qq[:, 3:4, :]  # (batch, 1, qqsl) all dijet masses
        mbb = bb[:, 3:4, 0:1]     # bb mass

        dphi_lep_met = calcDeltaPhi(l, nu)
        pt_bb = bb[:, 0:1, 0:1]
        dphi_bb_met = calcDeltaPhi(bb, nu)

        derived_kinematics = torch.cat([
            mjj_all.transpose(1, 2),  # (batch, qqsl, 1) all W mass candidates
            mbb,                      # Higgs mass candidate
            lnu_mT,                   # transverse mass
            dphi_lep_met,             # angular separation lep-MET
            pt_bb,                    # pT of bb system
            dphi_bb_met               # angular separation bb-MET
        ], dim=1)  # Shape: (batch, 5+qqsl, 1)

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
                mask_bWhad,  # (n, bsl*qqsl=12)
                mask_bWlep.view(n, -1)  # (n, 4)
            ),
            dim=1
        )  # Result: (n, 16)


        bWhad = bWhad.view(n, 4, -1)  # (n, 4, 2, qqsl) -> (n, 4, bsl*qqsl=12)
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
        mask_nb =  torch.cat([mask, mask[:, [3,2,1,0]]], 1) # augment mask from 2 to 4, matching pattern for jets
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
        bbqqMdR = bbqqMdR.view(n, 2, self.qqsl)
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
            n, self.dD, 1, self.qqsl
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

        bWhadMdR = MdRtt[:, :, :self.bsl * self.qqsl].view(
            n, self.dD, self.bsl, self.qqsl
        )
        bWlepMdR = MdRtt[:, :, self.bsl * self.qqsl:].view(
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
  

class HypothesisClassifier(nn.Module):
    """Siamese classifier for on-shell vs off-shell leptonic W.

    Shared-weight scorer evaluates each hypothesis independently, then subtracts.
    Positive logit → on-shell preferred.

    Per-branch input: shared_base (2*dD+5) + oss_corr (1) + nu_det (3) + mW (1) + sigma_pz (1) = 2*dD+11
    """

    def __init__(self, dD):
        super().__init__()
        # Shared scorer: processes each hypothesis branch independently
        dH = dD * 4
        self.hypothesis_scorer = nn.Sequential(
            GhostBatchNorm1d(2*dD + 12, features_out=dH, conv=True),
            NonLUModule(),
            GhostBatchNorm1d(dH, features_out=dH, conv=True),
            NonLUModule(),
            GhostBatchNorm1d(dH, features_out=dH, conv=True),
            NonLUModule(),
            GhostBatchNorm1d(dH, features_out=1, conv=True),
        )

    def forward(self, on_features, off_features):
        # on_features, off_features: (n, 2*dD+11, 1)
        score_on = self.hypothesis_scorer(on_features).squeeze(-1).squeeze(-1)   # (n,)
        score_off = self.hypothesis_scorer(off_features).squeeze(-1).squeeze(-1)  # (n,)
        return score_on - score_off  # positive → on-shell
      
class METRegressor(nn.Module):
    def __init__(
        self,
        dijetFeatures,
        ancillaryFeatures,
        device="cuda",
        architecture="bbWWBase",
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
            scalar_dim = 7,
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
            scalar_dim = self.inputEmbed.qqsl_tt + 1,  # 3 lepQQdR + 1 lnu_mT = 4
            inputLayers=[self.bWhadResNetBlock.conv[-1], self.bWlepResNetBlock.conv[-1]],
            device=self.device,
        )
        self.layers.addLayer(self.attention_tt, self.attention_tt.inputLayers)

        self.bsl = self.inputEmbed.bsl
        self.wsl = self.inputEmbed.wsl
        self.qqsl = self.inputEmbed.qqsl
        self.wsl_tt = self.inputEmbed.wsl_tt
        self.qqsl_tt = self.inputEmbed.qqsl_tt

        self.scalars_embed = GhostBatchNorm1d(
            self.qqsl_tt + 1,  # qqsl_tt lepQQdR values + 1 lnu_mT (only first 3 jets for TT)
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

        # Embed per-jet min |m_jj - mW| for attention bias
        self.jet_mjj_embed = GhostBatchNorm1d(
            1,
            features_out=self.dD,
            conv=True,
            name="jet dijet mass embedder",
        )

        # Learned combination of deltaR and dijet mass embeddings (2*dD → dD)
        self.qv_combine = GhostBatchNorm1d(
            2 * self.dD,
            features_out=self.dD,
            conv=True,
            name="qv combine deltaR+mjj",
        )

        self.onshell_classifier = HypothesisClassifier(self.dD)

        dH = self.dD * 4  # wider hidden dim for regressor heads

        # On-shell neutrino regressor: outputs (dpx, dpy)
        # +dD: lep_W0, +1: lnu_mT, +2: init_px/py, +1: hadW_mass = 2*dD+4
        self.nu_regressor_onshell = nn.Sequential(
            GhostBatchNorm1d(2*self.dD + 4, features_out=dH, conv=True),   # expand
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
        # Siamese pz solution scorer: shared weights score each solution independently.
        # Per-solution input: onshell_input (2*dD+4) + deta (1) + pz (1) + oss_corr (1)
        #                     + dR_b1 (1) + dR_b2 (1) = 2*dD+9
        # Runs twice (once per solution) with shared weights; output = score1 - score2.
        dH_sel = self.dD * 6  # wider hidden dim for selector (more feature interactions)
        self.pz_solution_scorer = nn.Sequential(
            GhostBatchNorm1d(2*self.dD + 9, features_out=dH_sel, conv=True),
            NonLUModule(),
            GhostBatchNorm1d(dH_sel, features_out=dH_sel, conv=True),
            NonLUModule(),
            GhostBatchNorm1d(dH_sel, features_out=dH_sel, conv=True),
            NonLUModule(),
            GhostBatchNorm1d(dH_sel, features_out=1, conv=True),         # scalar score
        )
        # Off-shell neutrino regressor: context + lep_W0 + 4 pz solutions + lnu_mT + hadW_mass
        # +dD: lep_W0, +4: pz solutions, +1: lnu_mT, +1: hadW_mass = 2*dD+6
        # Extra hidden layer vs on-shell head: unconstrained 3D regression is harder
        self.nu_regressor_offshell = nn.Sequential(
            GhostBatchNorm1d(2*self.dD + 6, features_out=dH, conv=True),  # expand
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
            GhostBatchNorm1d(2*self.dD + 4, features_out=dH, conv=True),
            NonLUModule(),
            GhostBatchNorm1d(dH, features_out=dH, conv=True),
            NonLUModule(),
            GhostBatchNorm1d(dH, features_out=6, conv=True),
        )
        self.nu_cholesky_offshell = nn.Sequential(
            GhostBatchNorm1d(2*self.dD + 6, features_out=dH, conv=True),
            NonLUModule(),
            GhostBatchNorm1d(dH, features_out=dH, conv=True),
            NonLUModule(),
            GhostBatchNorm1d(dH, features_out=6, conv=True),
        )

        # Post-hoc BDT gate for on-shell vs off-shell selection at inference.
        # Input: (p_onshell, sigma_pz_on, sigma_pz_off) → binary decision.
        # Fitted on validation data after training completes; replaces hard cuts.
        # Stored as a sklearn GradientBoostingClassifier (set by fit_selector_gate).
        self.selector_gate_bdt = None

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
        self.jet_mjj_embed.setGhostBatches(nGhostBatches)
        self.qv_combine.setGhostBatches(nGhostBatches)
        # Output heads: iterate GBN layers in sequential modules and classifier
        for name, module in [("nu_regressor_onshell", self.nu_regressor_onshell),
                             ("pz_solution_scorer", self.pz_solution_scorer),
                             ("nu_regressor_offshell", self.nu_regressor_offshell),
                             ("nu_cholesky_onshell", self.nu_cholesky_onshell),
                             ("nu_cholesky_offshell", self.nu_cholesky_offshell),
                             ("onshell_classifier.hypothesis_scorer", self.onshell_classifier.hypothesis_scorer)]:
            for layer in module:
                if hasattr(layer, "setGhostBatches"):
                    layer.setGhostBatches(nGhostBatches)
        self.nGhostBatches = nGhostBatches


    def forward(self, b, nb, l, nu, a):
        self.forwardCalls += 1
        # Save raw inputs before embedding overwrites them
        raw_met = nu.clone()  # (n, 2): [pt, phi]
        raw_lep = l.clone()   # (n, 6): [pt, eta, phi, mass, isE, isM]
        raw_b   = b.clone()   # (n, 10): [pt, eta, phi, mass, btag] x 2 b-jets
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
        # Scalars for TT attention: only first 3 jets (wsl_tt) to reduce combinatorial overhead
        scalars_tt = torch.cat([lepQQdR[:, :, :self.qqsl_tt], lnu_mT], dim=-1).squeeze(1)
        # Scalars for WW attention: all jets
        scalars = torch.cat([lepQQdR, lnu_mT], dim=-1).squeeze(1)

        n_bWhad_tt = self.bsl * self.qqsl_tt  # 2*3=6 top_had candidates
        n_bWlep = self.bsl * 2                 # 4 top_lep candidates
        n_tt = n_bWhad_tt * n_bWlep            # 24 total pairings

        # Slice bWhadMdR to first qqsl_tt pairs per b-jet
        bWhadMdR_tt = bWhadMdR[:, :, :, :self.qqsl_tt]  # (n, dD, bsl, qqsl_tt)
        bWhad_exp = bWhadMdR_tt.reshape(n, -1, n_bWhad_tt).repeat_interleave(n_bWlep, dim=2)
        bWlep_exp = bWlepMdR.squeeze(-1).repeat(1, 1, n_bWhad_tt)

        bbn_flat = bbnMdR.squeeze(2)[:, :, :self.wsl_tt]  # (n, dD, wsl_tt=3)
        # Map each qq pair to its constituent nonbjets: C(wsl_tt,2) pairs
        qq_idx_tt = []
        for i in range(self.wsl_tt):
            for j in range(i + 1, self.wsl_tt):
                qq_idx_tt.append((i, j))
        bbn_qq = torch.cat([
            torch.cat([bbn_flat[:, :, i:i+1], bbn_flat[:, :, j:j+1]], dim=1)
            for i, j in qq_idx_tt
        ], dim=2)  # (n, 2*dD, qqsl_tt=3)
        # Expand: repeat_interleave(4) for bWlep, repeat(2) for b-jets
        bbn_exp = bbn_qq.repeat_interleave(n_bWlep, dim=2).repeat(1, 1, self.bsl)  # (n, 2*dD, n_tt=24)

        bbqq_exp = bbqqMdR.squeeze(2)[:, :, :self.qqsl_tt]  # (n, dD, qqsl_tt=3)
        bbqq_exp = bbqq_exp.repeat_interleave(n_bWlep, dim=2).repeat(1, 1, self.bsl)  # (n, dD, 24)

        qv_tt = torch.cat([bWhad_exp, bWlep_exp, bbn_exp, bbqq_exp], dim=1)
        qv_tt = self.qv_embed(qv_tt)

        # Mask: bWhad[0:qqsl_tt] use b0, can't pair with bWlep[2:4] (also b0)
        #        bWhad[qqsl_tt:2*qqsl_tt] use b1, can't pair with bWlep[0:2] (also b1)
        mask_tt = torch.zeros(n, n_bWhad_tt, n_bWlep, dtype=torch.bool, device=self.device)
        mask_tt[:, :self.qqsl_tt, 2:4] = True
        mask_tt[:, self.qqsl_tt:, 0:2] = True

        # Slice bWhad to first qqsl_tt pairs per b-jet for TT attention
        # bWhad is (n, dD, bsl*qqsl=12), need indices [0:qqsl_tt] and [qqsl:qqsl+qqsl_tt]
        bWhad_tt_idx = list(range(self.qqsl_tt)) + list(range(self.qqsl, self.qqsl + self.qqsl_tt))
        bWhad_tt = bWhad[:, :, bWhad_tt_idx]  # (n, dD, 6)
        bWhad0_tt = bWhad0[:, :, bWhad_tt_idx]

        # TTbar pairing selection
        TT, TT0, TT_weights = self.attention_tt(
            bWhad_tt, bWlep, mask_tt, bWhad0_tt, qv_tt, scalars_tt, debug=self.debug
        )
        TT_logits = self.select_tt(TT)  # Shape: (n, n_bWhad_tt, 1)
        TT_logits = TT_logits.view(n, n_bWhad_tt)  # Shape: (n, 6)
        TT_score = F.softmax(TT_logits, dim=-1)  # Shape: (n, 6)
        TT_context = torch.matmul(TT, TT_score.unsqueeze(-1))

        # Individual jet attention: leptonic W queries individual non-b jets
        nb_jets = nb[:, :, :self.wsl]          # (n, dD, wsl) original jets (drop augmented permutations)
        jet_mask = mask_bbn.view(n, self.wsl)  # (n, wsl) per-jet padding mask

        # Compute deltaR between lepton and individual jets from raw kinematics
        nb_raw = raw_nb.view(n, 4, -1)[:, :, :self.wsl]  # (n, 4, wsl) original raw jets
        lep_raw = raw_lep.view(n, 6, 1)
        lepNBdR = calcDeltaR(lep_raw, nb_raw)      # (n, 1, wsl)
        jet_dR = self.jet_dR_embed(lepNBdR, jet_mask)  # (n, dD, wsl) embedded deltaR

        # Per-jet min |m_jj - mW|: how close is this jet's best pairing to a W?
        jet_mjj = compute_mjj(raw_nb, self.wsl)  # (n, 1, wsl)
        jet_mjj = self.jet_mjj_embed(jet_mjj, jet_mask)  # (n, dD, wsl)

        # Combined attention bias: concat deltaR and dijet mass, then project (2*dD → dD)
        jet_qv = self.qv_combine(torch.cat([jet_dR, jet_mjj], dim=1), jet_mask)  # (n, dD, wsl)

        WW, WW0, WW_weights = self.attention_WW(
            lep_W,                     # q:  (n, dD, 1) single leptonic W query
            nb_jets,                   # v:  (n, dD, wsl) individual jets
            jet_mask.unsqueeze(1),     # mask: (n, 1, wsl)
            lep_W0,                    # q0: (n, dD, 1) residual
            jet_qv,                    # qv: (n, dD, wsl) deltaR + dijet mass attention bias
            scalars,
            self.debug
        )
        # WW is (n, dD, 1) - enriched leptonic W after attending to jets
        WW_sel = WW

        # Per-jet attention weights (attached for jet_attn_loss gradient flow)
        # Concatenate heads to preserve per-head information: (n, h, 1, wsl) -> (n, h*wsl)
        jet_weights = WW_weights.squeeze(2).reshape(n, -1)  # (n, h*wsl)
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

        # Hadronic W mass from attention-selected jets (available early, only needs raw_nb + WW_weights)
        hadW_mass = _hadW_mass(raw_nb, WW_weights.detach())  # (n, 1, 1)

        # Shared enriched input: full_context + lep_W0 + init_px + init_py + hadW_mass
        # Used by on-shell px/py regressor, Siamese pz selector, and on-shell Cholesky (2*dD + 3)
        lnu_mT_feat = lnu_mT_raw.unsqueeze(1)         # (n, 1, 1)
        onshell_input = torch.cat([
            full_context, lep_W0, lnu_mT_feat,
            init_px.unsqueeze(1), init_py.unsqueeze(1),
            hadW_mass,
        ], dim=1)  # (n, 2*dD+3, 1)

        # initial pz: average of 80/40 GeV W mass constraint solutions for on/off shell
        init_pz_on = 0.5 * (kinematic_solutions[:, 0, :] + kinematic_solutions[:, 1, :])  # (n, 1)
        init_pz_off = 0.5 * (kinematic_solutions[:, 3, :] + kinematic_solutions[:, 4, :])  # (n, 1)

        nu_init_off = torch.cat([init_px, init_py, init_pz_off], dim=1)  # (n, 3)
        delta_on = self.nu_regressor_onshell(onshell_input).squeeze(-1)  # (n, 2): dpx, dpy
        nu_px_on = init_px.squeeze(1) + delta_on[:, 0]
        nu_py_on = init_py.squeeze(1) + delta_on[:, 1]

        # Solve W mass quadratic with corrected (px, py) at constant mW=80.379
        pz_sol1, pz_sol2, _, oss_corrected = get_nu_pz_cartesian(
            raw_lep[:, 0], raw_lep[:, 1], raw_lep[:, 2], raw_lep[:, 3],
            nu_px_on, nu_py_on, mW=80.379,
        )

        # Classifier: uses oss_80 (raw) + oss_corrected (from regressed px/py) + W mass discriminant
        oss_80 = torch.log1p(kinematic_solutions[:, 2:3, :] + 1e-3)  # (n, 1, 1)
        oss_corr = torch.log1p(oss_corrected.detach() + 1e-3).unsqueeze(-1).unsqueeze(-1)  # (n, 1, 1)

        # Lepton-neutrino rapidity gap for both pz solutions (from corrected MET)
        deta_sol1, deta_sol2 = _deta_solutions(raw_lep, nu_px_on, nu_py_on, pz_sol1, pz_sol2)

        # Siamese pz selector: score each solution with shared weights, then subtract
        oss_corr_sel = torch.log1p(oss_corrected + 1e-3).unsqueeze(-1).unsqueeze(-1)  # (n, 1, 1)

        # DeltaR between neutrino (per pz solution) and each b-jet
        dR_b1_sol1, dR_b2_sol1 = _nu_bjet_dR(nu_px_on, nu_py_on, pz_sol1, raw_b)
        dR_b1_sol2, dR_b2_sol2 = _nu_bjet_dR(nu_px_on, nu_py_on, pz_sol2, raw_b)

        sol1_input = torch.cat([
            onshell_input,
            deta_sol1.unsqueeze(-1).unsqueeze(-1),
            pz_sol1.unsqueeze(-1).unsqueeze(-1),
            oss_corr_sel,
            dR_b1_sol1.unsqueeze(-1).unsqueeze(-1),
            dR_b2_sol1.unsqueeze(-1).unsqueeze(-1),
        ], dim=1)  # (n, 2*dD+8, 1)

        sol2_input = torch.cat([
            onshell_input,
            deta_sol2.unsqueeze(-1).unsqueeze(-1),
            pz_sol2.unsqueeze(-1).unsqueeze(-1),
            oss_corr_sel,
            dR_b1_sol2.unsqueeze(-1).unsqueeze(-1),
            dR_b2_sol2.unsqueeze(-1).unsqueeze(-1),
        ], dim=1)  # (n, 2*dD+8, 1)

        score1 = self.pz_solution_scorer(sol1_input).squeeze(-1).squeeze(-1)  # (n,)
        score2 = self.pz_solution_scorer(sol2_input).squeeze(-1).squeeze(-1)  # (n,)
        logit_sol = score1 - score2  # positive → prefer sol1

        # Binary select: sigmoid(logit_sol) > 0.5 → use sol1, else sol2 for analytic nu_pz
        use_sol1 = logit_sol > 0.0  # equivalent to sigmoid > 0.5
        nu_pz_on = torch.where(use_sol1, pz_sol1, pz_sol2)

        nu_pred_on = torch.stack([nu_px_on, nu_py_on, nu_pz_on], dim=1)  # (n, 3): px, py, pz
        logit_sol_on = logit_sol

        # --- Off-shell neutrino: regress all 3 components ---
        pz_solutions = kinematic_solutions[:, [0, 1, 3, 4], :]  # (n, 4, 1): pz1_80, pz2_80, pz1_40, pz2_40
        offshell_input = torch.cat(
            [full_context, lep_W0, pz_solutions, lnu_mT_feat, hadW_mass], dim=1
            )  # (n, 2*dD+6, 1)
        delta_off = self.nu_regressor_offshell(offshell_input).squeeze(-1)  # (n, 3)
        nu_pred_off = nu_init_off + delta_off

        L_on = _build_cholesky(self.nu_cholesky_onshell(onshell_input).squeeze(-1))
        L_off = _build_cholesky(self.nu_cholesky_offshell(offshell_input).squeeze(-1))

        # --- Siamese hypothesis classifier ---
        nu_on_det = nu_pred_on.detach().unsqueeze(-1)   # (n, 3, 1)
        nu_off_det = nu_pred_off.detach().unsqueeze(-1)  # (n, 3, 1)
        mW_on_const = torch.full((n, 1, 1), 80.379, device=raw_lep.device)  # (n, 1, 1)
        mW_off = calc_mW(raw_lep[:, :4], nu_pred_off.detach()).unsqueeze(-1).unsqueeze(-1)  # (n, 1, 1)
        sigma_pz_on = L_on[:, 2, 2].detach().unsqueeze(-1).unsqueeze(-1)   # (n, 1, 1)
        sigma_pz_off = L_off[:, 2, 2].detach().unsqueeze(-1).unsqueeze(-1)  # (n, 1, 1)
        oss_40 = torch.log1p(kinematic_solutions[:, 5:6, :] + 1e-3)  # (n, 1, 1)

        # Corrected discriminant from off-shell regressor's px/py at mW=80 (symmetric with oss_corr)
        _, _, _, oss_corr_off = get_nu_pz_cartesian(
            raw_lep[:, 0], raw_lep[:, 1], raw_lep[:, 2], raw_lep[:, 3],
            nu_pred_off[:, 0].detach(), nu_pred_off[:, 1].detach(), mW=80.379,
        )
        oss_corr_off = torch.log1p(oss_corr_off + 1e-3).unsqueeze(-1).unsqueeze(-1)  # (n, 1, 1)

        # Shared base: event-level features independent of hypothesis
        shared_base = torch.cat([
            onshell_input, oss_80, oss_40,
        ], dim=1)  # (n, 2*dD+5, 1)

        # On-shell branch: hypothesis-specific features
        on_features = torch.cat([
            shared_base,
            oss_corr,         # corrected discriminant from on-shell regressor at mW=80
            nu_on_det,        # (n, 3, 1)
            mW_on_const,               # constant 80.379 GeV
            sigma_pz_on,      # pz uncertainty from on-shell Cholesky
        ], dim=1)  # (n, 2*dD+11, 1)

        # Off-shell branch: hypothesis-specific features
        off_features = torch.cat([
            shared_base,
            oss_corr_off,     # corrected discriminant from off-shell regressor at mW=80
            nu_off_det,       # (n, 3, 1)
            mW_off,           # reconstructed off-shell W mass
            sigma_pz_off,     # pz uncertainty from off-shell Cholesky
        ], dim=1)  # (n, 2*dD+11, 1)

        # Classify on-shell vs off-shell
        logit_onshell = self.onshell_classifier(on_features, off_features)

        return nu_pred_on, L_on, nu_pred_off, L_off, (logit_onshell, logit_sol_on)

    def setStore(self, store):
        self.store = store
        self.inputEmbed.store = store
        self.inputEmbed.storeData = self.storeData

    def writeStore(self):
        # print(self.storeData)
        print(self.store)
        np.save(self.store, self.storeData)
