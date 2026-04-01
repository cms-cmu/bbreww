import logging
import awkward as ak
import numpy as np
from functools import reduce
from coffea.nanoevents.methods import vector

def add_gen_info(events, is_mc):
    if is_mc:
        gen = events.GenPart
        events['GenPart','isTop'] = (abs(gen.pdgId)==6)&gen.hasFlags(['fromHardProcess', 'isLastCopy'])
        events['GenPart','isW'] = (abs(gen.pdgId)==24)&gen.hasFlags(['fromHardProcess', 'isLastCopy'])
        events['GenPart','isZ'] = (abs(gen.pdgId)==23)&gen.hasFlags(['fromHardProcess', 'isLastCopy'])
        is_genb = (abs(gen.pdgId)==5)&gen.hasFlags(['fromHardProcess', 'isLastCopy'])

        events['isHtoW'] = events.GenPart[(events.GenPart[events.GenPart[events.GenPart.isW].genPartIdxMother].pdgId== 25)]
        genNu = ak.pad_none(gen_match(events.GenPart, [12, 14], [24]), 1, axis=1)[:, 0]  # e, mu neutrinos coming from W decay
        events['genNu'] = genNu
        events['genNu_pt'] = ak.fill_none(genNu.pt, np.nan)
        events['genNu_eta'] = ak.fill_none(genNu.eta, np.nan)
        events['genNu_phi'] = ak.fill_none(genNu.phi, np.nan)
        events['genNu_pz'] = ak.fill_none(genNu.pz, np.nan)

        ## non-bjets gen matched with W jets decaying to quarks
        gen_qFromW = gen_match(events.GenPart, [1,2,3,4], [24])
        events['gen_bFromH'] = gen_match(events.GenPart, [5], [25])
        
        # find hadronically decaying tops
        top_parent_indices = get_ancestor_index(gen, 6) # all gen particles decaying from top
        is_light_quark = (abs(gen.pdgId) >= 1) & (abs(gen.pdgId) <= 4)
        is_from_W = (get_ancestor_id(gen, 24) == 24)
        parent_top_index = top_parent_indices[is_light_quark & is_from_W]
        parent_top_index = ak.fill_none(ak.firsts(parent_top_index), -999)

        # find b-jets from hadronically decaying top
        b_parent_top_idx = top_parent_indices[is_genb]
        is_b_from_had_top = ak.any(b_parent_top_idx == parent_top_index, axis=-1)
        gen_b_from_had_top = gen[is_genb][is_b_from_had_top]
        #isHadB= ak.any(gen_b_from_had_top.metric_table(events.Jet)< 0.2,axis=1)
        #is_max_pt = (events.Jet.pt == ak.max(events.Jet[isHadB].pt, axis=1, keepdims=True))
        #events['Jet', 'isHadB'] = isHadB & is_max_pt

        try:
            events['Jet', 'isQfromW']= ak.any(gen_qFromW.metric_table(events.Jet)< 0.2,axis=1)
            events['Jet', 'isGenFromW'] = ak.sum(events.Jet.isQfromW, axis=1) == 2
                
            ## flag which W is on shell (only for signal)
            is_lep = ((events.GenPart[events.GenPart.genPartIdxMother].isW) &
                ((abs(events.GenPart.pdgId) == 11) | (abs(events.GenPart.pdgId) == 13))) # electrons or muons
            
            lepWidx = gen[is_lep].genPartIdxMother
            gen_lepW = gen[lepWidx]
            events['gen_lepW_mass'] = ak.fill_none(
                ak.firsts(ak.values_astype(gen_lepW.mass, np.float32)),
                np.nan
            )

            hadWidx = gen[~is_lep & (abs(gen.pdgId) <= 5)].genPartIdxMother
            hadW = events.GenPart[hadWidx]
            hadW = hadW[hadW.isW]
            events['gen_hadW'] = hadW[:,0] # (pick 0 index because there are duplicate W's due to 2 quarks) 
            
            if 'HH' in events.metadata['dataset']:
                events['Jet', 'isbFromH'] = ak.any(events.gen_bFromH.metric_table(events.Jet)< 0.2,axis=1)
                events['isLepW'] = ak.fill_none(events.gen_lepW_mass > events.gen_hadW.mass, -1)
            else:
                events['isLepW'] = ak.ones_like(events.event) * -1
        except:
            events['Jet', 'isQfromW'] = ak.zeros_like(events.Jet.pt, dtype=bool)
            events['isLepW'] = ak.ones_like(events.event) * -1
            events['gen_lepW_mass'] = ak.full_like(events.event, np.nan, dtype=np.float32)
            
        events['isHtoW'] = events.GenPart[(events.GenPart[events.GenPart[events.GenPart.isW].genPartIdxMother].pdgId== 25)]

    else:
        # add placeholders when running on data
        events['gen_lepW_mass'] = ak.full_like(events.event, np.nan, dtype=np.float32)
        events['genNu_pt'] =  ak.full_like(events.event, np.nan, dtype=np.float32)
        events['genNu_eta'] = ak.full_like(events.event, np.nan, dtype=np.float32)
        events['genNu_phi'] = ak.full_like(events.event, np.nan, dtype=np.float32)
        events['genNu_pz'] =  ak.full_like(events.event, np.nan, dtype=np.float32)
        events['Jet', 'isQfromW']= ak.full_like(events.Jet.pt, np.nan, dtype = bool)
        events['isLepW'] =  ak.full_like(events.event, np.nan, dtype=bool)
        
    return events

def gen_process(events, weights):

    nnlo_nlo = {}
    nlo_qcd = ak.ones_like(events.MET.pt, dtype=float)
    nlo_ewk = ak.ones_like(events.MET.pt, dtype=float)


    ###
    # Isolation weights for muons
    ###

    if hasattr(events, "L1PreFiringWeight"):
        weights.add('prefiring', events.L1PreFiringWeight.Nom, events.L1PreFiringWeight.Up, events.L1PreFiringWeight.Dn)
    weights.add('genw',events.genWeight)
    weights.add('nlo_ewk',nlo_ewk)
    #weights.add('nlo',nlo)
    #if 'cen' in nnlo_nlo:
        #weights.add('nnlo_nlo',nnlo_nlo['cen'])
        #weights.add('qcd1',ak.ones_like(events.MET.pt, dtype=float), nnlo_nlo['qcd1up']/nnlo_nlo['cen'], nnlo_nlo['qcd1do']/nnlo_nlo['cen'])
        #weights.add('qcd2',ak.ones_like(events.MET.pt, dtype=float), nnlo_nlo['qcd2up']/nnlo_nlo['cen'], nnlo_nlo['qcd2do']/nnlo_nlo['cen'])
        #weights.add('qcd3',ak.ones_like(events.MET.pt, dtype=float), nnlo_nlo['qcd3up']/nnlo_nlo['cen'], nnlo_nlo['qcd3do']/nnlo_nlo['cen'])
        #weights.add('ew1',ak.ones_like(events.MET.pt, dtype=float), nnlo_nlo['ew1up']/nnlo_nlo['cen'], nnlo_nlo['ew1do']/nnlo_nlo['cen'])
        #weights.add('ew2G',ak.ones_like(events.MET.pt, dtype=float), nnlo_nlo['ew2Gup']/nnlo_nlo['cen'], nnlo_nlo['ew2Gdo']/nnlo_nlo['cen'])
        #weights.add('ew3G',ak.ones_like(events.MET.pt, dtype=float), nnlo_nlo['ew3Gup']/nnlo_nlo['cen'], nnlo_nlo['ew3Gdo']/nnlo_nlo['cen'])
        #weights.add('ew2W',ak.ones_like(events.MET.pt, dtype=float), nnlo_nlo['ew2Wup']/nnlo_nlo['cen'], nnlo_nlo['ew2Wdo']/nnlo_nlo['cen'])
        #weights.add('ew3W',ak.ones_like(events.MET.pt, dtype=float), nnlo_nlo['ew3Wup']/nnlo_nlo['cen'], nnlo_nlo['ew3Wdo']/nnlo_nlo['cen'])
        #weights.add('ew2Z',ak.ones_like(events.MET.pt, dtype=float), nnlo_nlo['ew2Zup']/nnlo_nlo['cen'], nnlo_nlo['ew2Zdo']/nnlo_nlo['cen'])
        #weights.add('ew3Z',ak.ones_like(events.MET.pt, dtype=float), nnlo_nlo['ew3Zup']/nnlo_nlo['cen'], nnlo_nlo['ew3Zdo']/nnlo_nlo['cen'])
        #weights.add('mix',ak.ones_like(events.MET.pt, dtype=float), nnlo_nlo['mixup']/nnlo_nlo['cen'], nnlo_nlo['mixdo']/nnlo_nlo['cen'])
        #weights.add('muF',ak.ones_like(events.MET.pt, dtype='float'), nnlo_nlo['muFup']/nnlo_nlo['cen'], nnlo_nlo['muFdo']/nnlo_nlo['cen'])
        #weights.add('muR',ak.ones_like(events.MET.pt, dtype='float'), nnlo_nlo['muRup']/nnlo_nlo['cen'], nnlo_nlo['muRdo']/nnlo_nlo['cen'])
    return weights


### copied from https://github.com/aebid/HHbbWW_Run3/blob/29a4943b313e6c006858b16a1c899acc11f11ace/python/genparticles.py#L9
def gen_match(genpart, pdgid, ancestors):
    """
    Find gen level particles given pdgId (and ancestors ids)

    Parameters:
    genpart (GenPart): NanoAOD GenPart collection.
    pdgid (list): pdgIds for the target particles.
    idmother (list): pdgIds for the ancestors of the target particles.

    Returns:
    NanoAOD GenPart collection
    """

    def check_id(p):
        return np.abs(genpart.pdgId) == p

    pid = reduce(np.logical_or, map(check_id, pdgid))

    if ancestors:
        ancs, ancs_idx = [], []
        for i, mother_id in enumerate(ancestors):
            if i == 0:
                mother_idx = genpart[pid].genPartIdxMother
            else:
                mother_idx = genpart[ancs_idx[i-1]].genPartIdxMother
            ancs.append(np.abs(genpart[mother_idx].pdgId) == mother_id)
            ancs_idx.append(mother_idx)

        decaymatch =  reduce(np.logical_and, ancs)
        return genpart[pid][decaymatch]

    return genpart[pid]

def get_ancestor_id(genpart, ancestor_id):
    # Start with the immediate mothers
    mother_idx = genpart.genPartIdxMother
    
    for _ in range(10): 
        # Get the PDG ID of the current 'mothers'
        valid_idx = mother_idx >= 0
        current_mother_ids = ak.where(valid_idx, abs(genpart[mother_idx].pdgId), 0)
        
        # If the mother is what we want, we're done for that particle
        # If not, we update mother_idx to the next level up
        mother_idx = ak.where((current_mother_ids != ancestor_id) & valid_idx, 
                              genpart[mother_idx].genPartIdxMother, 
                              mother_idx)
    
    return ak.where(mother_idx >= 0, abs(genpart[mother_idx].pdgId), 0)

def get_ancestor_index(genpart, target_pdgid):
    # Start with the immediate mothers of every particle
    current_idx = genpart.genPartIdxMother
    
    # We will track the 'best' index we've found so far
    # Initialize with -1 (no ancestor found)
    ancestor_idx = ak.full_like(current_idx, -1)

    # Climb the tree (10 iterations is usually the max depth for ttbar/HH)
    for _ in range(10):
        # 1. Identify where we are currently pointing
        in_bounds = (current_idx >= 0)
        current_pdg = ak.where(in_bounds, abs(genpart[current_idx].pdgId), 0)
        
        # 2. If the particle we are looking at IS the target (e.g. Top), 
        # save this index as our final answer for those specific particles.
        is_target = (current_pdg == target_pdgid)
        ancestor_idx = ak.where(is_target, current_idx, ancestor_idx)
        
        # 3. For particles where we HAVEN'T hit the target yet, 
        # move 'current_idx' up one more level to the mother of the current mother.
        current_idx = ak.where(in_bounds & ~is_target, 
                               genpart[current_idx].genPartIdxMother, 
                               current_idx)
                               
    return ancestor_idx

def gen_studies(events, is_mc, run_MET_regression):
    if is_mc:
        ## gen level studies
        events = add_gen_info(events, is_mc)
        gen_W= events.GenPart[events.GenPart.isW]
        gen_b = ak.pad_none(events.gen_bFromH, 2,axis=1)

        if 1==1:
            ## non-bjets gen matched with W jets decaying to quarks
            j_sel = events.Jet[events.Jet.isclean]
            j_sel = j_sel[j_sel.preselected]
            matched_jets_pre = j_sel[j_sel.isQfromW]
            matched_jets_pre = matched_jets_pre[ak.argsort(matched_jets_pre.pt, axis=1, ascending=False)]
            matched_jets_pre = ak.mask(matched_jets_pre.pt, (ak.sum(matched_jets_pre.isQfromW,axis=1) == 2))

            ### store only pT although not explicitly labeled
            events['true_ak4_1'] = ak.fill_none(matched_jets_pre[:,0],np.nan)
            events['true_ak4_2'] = ak.fill_none(matched_jets_pre[:,1],np.nan)
            
            sel_jets_soft = events.q_cands_soft[events.q_cands_soft.isQfromW]
            sel_jets_nom = events.q_cands_nom[events.q_cands_nom.isQfromW]
            
            events['q_soft_true_sublead'] =  ak.fill_none(ak.mask(sel_jets_soft.pt, 
                                                          (ak.sum(sel_jets_soft.isQfromW,axis=1) == 2))[:,1], 
                                                          np.nan)
            events['q_soft_true_lead'] =  ak.fill_none(ak.mask(sel_jets_soft.pt, 
                                                       (ak.sum(sel_jets_soft.isQfromW,axis=1) == 2))[:,0], 
                                                       np.nan)
            events['q_nom_true_sublead'] =  ak.fill_none(ak.mask(sel_jets_nom.pt, 
                                                         (ak.sum(sel_jets_nom.isQfromW,axis=1) == 2))[:,1], 
                                                         np.nan)
            events['q_nom_true_lead'] =  ak.fill_none(ak.mask(sel_jets_nom.pt, 
                                                      (ak.sum(sel_jets_nom.isQfromW,axis=1) == 2))[:,0], 
                                                      np.nan)

            if run_MET_regression:
                # non b-jets selected from regressor classification
                n_true_soft = ak.sum(events.q_cands_soft.isQfromW, axis=1)
                ml_lead_correct = (events.sel_qq_l.isQfromW == 1)
                ml_sublead_correct = (events.sel_qq_sl.isQfromW == 1)
                
                # both jets correct (denominator: 2 true jets in q_cands_soft)
                both_in_soft = n_true_soft == 2
                ml_both_correct = ml_lead_correct & ml_sublead_correct & both_in_soft
                events['q_ml_true_lead'] = ak.where(ml_both_correct, events.q_soft_true_lead, np.nan)
                events['q_ml_true_sublead'] = ak.where(ml_both_correct, events.q_soft_true_sublead, np.nan)
                
                # leading ML jet correct (denominator: >= 1 true jet in q_cands_soft)
                at_least_one = n_true_soft >= 1
                lead_pt = ak.fill_none(ak.mask(sel_jets_soft.pt, at_least_one)[:,0], np.nan)
                events['q_ml_lead_denom'] = lead_pt
                events['q_ml_lead_numer'] = ak.where(ml_lead_correct & at_least_one, lead_pt, np.nan)
                
           	# subleading ML jet correct (denominator: >= 2 true jets in q_cands_soft)
                events['q_ml_sublead_denom'] = events.q_soft_true_sublead  # already requires == 2
                events['q_ml_sublead_numer'] = ak.where(ml_sublead_correct & both_in_soft,
                                                        events.q_soft_true_sublead, np.nan)
                
                # p_onshell and sigma_pz_on for misclassified events with gen_lepW_mass < 20 GeV
                misclass_low_genW = abs(events.reg_mW - 80.0) <= 5.0 
                events['misclass_p_onshell'] = ak.where(misclass_low_genW, events.met_regressor.p_onshell , np.nan)
                events['misclass_sigma_pz_on'] = ak.where(misclass_low_genW, events.met_regressor.sigma_pz_on, np.nan)
        #except:
        #    logging.info("warning: skipping gen studies of true W jets due to error")
        #    pass #above sequence will fail for datasets that don't have jets in every event

        ## met and W mass resolution
        #events['W_mass_res'] = ak.firsts(gen_W.mass[gen_W.mass < 55.0]) - events.qq_sel_mass
        #events['genW_mass'] = gen_W.mass[gen_W.mass > 55.0]
        #####################
        
        ### study input parameters to chi square
        events['bjets_genjets_mass'] = ak.fill_none((events.b_cands[:,0].matched_gen + events.b_cands[:,1].matched_gen).mass,np.nan)
        events['bjets_genjets_dr'] = ak.fill_none(events.b_cands[:,0].matched_gen.delta_r(events.b_cands[:,1].matched_gen),np.nan)
        #events['bcand_genjets_mass'] = (events.b_cands[:,0].matched_gen + events.b_cands[:,1].matched_gen)
        events['gen_bb'] = ak.fill_none(gen_b[:,0] + gen_b[:,1], np.nan)
        
        if 'HH' in events.metadata['dataset']:
            genjet_from_b =  ak.pad_none(events.b_cands[events.b_cands.isbFromH].matched_gen,2,axis=1)
            events['genjet_from_b'] = ak.fill_none(genjet_from_b[:,0] + genjet_from_b[:,1], np.nan)
            recojet_from_b = ak.pad_none(events.b_cands[events.b_cands.isbFromH], 2, axis=1)
            events['mass_reco_b_gen_match'] = ak.fill_none(recojet_from_b [:,0] + recojet_from_b[:,1], np.nan)

    return events
