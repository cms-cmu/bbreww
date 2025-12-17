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
        events['GenPart','isNu'] = ((abs(gen.pdgId)==12)|(abs(gen.pdgId)==14))#& gen.hasFlags(['isPrompt'])

        #print(ak.local_index(events.GenPart[events.GenPart.isW].pdgId))
        #print(events.GenPart[events.GenPart.isLepW].pdgId)
        
        
        events['isHtoW'] = events.GenPart[(events.GenPart[events.GenPart[events.GenPart.isW].genPartIdxMother].pdgId== 25)]

        ## non-bjets gen matched with W jets decaying to quarks
        gen_qFromW = gen_match(events.GenPart, [1,2,3,4], [24])
        events['gen_bFromH'] = gen_match(events.GenPart, [5], [25])

        try:
            events['Jet', 'isQfromW']= ak.any(gen_qFromW.metric_table(events.Jet)< 0.2,axis=1)
            events['Jet', 'isGenFromW'] = ak.sum(events.Jet.isQfromW, axis=1) == 2

            if 'HH' in events.metadata['dataset']:
                events['Jet', 'isbFromH'] = ak.any(events.gen_bFromH.metric_table(events.Jet)< 0.2,axis=1)

            ## flag which W is on shell (only for signal)
            is_lep = ((events.GenPart[events.GenPart.genPartIdxMother].isW) &
                ((abs(events.GenPart.pdgId) == 11) | (abs(events.GenPart.pdgId) == 13))) # electrons or muons
            
            lepWidx = gen[is_lep].genPartIdxMother
            lepW = gen[lepWidx]

            hadWidx = gen[~is_lep & (abs(gen.pdgId) <= 5)].genPartIdxMother
            hadW = events.GenPart[hadWidx]
            hadW = hadW[hadW.isW]
            events['gen_hadW'] = hadW[:,0] # (pick 0 index because there are duplicate W's due to 2 quarks) 
            print(events.gen_hadW.pt)
            events['isLepW'] = lepW.mass > events.gen_hadW.mass 

        except:
            events['Jet', 'isQfromW'] = ak.zeros_like(events.Jet.pt, dtype=bool)
            events['isLepW'] = ak.ones_like(events.Jet.pt, dtype = bool)

        events['isHtoW'] = events.GenPart[(events.GenPart[events.GenPart[events.GenPart.isW].genPartIdxMother].pdgId== 25)]

        if 'HH' in events.metadata['dataset']:
            events['Jet', 'isbFromH'] = ak.any(events.gen_bFromH.metric_table(events.Jet)< 0.2,axis=1)

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

def gen_studies(events, is_mc):
    if is_mc:
        ## gen level studies
        events = add_gen_info(events, is_mc)
        gen_nu= ak.firsts(events.GenPart[events.GenPart.isNu])
        gen_W= events.GenPart[events.GenPart.isW]
        gen_b = ak.pad_none(events.gen_bFromH, 2,axis=1)

        try:
            ## non-bjets gen matched with W jets decaying to quarks
            matched_jets_pre = events.Jet[events.Jet.isQfromW]
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
        except:
            pass #above sequence will fail for datasets that don't have jets in every event

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
