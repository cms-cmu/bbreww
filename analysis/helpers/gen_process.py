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

        events['isHtoW'] = events.GenPart[(events.GenPart[events.GenPart[events.GenPart.isW].genPartIdxMother].pdgId== 25)]

        ## non-bjets gen matched with W jets decaying to quarks
        gen_qFromW = gen_match(events.GenPart, [1,2,3,4], [24])
        events['gen_bFromH'] = gen_match(events.GenPart, [5], [25] )

        try:
            events['Jet', 'isQfromW']= ak.any(gen_qFromW.metric_table(events.Jet)< 0.2,axis=1)
            if 'HH' in events.metadata['dataset']:
                events['Jet', 'isbFromH'] = ak.any(events.gen_bFromH.metric_table(events.Jet)< 0.2,axis=1)
        except: 
            events['Jet', 'isQfromW'] = ak.zeros_like(events.Jet.pt, dtype=bool)
    
        events['isHtoW'] = events.GenPart[(events.GenPart[events.GenPart[events.GenPart.isW].genPartIdxMother].pdgId== 25)]

        ## non-bjets gen matched with W jets decaying to quarks
        gen_qFromW = gen_match(events.GenPart, [1,2,3,4], [24])
        events['Jet', 'isQfromW']= ak.any(gen_qFromW.metric_table(events.Jet)< 0.2,axis=1)
        events['gen_bFromH'] = gen_match(events.GenPart, [5], [25] )

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
            matched_jets_pre = ak.mask(events.j_init,events.j_init.isQfromW)
            matched_jets_pre = matched_jets_pre[ak.argsort(matched_jets_pre.pt, axis=1, ascending=False)]
            events['true_ak4_1'] = matched_jets_pre[:,0]
            events['true_ak4_2'] = matched_jets_pre[:,1]
        except: 
            pass #above sequence will fail for datasets that don't have jets in every event

        '''events['Wjets_pre_lead'] = ak.pad_none(matched_jets_pre,2,axis=1)[:,0]
        events['Wjets_pre_sublead'] = ak.pad_none(matched_jets_pre,2,axis=1)[:,1]

        ## check if both jets are gen matched within an event
        true_dijet_mask = (ak.count(matched_jets_pre.pt,axis=1) == 2) 
        events['dijets_pre_lead'] = ak.mask(matched_jets_pre,true_dijet_mask)[:,0]
        events['dijets_pre_sublead'] = ak.mask(matched_jets_pre,true_dijet_mask)[:,1]

        # repeat procedure to check how many jets we lose to quark vs. gluon score selection
        matched_jets_post = ak.mask(events.j_nonbcand,events.j_nonbcand.isQfromW)
        matched_jets_post = matched_jets_post[ak.argsort(matched_jets_post.pt, axis=1, ascending=False)]
        events['Wjets_post_lead'] = ak.pad_none(matched_jets_post,2,axis=1)[:,0]
        events['Wjets_post_sublead'] = ak.pad_none(matched_jets_post,2,axis=1)[:,1]
        
        ## check if both jets are gen matched after quark vs. gluon selection
        true_dijet_mask = (ak.count(matched_jets_post.pt,axis=1) == 2) 
        events['dijets_post_lead'] = ak.mask(matched_jets_post,true_dijet_mask)[:,0]
        events['dijets_post_sublead'] = ak.mask(matched_jets_post,true_dijet_mask)[:,1]'''

        ## met and W mass resolution
        events['W_mass_res'] = ak.firsts(gen_W.mass[gen_W.mass < 55.0]) - events.qq_sel_mass
        events['genW_mass'] = gen_W.mass[gen_W.mass > 55.0] 
        #####################

        ### study input parameters to chi square 

        events['gen_bb'] = ak.fill_none(gen_b[:,0] + gen_b[:,1], np.nan)
        if 'HH' in events.metadata['dataset']:
            genjet_from_b =  ak.pad_none(events.j_bcand[events.j_bcand.isbFromH].matched_gen,2,axis=1)
            events['genjet_from_b'] = ak.fill_none(genjet_from_b[:,0] + genjet_from_b[:,1], np.nan)
            recojet_from_b = ak.pad_none(events.j_bcand[events.j_bcand.isbFromH], 2, axis=1)
            events['mass_reco_b_gen_match'] = ak.fill_none(recojet_from_b [:,0] + recojet_from_b[:,1], np.nan)

    return events
