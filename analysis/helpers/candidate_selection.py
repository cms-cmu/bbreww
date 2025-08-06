import awkward as ak
import numpy as np

def candidate_selection(events, params, year, is_mc):
    j_clean = events.Jet[events.Jet.isclean]
    j_soft = j_clean[j_clean.issoft]
    events['j_init'] = j_soft ## TEMP: initial preselected jets

    QvG_key = 'btagPNetQvG' if '202' in year else 'particleNetAK4_QvsG'
    bTag_key = 'btagPNetB' if '202' in year else 'particleNetAK4_B'
    btag_threshold = params.bTagWPs[bTag_key][year]['loose'] # using loose working point

    j_candidates = j_soft[ak.argsort(j_soft.pt, axis=1, ascending=False)]
    j_candidates = j_candidates[ak.argsort(getattr(j_candidates,bTag_key), axis=1, ascending=False)]#particleNetAK4_B btagPNetB
    j_candidates = ak.mask(j_candidates,ak.num(j_candidates) >= 4) # proceed only if we have at least 4 ak4 jets
    j_bcand_pool = j_candidates[j_candidates.pt > 25.0]  # Only jets > 25 GeV for b-jets
    j_bcand_pool = j_bcand_pool[getattr(j_bcand_pool,bTag_key) > btag_threshold]

    events['has_2_bjets'] = ak.num(j_bcand_pool, axis=1) >= 2

    # Mask the entire event if not enough b-jets
    j_candidates = ak.mask(j_candidates, events.has_2_bjets)
    j_bcand_pool = ak.mask(j_bcand_pool, events.has_2_bjets)
    events['j_bcand'] = ak.pad_none(j_bcand_pool[:,:2], 2, axis=1)

    j_candidates = j_candidates[:,2:] # non b-jets
    events['j_init'] = j_candidates
    j_candidates = j_candidates[ak.argsort(getattr(j_candidates,QvG_key), axis=1, ascending=False)] #particleNetAK4_QvsG btagPNetQvG
    j_candidates = ak.pad_none(j_candidates[:,:3],3,axis=1) #top 3 quark vs gluon non b-jets
    j_candidates = j_candidates[ak.argsort(j_candidates.pt, axis=1, ascending=False)] #pt sort the jets

    jj_i = ak.argcombinations(j_candidates,2,replacement = False, fields=["j1","j2"]) #take dijet combinations
    events['dijet_combs'] = jj_i
    jj_i = ak.mask(jj_i,(j_candidates[jj_i.j1]-j_candidates[jj_i.j2]).eta<2.0)
    jj_i = ak.mask(jj_i,(j_candidates[jj_i.j1]+ j_candidates[jj_i.j2]).mass<120.0) #dijet cuts
    
    events['j_nonbcand'] = j_candidates # 3 non b-jets
    events['j_sublead'] =  j_candidates[jj_i.j2] # subleading jet

    events['qq'] = ak.pad_none(j_candidates[jj_i.j1] + j_candidates[jj_i.j2], 3, axis=1)
    if is_mc: events['qq_gen_mass'] = ak.pad_none((j_candidates[jj_i.j1].matched_gen + j_candidates[jj_i.j2].matched_gen).mass, 3, axis=1)
    events['mbb'] = (events.j_bcand[:,0]+events.j_bcand[:,1]).mass

    return events