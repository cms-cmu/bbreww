import awkward as ak
import numpy as np

def candidate_selection(events, params, year):
    j_clean = events.Jet[events.Jet.isclean]
    events['j_init'] = j_clean[j_clean.preselected] # initial preselected jets

    QvG_key = 'btagPNetQvG' if '202' in year else 'particleNetAK4_QvsG' # use particleNET for quark vs. gluon tagging
    bTag_key = 'btagPNetB' if '202' in year else 'particleNetAK4_B' # use particleNET b-tagging
    btag_threshold = params[year].btagWP.M # using medium working point

    j_candidates = events.j_init[ak.argsort(events.j_init.pt, axis=1, ascending=False)] # pt sort to take higher pT when b-tag scores are tied
    j_candidates = j_candidates[ak.argsort(getattr(j_candidates,bTag_key), axis=1, ascending=False)]#particleNetAK4_B btagPNetB
    
    j_bcand_pool = j_candidates[j_candidates.pt > 25.0]  # Only jets > 25 GeV for b-jets
    j_bcand_pool = j_bcand_pool[getattr(j_bcand_pool,bTag_key) > btag_threshold]

    # bjets requirement
    events['has_2_bjets'] = ak.num(j_bcand_pool, axis=1) >= 2
    events['has_1_bjet'] = ak.num(j_bcand_pool, axis=1) >= 1 #add for cutflow plot

    # Mask the entire event if not enough jets or b-jets   
    j_candidates = ak.mask(j_candidates, events.has_2_bjets)
    j_bcand_pool = ak.mask(j_bcand_pool, events.has_2_bjets)
    events['j_bcand'] = j_bcand_pool[:,:2]
    j_bcand_sorted = events.j_bcand[ak.argsort(events.j_bcand.pt, axis=1, ascending=False)]
    events['j_bcand_lead'] = j_bcand_sorted[:,0]
    events['j_bcand_sublead'] = j_bcand_sorted[:,1] 

    # nominal non-bjet selection
    j_candidates_nom = j_candidates[j_candidates.isnominal] # pt > 25 GeV jets (nominal)
    j_candidates_nom = ak.mask(j_candidates_nom[:,2:], ak.num(j_candidates_nom[:,2:],axis=1)>=2) # require 2 or more non-bjets
    j_candidates_nom = j_candidates_nom[ak.argsort(j_candidates_nom.pt, axis=1, ascending=False)] # pT sort the jets
    j_candidates_nom = j_candidates_nom[:,:2] # take leading two pT jets
    events['qq_nom'] = j_candidates_nom[:,0] + j_candidates_nom[:,1]

    # soft jets analysis
    j_candidates = j_candidates[ak.argsort(getattr(j_candidates,QvG_key), axis=1, ascending=False)] #particleNetAK4_QvsG btagPNetQvG
    j_candidates = j_candidates[:,:3] #top 3 quark vs gluon non b-jets
    j_candidates = j_candidates[ak.argsort(j_candidates.pt, axis=1, ascending=False)] #pt sort the jets
    events['j_nonbcand'] = j_candidates[:,2:]

    jj_i = ak.argcombinations(j_candidates,2,replacement = False, fields=["j1","j2"]) #take dijet combinations
    jj_i = jj_i[(j_candidates[jj_i.j1]-j_candidates[jj_i.j2]).eta<2.0]
    #jj_i = jj_i[(j_candidates[jj_i.j1]+ j_candidates[jj_i.j2]).mass<120.0] #dijet cuts
    events['dijet_combs'] = jj_i
    
    events['j_sublead'] =  j_candidates[jj_i.j2] # subleading jet

    events['qq_mass'] = ak.fill_none((j_candidates[jj_i.j1] + j_candidates[jj_i.j2]).mass,np.nan) # plotting gives issues with None values
    events['qq_soft'] = ak.pad_none(j_candidates[jj_i.j1] + j_candidates[jj_i.j2], 3, axis=1)
    events['mbb'] = (events.j_bcand[:,0]+events.j_bcand[:,1]).mass
    events['bb_dr'] = events.j_bcand[:,0].delta_r(events.j_bcand[:,1])

    events['njets'] = ak.fill_none(ak.num(j_clean[j_clean.isnominal],axis=1),np.nan)

    return events

## function only for skimmer
def bjet_flag(events,params,year):
    j_clean = events.Jet[events.Jet.isclean]
    j_soft = j_clean[j_clean.preselected]
    events['j_init'] = j_soft # initial preselected jets

    QvG_key = 'btagPNetQvG' if '202' in year else 'particleNetAK4_QvsG' # use particleNET for quark vs. gluon tagging
    bTag_key = 'btagPNetB' if '202' in year else 'particleNetAK4_B' # use particleNET b-tagging
    btag_threshold = params[year].btagWP.L # using loose working point

    j_candidates = j_soft[ak.argsort(j_soft.pt, axis=1, ascending=False)]
    j_candidates = j_candidates[ak.argsort(getattr(j_candidates,bTag_key), axis=1, ascending=False)]#particleNetAK4_B btagPNetB
    j_bcand_pool = j_candidates[j_candidates.pt > 25.0]  # Only jets > 25 GeV for b-jets
    j_bcand_pool = j_bcand_pool[getattr(j_bcand_pool,bTag_key) > btag_threshold]

    events['has_1_bjet'] = ak.num(j_bcand_pool, axis=1) >= 1

    return events
