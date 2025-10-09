import awkward as ak
import numpy as np
from bbreww.analysis.helpers.common import met_reconstr

def Hbb_candidate_selection(events):

    # this funciton called assuming we have applied basic preseleciton
    #   >= 2 bjets

    Hbb_cand = events.b_cands[:,0] + events.b_cands[:,1]
    Hbb_cand["lead"] = events.b_cands[:,0]
    Hbb_cand["subl"] = events.b_cands[:,1]
    Hbb_cand["st"]   = Hbb_cand["lead"].pt + Hbb_cand["subl"].pt
    Hbb_cand["dr"]   = Hbb_cand["lead"].delta_r  (Hbb_cand["subl"])
    Hbb_cand["dphi"] = Hbb_cand["lead"].delta_phi(Hbb_cand["subl"])

    events['Hbb_cand'] = Hbb_cand

    return events



def candidate_selection(events, params, year):

    #
    #  Wlnu Cand
    #

    # calculate MET pz requiring (lepton + nu).mass == W_mass
    nu = met_reconstr(events, events.leading_lep)

    Wlnu_cand = events.leading_lep + nu
    Wlnu_cand["lep"] = events.leading_lep
    Wlnu_cand["nu"]  = nu
    Wlnu_cand["dr"]   = Wlnu_cand["lep"].delta_r  (Wlnu_cand["nu"])
    Wlnu_cand["dphi"] = Wlnu_cand["lep"].delta_phi(Wlnu_cand["nu"])
    Wlnu_cand["mT"]   = np.sqrt(2 * Wlnu_cand.lep.pt * Wlnu_cand.nu.pt * (1 - np.cos(Wlnu_cand.dphi)))

    events['Wlnu_cand'] = Wlnu_cand

    #
    #  Wqq for Nominal Analysis
    #
    Wqq_cand = events.q_cands_nom[:,0] + events.q_cands_nom[:,1]
    Wqq_cand["lead"] = events.q_cands_nom[:,0]
    Wqq_cand["subl"] = events.q_cands_nom[:,1]
    Wqq_cand["st"]   = Wqq_cand["lead"].pt + Wqq_cand["subl"].pt
    Wqq_cand["dr"]   = Wqq_cand["lead"].delta_r  (Wqq_cand["subl"])
    Wqq_cand["dphi"] = Wqq_cand["lead"].delta_phi(Wqq_cand["subl"])

    events['Wqq_cand'] = Wqq_cand

    #
    # soft jets analysis
    #
    QvG_key = 'btagPNetQvG' if '202' in year else 'particleNetAK4_QvsG' # use particleNET for quark vs. gluon tagging

    q_cands_soft = events.q_cands_soft[ak.argsort(getattr(events.q_cands_soft,QvG_key), axis=1, ascending=False)] #particleNetAK4_QvsG btagPNetQvG
    q_cands_soft = q_cands_soft[:,:3] #top 3 quark vs gluon non b-jets
    q_cands_soft = q_cands_soft[ak.argsort(q_cands_soft.pt, axis=1, ascending=False)] #pt sort the jets
    events['q_cands_soft'] = q_cands_soft

    jj_i = ak.argcombinations(q_cands_soft, 2, replacement = False, fields=["j1","j2"]) #take dijet combinations
    jj_i = jj_i[(q_cands_soft[jj_i.j1] - q_cands_soft[jj_i.j2]).eta<2.0]
    #jj_i = jj_i[(q_cands_soft[jj_i.j1] + q_cands_soft[jj_i.j2]).mass<120.0] #dijet cuts
    events['dijet_combs_new'] = jj_i

    events['j_lead_new'] =  q_cands_soft[jj_i.j1] # leading jet
    events['j_sublead_new'] =  q_cands_soft[jj_i.j2] # subleading jet

    events['qq_mass'] = ak.fill_none((q_cands_soft[jj_i.j1] + q_cands_soft[jj_i.j2]).mass,np.nan) # plotting gives issues with None values
    events['qq_soft'] = ak.pad_none(q_cands_soft[jj_i.j1] + q_cands_soft[jj_i.j2], 3, axis=1)

    return events

## function only for skimmer
def bjet_flag(events,params,year):
    j_clean = events.Jet[events.Jet.isclean]
    j_soft = j_clean[j_clean.preselected]
    events['j_init'] = j_soft # initial preselected jets

    # QvG_key = 'btagPNetQvG' if '202' in year else 'particleNetAK4_QvsG' # use particleNET for quark vs. gluon tagging
    bTag_key = 'btagPNetB' if '202' in year else 'particleNetAK4_B' # use particleNET b-tagging
    btag_threshold = params[year].btagWP.L # using loose working point

    j_candidates = j_soft[ak.argsort(j_soft.pt, axis=1, ascending=False)]
    j_candidates = j_candidates[ak.argsort(getattr(j_candidates,bTag_key), axis=1, ascending=False)]#particleNetAK4_B btagPNetB
    j_bcand_pool = j_candidates[j_candidates.pt > 25.0]  # Only jets > 25 GeV for b-jets
    j_bcand_pool = j_bcand_pool[getattr(j_bcand_pool,bTag_key) > btag_threshold]

    events['has_1_bjet'] = ak.num(j_bcand_pool, axis=1) >= 1

    return events
