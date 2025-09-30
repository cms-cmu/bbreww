import awkward as ak
import numpy as np
from bbreww.analysis.helpers.common import chi_square, met_reconstr

def chi_sq(events):
    leading_mu = events.Muon[events.Muon.istight]
    leading_e = events.Electron[events.Electron.istight]

    #select leading lepton out of electrons/muons. Use ak.singletons to slice entries, not whole events
    leading_lep = ak.firsts(ak.concatenate([leading_e[ak.singletons(events.flavor.e)],
                                            leading_mu[ak.singletons(events.flavor.mu)]],axis=1))
    events['leading_lep'] = ak.with_name(leading_lep, 'PtEtaPhiMLorentzVector') #reapply 4-vector behavior after concatenate

    # hadronic W* chi square calculation
    nu = met_reconstr(events, events.leading_lep) # calculate MET pz requiring (lepton + nu).mass == W_mass
    mlvqq_hadWs_nom = (events.leading_lep + nu + events.Wqq_cand ).mass # H -> lvqq candidates (nonbjet_pt > 25 GeV)
    mlvqq_hadWs_soft = (events.leading_lep + nu + events.qq_soft ).mass # H -> lvqq candidates (15 GeV > nonbjet_pt > 25 GeV)
    events['mlvqq_hadWs'] = ak.fill_none(mlvqq_hadWs_nom,np.nan)

    #individual chi squares for hadronic W* signal selection
    chi1_hadWs = chi_square(events.mbb,113.22, 30.74) # H -> bb
    chi2_hadWs_nom = chi_square(mlvqq_hadWs_nom, 161.15, 34.23) # H -> lvqq in nominal region
    chi2_hadWs_soft = chi_square(mlvqq_hadWs_soft, 161.15, 34.23) # H -> lvqq in low pt region
    chi3_hadWs_nom = chi_square(events.Wqq_cand.mass,39.13, 10.02) # W* -> qq in nominal region
    chi3_hadWs_soft = chi_square(events.qq_soft.mass,39.13, 10.02) # W* -> qq in low pt region
    chi4_hadWs = chi_square(events.bb_dr,1.90, 0.70) #delta R between b-jets


    #total chi square
    chi_sq_hadWs_nom_4j = ak.singletons(np.sqrt(chi1_hadWs + chi2_hadWs_nom + chi3_hadWs_nom + chi4_hadWs))# 4 jet region chi square
    chi_sq_hadWs_nom_3j = ak.singletons(np.sqrt(chi1_hadWs + chi4_hadWs)) # can't use di-jet variables for 3 jet region
    chi_sq_hadWs_soft = np.sqrt(chi1_hadWs + chi2_hadWs_soft + chi3_hadWs_soft + chi4_hadWs)
    min_chi_sq_hadWs_soft = ak.argmin(chi_sq_hadWs_soft, axis=1, keepdims = True) #index of the minimum chi square non-bjet pairs
    chi_sq_hadWs_soft = chi_sq_hadWs_soft[min_chi_sq_hadWs_soft]

    #combined chi square based on selection region (need singletons here to not remove whole events)
    chi_sq_hadWs =   ak.concatenate([chi_sq_hadWs_soft[ak.singletons(events.lowpt_4j2b)],
                                   chi_sq_hadWs_nom_4j[ak.singletons(events.nominal_4j2b)],
                                   chi_sq_hadWs_nom_3j[ak.singletons(events.nominal_3j2b)]], axis=1)

    ## hadronic W chi square calculation
    events['mT_leading_lep'] = np.sqrt(2*events.leading_lep.pt*events.MET.pt*(1-np.cos(events.MET.delta_phi(events.leading_lep))))

    #individual chi squares for hadronic W signal selection
    chi1_hadW = chi_square(events.mbb,111.13, 23.63) # H -> bb
    chi2_hadW = chi_square(events.mT_leading_lep, 58.87, 37.35) #transverse mass
    chi3_hadW_nom = chi_square(events.Wqq_cand.mass,76.84, 10.98) #hadronic W
    chi3_hadW_soft = chi_square(events.qq_soft.mass,76.84, 10.98) #hadronic W
    chi4_hadW = chi_square(events.bb_dr,1.69, 0.60) #delta R between b-jets

    chi_sq_hadW_nom_4j = ak.singletons(np.sqrt(chi1_hadW + chi2_hadW + chi3_hadW_nom + chi4_hadW))
    chi_sq_hadW_nom_3j = ak.singletons(np.sqrt(chi1_hadW + chi2_hadW + chi4_hadW)) # don't use dijet variables for 3j region
    chi_sq_hadW_soft = np.sqrt(chi1_hadW + chi2_hadW + chi3_hadW_soft + chi4_hadW)
    min_chi_sq_hadW_soft= ak.argmin(chi_sq_hadW_soft, axis=1, keepdims = True) #index of the minimum chi square non-bjet pair
    chi_sq_hadW_soft = chi_sq_hadW_soft[min_chi_sq_hadW_soft]

    # combine chi square values for all selection regions
    chi_sq_hadW =   ak.concatenate([chi_sq_hadW_soft[ak.singletons(events.lowpt_4j2b)],
                                   chi_sq_hadW_nom_4j[ak.singletons(events.nominal_4j2b)],
                                   chi_sq_hadW_nom_3j[ak.singletons(events.nominal_3j2b)]], axis=1)

    ## ttbar reconstruction

    #leptonic top
    mlvb1 = (events.leading_lep + nu + events.b_cands[:,0]).mass
    mlvb2 = (events.leading_lep + nu + events.b_cands[:,1]).mass


    mbqq1_soft = ak.pad_none((events.b_cands[:,0] + events.qq_soft).mass,3,axis=1) #hadronic candidate 1
    mbqq2_soft = ak.pad_none((events.b_cands[:,1] + events.qq_soft).mass,3,axis=1) #hadronic candidate 2
    mbqq1_nom = ak.singletons((events.b_cands[:,0] + events.Wqq_cand).mass) #hadronic candidate 1
    mbqq2_nom = ak.singletons((events.b_cands[:,1] + events.Wqq_cand).mass) #hadronic candidate 2

    def distance(x1,y1,x2,y2):
        return ak.fill_none(np.sqrt((x2-x1)**2+(y2-y1)**2),np.nan)

    #ttbar candidates
    tt1_soft = ak.cartesian({"t1":mlvb1,"t2":mbqq2_soft},axis=1)
    tt2_soft = ak.cartesian({"t1":mlvb2,"t2":mbqq1_soft},axis=1)
    tt1_nom = ak.cartesian({"t1":mlvb1,"t2":mbqq2_nom},axis=1)
    tt2_nom = ak.cartesian({"t1":mlvb2,"t2":mbqq1_nom},axis=1)
    b_sel_soft = abs(distance(tt1_soft.t1,tt1_soft.t2,172.5,172.5))< abs(distance(tt2_soft.t1,tt2_soft.t2,172.5,172.5)) #pick pair closest to ttbar mass
    b_sel_nom =  abs(distance(tt1_nom.t1,tt1_nom.t2,172.5,172.5)) < abs(distance(tt2_nom.t1,tt2_nom.t2,172.5,172.5)) #pick pair closest to ttbar mass

    #final ttbar candidates
    tt_soft = ak.where(b_sel_soft, tt1_soft , tt2_soft)
    tt_nom = ak.where(b_sel_nom, tt1_nom , tt2_nom)

    chi1_tt_soft = chi_square(tt_soft.t1,165.55 , 35.49 ) #leptonic top
    chi2_tt_soft = chi_square(tt_soft.t2, 171.55, 44.95 ) #hadronic top
    chi1_tt_nom =  chi_square(tt_nom.t1,165.55 , 35.49 ) #leptonic top
    chi2_tt_nom =  chi_square(tt_nom.t2, 171.55, 44.95 ) #hadronic top
    chi3_tt_soft = chi_square(events.qq_soft.mass,73.9, 23.56) #hadronic W
    chi3_tt_nom = chi_square(events.Wqq_cand.mass,73.9, 23.56) #hadronic W
    chi4_tt = chi_square(events.bb_dr,2.30, 0.81) #delta R between b-jets

    chi_sq_tt_soft = np.sqrt(chi1_tt_soft + chi2_tt_soft + chi3_tt_soft + chi4_tt)
    chi_sq_tt_nom_4j =  np.sqrt(chi1_tt_nom + chi2_tt_nom + chi3_tt_nom + chi4_tt)
    min_chi_sq_tt_soft = ak.argmin(chi_sq_tt_soft, axis=1, keepdims = True) #get index of the minimum chi square
    events['chi_sq_tt'] = ak.where(events.lowpt_4j2b, chi_sq_tt_soft[min_chi_sq_tt_soft], chi_sq_tt_nom_4j)

    # select jets with lower chi square across two signal regions
    qq_sel_index = ak.where(chi_sq_hadW_soft <= chi_sq_hadWs_soft, min_chi_sq_hadW_soft, min_chi_sq_hadWs_soft)
    events['qq_sel_mass'] = events.qq_soft[qq_sel_index].mass
    events['sr_boolean'] = ak.where(events.qq_sel_mass > 55.0, 1, 0)

    events['leading_e'] =  ak.fill_none(events.Electron[events.Electron.istight], np.nan)
    events['leading_mu'] = ak.fill_none(events.Muon[events.Muon.istight],np.nan)

    events['chi_sq_hadWs'] = chi_sq_hadWs
    events['chi_sq_hadW'] = chi_sq_hadW

    return events

# apply chi square cuts
def chi_sq_cut(events):
    #apply cuts on chi square calculation in leptonic W region and 4 jets hadronic W region
    events['passChiSqTT'] = ak.firsts(events.sr_boolean == 0) | ak.firsts(events.chi_sq_tt > 1.0) # hadronic W region cut
    events['passChiSqLepW'] = ak.firsts(events.sr_boolean == 1) | ak.firsts(events.chi_sq_hadWs < 2.0) #leptonic W region cut
    events['passChiSqHadW'] = ak.where(events.nominal_3j2b, ak.firsts(events.chi_sq_hadW < 1.2), False)
    return events



def chi_sq_new(events):
    leading_mu = events.Muon[events.Muon.istight]
    leading_e  = events.Electron[events.Electron.istight]

    #select leading lepton out of electrons/muons. Use ak.singletons to slice entries, not whole events
    leading_lep = ak.firsts(ak.concatenate([leading_e[ak.singletons(events.flavor.e)],
                                            leading_mu[ak.singletons(events.flavor.mu)]],axis=1))
    events['leading_lep'] = ak.with_name(leading_lep, 'PtEtaPhiMLorentzVector') #reapply 4-vector behavior after concatenate

    # hadronic W* chi square calculation
    nu = met_reconstr(events, events.leading_lep) # calculate MET pz requiring (lepton + nu).mass == W_mass
    mlvqq_hadWs_nom = (events.leading_lep + nu + events.Wqq_cand ).mass # H -> lvqq candidates (nonbjet_pt > 25 GeV)
    mlvqq_hadWs_soft = (events.leading_lep + nu + events.qq_soft ).mass # H -> lvqq candidates (15 GeV > nonbjet_pt > 25 GeV)
    events['mlvqq_hadWs'] = ak.fill_none(mlvqq_hadWs_nom,np.nan)

    #individual chi squares for hadronic W* signal selection
    chi1_hadWs = chi_square(events.Hbb_cand.mass,113.22, 30.74) # H -> bb
    chi2_hadWs_nom = chi_square(mlvqq_hadWs_nom, 161.15, 34.23) # H -> lvqq in nominal region
    chi2_hadWs_soft = chi_square(mlvqq_hadWs_soft, 161.15, 34.23) # H -> lvqq in low pt region
    chi3_hadWs_nom = chi_square(events.Wqq_cand.mass,39.13, 10.02) # W* -> qq in nominal region
    chi3_hadWs_soft = chi_square(events.qq_soft.mass,39.13, 10.02) # W* -> qq in low pt region
    chi4_hadWs = chi_square(events.Hbb_cand.dr,1.90, 0.70) #delta R between b-jets

    #total chi square
    chi_sq_hadWs_nom_4j = ak.singletons(np.sqrt(chi1_hadWs + chi2_hadWs_nom + chi3_hadWs_nom + chi4_hadWs))# 4 jet region chi square
    chi_sq_hadWs_nom_3j = ak.singletons(np.sqrt(chi1_hadWs + chi4_hadWs)) # can't use di-jet variables for 3 jet region
    chi_sq_hadWs_soft = np.sqrt(chi1_hadWs + chi2_hadWs_soft + chi3_hadWs_soft + chi4_hadWs)
    min_chi_sq_hadWs_soft = ak.argmin(chi_sq_hadWs_soft, axis=1, keepdims = True) #index of the minimum chi square non-bjet pairs
    chi_sq_hadWs_soft = chi_sq_hadWs_soft[min_chi_sq_hadWs_soft]

    #combined chi square based on selection region (need singletons here to not remove whole events)
    chi_sq_hadWs =   ak.concatenate([chi_sq_hadWs_soft[ak.singletons(events.lowpt_4j2b)],
                                   chi_sq_hadWs_nom_4j[ak.singletons(events.nominal_4j2b)],
                                   chi_sq_hadWs_nom_3j[ak.singletons(events.nominal_3j2b)]], axis=1)

    ## hadronic W chi square calculation
    events['mT_leading_lep'] = np.sqrt(2*events.leading_lep.pt*events.MET.pt*(1-np.cos(events.MET.delta_phi(events.leading_lep))))

    #individual chi squares for hadronic W signal selection
    chi1_hadW = chi_square(events.Hbb_cand.mass,111.13, 23.63) # H -> bb
    chi2_hadW = chi_square(events.mT_leading_lep, 58.87, 37.35) #transverse mass
    chi3_hadW_nom = chi_square(events.Wqq_cand.mass,76.84, 10.98) #hadronic W
    chi3_hadW_soft = chi_square(events.qq_soft.mass,76.84, 10.98) #hadronic W
    chi4_hadW = chi_square(events.Hbb_cand.dr,1.69, 0.60) #delta R between b-jets

    chi_sq_hadW_nom_4j = ak.singletons(np.sqrt(chi1_hadW + chi2_hadW + chi3_hadW_nom + chi4_hadW))
    chi_sq_hadW_nom_3j = ak.singletons(np.sqrt(chi1_hadW + chi2_hadW + chi4_hadW)) # don't use dijet variables for 3j region
    chi_sq_hadW_soft = np.sqrt(chi1_hadW + chi2_hadW + chi3_hadW_soft + chi4_hadW)
    min_chi_sq_hadW_soft= ak.argmin(chi_sq_hadW_soft, axis=1, keepdims = True) #index of the minimum chi square non-bjet pair
    chi_sq_hadW_soft = chi_sq_hadW_soft[min_chi_sq_hadW_soft]

    # combine chi square values for all selection regions
    chi_sq_hadW =   ak.concatenate([chi_sq_hadW_soft[ak.singletons(events.lowpt_4j2b)],
                                   chi_sq_hadW_nom_4j[ak.singletons(events.nominal_4j2b)],
                                   chi_sq_hadW_nom_3j[ak.singletons(events.nominal_3j2b)]], axis=1)

    ## ttbar reconstruction

    #leptonic top
    mlvb1 = (events.leading_lep + nu + events.Hbb_cand.lead).mass
    mlvb2 = (events.leading_lep + nu + events.Hbb_cand.subl).mass

    mbqq1_soft = ak.pad_none((events.Hbb_cand.lead + events.qq_soft).mass,3,axis=1) #hadronic candidate 1
    mbqq2_soft = ak.pad_none((events.Hbb_cand.subl + events.qq_soft).mass,3,axis=1) #hadronic candidate 2
    mbqq1_nom  = ak.singletons((events.Hbb_cand.lead + events.Wqq_cand).mass) #hadronic candidate 1
    mbqq2_nom  = ak.singletons((events.Hbb_cand.subl + events.Wqq_cand).mass) #hadronic candidate 2

    def distance(x1,y1,x2,y2):
        return ak.fill_none(np.sqrt((x2-x1)**2+(y2-y1)**2),np.nan)

    #ttbar candidates
    tt1_soft = ak.cartesian({"t1":mlvb1,"t2":mbqq2_soft},axis=1)
    tt2_soft = ak.cartesian({"t1":mlvb2,"t2":mbqq1_soft},axis=1)
    tt1_nom = ak.cartesian({"t1":mlvb1,"t2":mbqq2_nom},axis=1)
    tt2_nom = ak.cartesian({"t1":mlvb2,"t2":mbqq1_nom},axis=1)
    b_sel_soft = abs(distance(tt1_soft.t1,tt1_soft.t2,172.5,172.5))< abs(distance(tt2_soft.t1,tt2_soft.t2,172.5,172.5)) #pick pair closest to ttbar mass
    b_sel_nom =  abs(distance(tt1_nom.t1,tt1_nom.t2,172.5,172.5)) < abs(distance(tt2_nom.t1,tt2_nom.t2,172.5,172.5)) #pick pair closest to ttbar mass

    #final ttbar candidates
    tt_soft = ak.where(b_sel_soft, tt1_soft , tt2_soft)
    tt_nom = ak.where(b_sel_nom, tt1_nom , tt2_nom)

    chi1_tt_soft = chi_square(tt_soft.t1,165.55 , 35.49 ) #leptonic top
    chi2_tt_soft = chi_square(tt_soft.t2, 171.55, 44.95 ) #hadronic top
    chi1_tt_nom =  chi_square(tt_nom.t1,165.55 , 35.49 ) #leptonic top
    chi2_tt_nom =  chi_square(tt_nom.t2, 171.55, 44.95 ) #hadronic top
    chi3_tt_soft = chi_square(events.qq_soft.mass,73.9, 23.56) #hadronic W
    chi3_tt_nom = chi_square(events.Wqq_cand.mass,73.9, 23.56) #hadronic W
    chi4_tt = chi_square(events.Hbb_cand.dr,2.30, 0.81) #delta R between b-jets

    chi_sq_tt_soft = np.sqrt(chi1_tt_soft + chi2_tt_soft + chi3_tt_soft + chi4_tt)
    chi_sq_tt_nom_4j =  np.sqrt(chi1_tt_nom + chi2_tt_nom + chi3_tt_nom + chi4_tt)
    min_chi_sq_tt_soft = ak.argmin(chi_sq_tt_soft, axis=1, keepdims = True) #get index of the minimum chi square
    events['chi_sq_tt'] = ak.where(events.lowpt_4j2b, chi_sq_tt_soft[min_chi_sq_tt_soft], chi_sq_tt_nom_4j)

    # select jets with lower chi square across two signal regions
    qq_sel_index = ak.where(chi_sq_hadW_soft <= chi_sq_hadWs_soft, min_chi_sq_hadW_soft, min_chi_sq_hadWs_soft)
    events['qq_sel_mass'] = events.qq_soft[qq_sel_index].mass
    events['sr_boolean'] = ak.where(events.qq_sel_mass > 55.0, 1, 0)

    events['leading_e'] =  ak.fill_none(events.Electron[events.Electron.istight], np.nan)
    events['leading_mu'] = ak.fill_none(events.Muon[events.Muon.istight],np.nan)

    events['chi_sq_hadWs'] = chi_sq_hadWs
    events['chi_sq_hadW'] = chi_sq_hadW

    return events

# apply chi square cuts
def chi_sq_cut_new(events):
    #apply cuts on chi square calculation in leptonic W region and 4 jets hadronic W region
    events['passChiSqTT'] = ak.firsts(events.sr_boolean == 0) | ak.firsts(events.chi_sq_tt > 1.0) # hadronic W region cut
    events['passChiSqLepW'] = ak.firsts(events.sr_boolean == 1) | ak.firsts(events.chi_sq_hadWs < 2.0) #leptonic W region cut
    events['passChiSqHadW'] = ak.where(events.nominal_3j2b, ak.firsts(events.chi_sq_hadW < 1.2), False)
    return events
