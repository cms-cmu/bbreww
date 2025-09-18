import awkward as ak
import numpy as np
from bbreww.analysis.helpers.common import chi_square, met_reconstr

def chi_sq(events):
    e_clean = events.Electron[events.Electron.isclean]
    leading_mu = ak.firsts(events.Muon[events.Muon.istight])
    leading_e = ak.firsts(e_clean[e_clean.istight])
    met = events.MET

    # hadronic W* signal reconstruction
    v_mu, v_e = met_reconstr(events, leading_e, leading_mu)
    # H -> lvqq with electrons and muons
    mevqq_nom = (leading_e + v_e + events.wqq_cand ).mass
    mmuvqq_nom = (leading_mu + v_mu + events.wqq_cand).mass
    mevqq_soft = (leading_e + v_e + events.qq_soft ).mass
    mmuvqq_soft = (leading_mu + v_mu + events.qq_soft).mass
    l_mu = ~ak.is_none(leading_mu.pt)
    l_e = ~ak.is_none(leading_e.pt)
    muge = leading_mu.pt > leading_e.pt

    # 0 = electron, 1 = muon, -1 = neither
    events['lepton_choice'] = ak.where(l_mu & l_e,
                            ak.where(muge, 1, 0),  # both present: 1 if mu>e, else 0
                            ak.where(l_mu, 1,      # only mu: 1
                                    ak.where(l_e, 0, -1)))  # only e: 0, neither: -1

    mlvqq_hadWs_nom = ak.where(l_mu & l_e,
                        ak.where(muge, mmuvqq_nom, mevqq_nom),  # both present: select muon calculation if mu>e else electron
                        ak.where(l_mu, mmuvqq_nom, mevqq_nom))     # only muon present
    mlvqq_hadWs_soft = ak.where(l_mu & l_e,
                        ak.where(muge, mmuvqq_soft, mevqq_soft),  # both present: select muon calculation if mu>e else electron
                        ak.where(l_mu, mmuvqq_soft, mevqq_soft))     # only muon present
    events['mlvqq_hadWs'] = ak.fill_none(mlvqq_hadWs_nom,np.nan)


    #individual chi squares for hadronic W* signal selection
    chi1_hadWs = chi_square(events.mbb,116.02, 45.04) # H -> bb
    chi2_hadWs_nom = chi_square(mlvqq_hadWs_nom, 161.15, 34.23) # H -> lvqq in nominal region
    chi2_hadWs_soft = chi_square(mlvqq_hadWs_soft, 161.15, 34.23) # H -> lvqq in low pt region
    chi3_hadWs_nom = chi_square(events.wqq_cand.mass,39.13, 10.02) #hadronic W* in nominal region
    chi3_hadWs_soft = chi_square(events.qq_soft.mass,39.13, 10.02) #hadronic W* in low pt region
    chi4_hadWs = chi_square(events.hbb_cand.dr,1.79, 0.79) #delta R between b-jets

    #total chi square
    chi_sq_hadWs_nom_4j = np.sqrt(chi1_hadWs + chi2_hadWs_nom + chi3_hadWs_nom + chi4_hadWs) # 4 jet region chi square
    chi_sq_hadWs_nom_3j = np.sqrt(chi1_hadWs + chi4_hadWs) # can't use di-jet variables for 3 jet region
    chi_sq_hadWs_soft = np.sqrt(chi1_hadWs + chi2_hadWs_soft + chi3_hadWs_soft + chi4_hadWs)
    min_chi_sq_hadWs_soft = ak.argmin(chi_sq_hadWs_soft, axis=1, keepdims = True) #index of the minimum chi square non-bjet pairs

    #combined chi square based on selection region
    chi_sq_hadWs = ak.where(events.selection.lowpt_4j2b, chi_sq_hadWs_soft[min_chi_sq_hadWs_soft],
                            ak.where(events.selection.nominal_4j2b, ak.singletons(chi_sq_hadWs_nom_4j),
                            ak.singletons(chi_sq_hadWs_nom_3j)))

    #transverse mass
    mT = {
        'esr'  : np.sqrt(2*leading_e.pt*met.pt*(1-np.cos(met.delta_phi(leading_e)))),
        'msr'  : np.sqrt(2*leading_mu.pt*met.pt*(1-np.cos(met.delta_phi(leading_mu))))
    }

    events['mT_leading_lep'] =  ak.where(l_mu & l_e,
                        ak.where(muge, mT['msr'], mT['esr']),
                        ak.where(l_mu, mT['msr'], mT['esr']))

    #individual chi squares for hadronic W signal selection
    chi1_hadW = chi_square(events.mbb,112.46, 46.61) # H -> bb
    chi2_hadW = chi_square(events.mT_leading_lep, 58.87, 37.35) #transverse mass
    chi3_hadW_nom = chi_square(events.wqq_cand.mass,76.84, 10.98) #hadronic W
    chi3_hadW_soft = chi_square(events.qq_soft.mass,76.84, 10.98) #hadronic W
    chi4_hadW = chi_square(events.hbb_cand.dr,1.76, 0.81) #delta R between b-jets

    chi_sq_hadW_nom_4j = np.sqrt(chi1_hadW + chi2_hadW + chi3_hadW_nom + chi4_hadW)
    chi_sq_hadW_nom_3j = np.sqrt(chi1_hadW + chi2_hadW + chi4_hadW) # don't use dijet variables for 3j region
    chi_sq_hadW_soft = np.sqrt(chi1_hadW + chi2_hadW + chi3_hadW_soft + chi4_hadW)
    min_chi_sq_hadW_soft= ak.argmin(chi_sq_hadW_soft, axis=1, keepdims = True) #index of the minimum chi square non-bjet pair
    chi_sq_hadW =  ak.where(events.selection.lowpt_4j2b, chi_sq_hadW_soft[min_chi_sq_hadW_soft],
                            ak.where(events.selection.nominal_4j2b, ak.singletons(chi_sq_hadW_nom_4j),
                            ak.singletons(chi_sq_hadW_nom_3j)))

    ## ttbar reconstruction

    #leptonic top with electrons
    mevb1 = (leading_e + v_e + events.b_cands[:,0]).mass
    mevb2 = (leading_e + v_e + events.b_cands[:,1]).mass

    #leptonic top with muons
    mmvb1 = (leading_mu + v_mu + events.b_cands[:,0]).mass
    mmvb2 = (leading_mu + v_mu + events.b_cands[:,1]).mass

    mlvb1 = ak.where(l_mu & l_e,
                ak.where(muge, mmvb1, mevb1),
                ak.where(l_mu, mmvb1, mevb1)) #leptonic candidate 1
    mlvb2 = ak.where(l_mu & l_e,
                ak.where(muge, mmvb2, mevb2),
                ak.where(l_mu, mmvb2, mevb2)) #leptonic candidate 2

    mbqq1_soft = ak.pad_none((events.b_cands[:,0] + events.qq_soft).mass,3,axis=1) #hadronic candidate 1
    mbqq2_soft = ak.pad_none((events.b_cands[:,1] + events.qq_soft).mass,3,axis=1) #hadronic candidate 2
    mbqq1_nom = ak.singletons((events.b_cands[:,0] + events.wqq_cand).mass) #hadronic candidate 1
    mbqq2_nom = ak.singletons((events.b_cands[:,1] + events.wqq_cand).mass) #hadronic candidate 2
    def distance(x1,y1,x2,y2):
        return np.sqrt((x2-x1)**2+(y2-y1)**2)

    #ttbar candidates
    tt1_soft = ak.cartesian({"t1":mlvb1,"t2":mbqq2_soft},axis=1)
    tt2_soft = ak.cartesian({"t1":mlvb2,"t2":mbqq1_soft},axis=1)
    tt1_nom = ak.cartesian({"t1":mlvb1,"t2":mbqq2_nom},axis=1)
    tt2_nom = ak.cartesian({"t1":mlvb2,"t2":mbqq1_nom},axis=1)
    b_sel_soft = abs(distance(tt1_soft.t1,tt1_soft.t2,172.5,172.5))< abs(distance(tt2_soft.t1,tt2_soft.t2,172.5,172.5)) #pick pair closest to ttbar mass
    b_sel_nom =  abs(distance(tt1_nom.t1,tt1_nom.t2,172.5,172.5)) < abs(distance(tt2_nom.t1,tt2_nom.t2,172.5,172.5)) #pick pair closest to ttbar mass

    #conditions to work around None values
    c1_soft = ~ak.is_none(distance(tt1_soft.t1, tt1_soft.t2,172.5,172.5))
    c2_soft = ~ak.is_none(distance(tt2_soft.t1, tt2_soft.t2,172.5,172.5))
    c1_nom = ~ak.is_none(distance(tt1_nom.t1, tt1_nom.t2,172.5,172.5))
    c2_nom = ~ak.is_none(distance(tt2_nom.t1, tt2_nom.t2,172.5,172.5))

    #final ttbar candidates
    tt_soft = ak.pad_none(ak.where( c1_soft & c2_soft, ak.where(b_sel_soft, tt1_soft , tt2_soft),
                              ak.where(c1_soft, tt1_soft, tt2_soft)),3,axis=1)
    tt_nom = ak.pad_none(ak.where( c1_nom & c2_nom, ak.where(b_sel_nom, tt1_nom , tt2_nom),
                            ak.where(c1_nom, tt1_nom, tt2_nom)),3,axis=1)

    chi1_tt_soft = chi_square(tt_soft.t1,165.55 , 35.49 ) #leptonic top
    chi2_tt_soft = chi_square(tt_soft.t2, 171.55, 44.95 ) #hadronic top
    chi1_tt_nom =  chi_square(tt_nom.t1,165.55 , 35.49 ) #leptonic top
    chi2_tt_nom =  chi_square(tt_nom.t2, 171.55, 44.95 ) #hadronic top
    chi3_tt_soft = chi_square(events.qq_soft.mass,73.9, 23.56) #hadronic W
    chi3_tt_nom = chi_square(events.wqq_cand.mass,73.9, 23.56) #hadronic W
    chi4_tt = chi_square(events.hbb_cand.dr,2.36, 0.81) #delta R between b-jets

    chi_sq_tt_soft = np.sqrt(chi1_tt_soft + chi2_tt_soft + chi3_tt_soft + chi4_tt)
    chi_sq_tt_nom_4j =  np.sqrt(chi1_tt_nom + chi2_tt_nom + chi3_tt_nom + chi4_tt)
    min_chi_sq_tt_soft = ak.argmin(chi_sq_tt_soft, axis=1, keepdims = True) #get index of the minimum chi square
    events['chi_sq_tt'] = ak.firsts(ak.where(events.selection.lowpt_4j2b, chi_sq_tt_soft[min_chi_sq_tt_soft], chi_sq_tt_nom_4j))

    # select jets with lower chi square across two signal regions
    qq_sel_index = ak.where(chi_sq_hadW_soft[min_chi_sq_hadW_soft] < chi_sq_hadWs_soft[min_chi_sq_hadWs_soft],
        min_chi_sq_hadW_soft, min_chi_sq_hadWs_soft)
    qq_sel_index = qq_sel_index[~ak.is_none(qq_sel_index)]

    events['qq_sel_mass'] = ak.where(
        chi_sq_hadW_soft < chi_sq_hadWs_soft ,
        ak.firsts(events.qq_soft[min_chi_sq_hadW_soft].mass),
        ak.firsts(events.qq_soft[min_chi_sq_hadWs_soft].mass))

    events['sr_boolean'] = ak.where(events.qq_sel_mass > 55.0, 1,
                                    ak.where(events.qq_sel_mass > 0, 0, 5))

    events['leading_e'] =  ak.fill_none(e_clean[e_clean.istight], np.nan)
    events['leading_mu'] = ak.fill_none(events.Muon[events.Muon.istight],np.nan)

    events['chi_sq_hadWs'] = ak.firsts(chi_sq_hadWs)
    events['chi_sq_hadW'] = ak.firsts(chi_sq_hadW)

    return events

# apply chi square cuts
def chi_sq_cut(events):
    #apply cuts on chi square calculation in leptonic W region and 4 jets hadronic W region
    events['passChiSqTT'] = ak.firsts(ak.where(events.sr_boolean == 1 & ~(events.selection.nominal_3j2b),
                                                ak.where((events.chi_sq_tt > 1.0), True, False), #hadronic W region cut
                                                True))
    events['passChiSqLepW'] = ak.firsts(ak.where(events.sr_boolean == 0,
                                                ak.where((events.chi_sq_hadWs < 2.0), True, False), #leptonic W region cut
                                                True))
    return events
