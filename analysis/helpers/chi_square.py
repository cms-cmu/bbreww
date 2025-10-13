import awkward as ak
import numpy as np
from bbreww.analysis.helpers.common import chi_square, met_reconstr, distance

def chi_sq(events):

    #
    # Nominal hadronic W* chi square calculation
    #
    chi2_hadWs = ak.zip( { "Hbb_mass" : chi_square(events.Hbb_cand.mass, 113.22, 30.74),   # H -> bb
                           "Hww_mass" : chi_square(events.Hww_cand.mass, 161.15, 34.23), # H -> lvqq in nominal region
                           "Wqq_mass" : chi_square(events.Wqq_cand.mass,  39.13, 10.02), # W* -> qq in nominal region
                           "Hbb_dr"   : chi_square(events.Hbb_cand.dr,     1.90,  0.70)  #delta R between b-jets
                          })
    # 4 jet region chi square
    chi2_hadWs["tot_4j"] = np.sqrt(chi2_hadWs.Hbb_mass**2 + chi2_hadWs.Hww_mass**2 + chi2_hadWs.Wqq_mass**2 + chi2_hadWs.Hbb_dr**2)

    # 3 jet region chi square (can't use di-jet variables for 3 jet region)
    chi2_hadWs["tot_3j"] = np.sqrt(chi2_hadWs.Hbb_mass**2 + chi2_hadWs.Hbb_dr**2)

    events['chi2_hadWs'] = chi2_hadWs

    #
    # Soft hadronic W* chi square calculation
    #
    chi2_hadWs_soft = ak.zip( { "Hbb_mass" : chi2_hadWs.Hbb_mass,   # H -> bb
                                "Hww_mass" : chi_square(events.Hww_cand_soft.mass, 161.15, 34.23), # H -> lvqq in low pt region
                                "Wqq_mass" : chi_square(events.qq_soft.mass,        39.13, 10.02), # W* -> qq in low pt region
                                "Hbb_dr"   : chi2_hadWs.Hbb_dr  #delta
                                })

    chi2_hadWs_soft["tot_4j"] = np.sqrt(chi2_hadWs_soft.Hbb_mass**2 + chi2_hadWs_soft.Hww_mass**2 + chi2_hadWs_soft.Wqq_mass**2 + chi2_hadWs_soft.Hbb_dr**2)
    min_chi_sq_hadWs_soft = ak.argmin(chi2_hadWs_soft.tot_4j, axis=1, keepdims = True) #index of the minimum chi square non-bjet pairs
    chi2_hadWs_soft = chi2_hadWs_soft[min_chi_sq_hadWs_soft]

    events["chi2_hadWs_soft"] = chi2_hadWs_soft

    chi_sq_hadWs_soft_flat = ak.flatten(chi2_hadWs_soft.tot_4j)
    chi_sq_hadWs = ak.where(events.nominal_4j2b, events.chi2_hadWs.tot_4j , events.chi2_hadWs.tot_3j)
    chi_sq_hadWs = ak.where(events.lowpt_4j2b,   chi_sq_hadWs_soft_flat, chi_sq_hadWs )

    #
    #  Nominal hadronic W chi square calculation
    #
    chi2_hadW = ak.zip( { "Hbb_mass" : chi_square(events.Hbb_cand.mass, 111.13, 23.63), # H -> bb
                          "Wln_mT"   : chi_square(events.Wlnu_cand.mT,   58.87, 37.35), #transverse mass
                          "Wqq_mass" : chi_square(events.Wqq_cand.mass,  76.84, 10.98), #hadronic W
                          "Hbb_dr"   : chi_square(events.Hbb_cand.dr,     1.69,  0.60), #delta R between b-jets
                          })

    # 4 jet region chi square
    chi2_hadW["tot_4j"] = np.sqrt(chi2_hadW.Hbb_mass**2 + chi2_hadW.Wln_mT**2 + chi2_hadW.Wqq_mass**2 + chi2_hadW.Hbb_dr**2)

    # 3 jet region chi square
    chi2_hadW["tot_3j"] = np.sqrt(chi2_hadW.Hbb_mass**2 + chi2_hadW.Wln_mT**2 + chi2_hadW.Hbb_dr**2) # don't use dijet variables for 3j region

    events['chi2_hadW'] = chi2_hadW

    #
    # Soft hadronic W chi square calculation
    #
    chi2_hadW_soft = ak.zip( { "Hbb_mass" : chi2_hadW.Hbb_mass,   # H -> bb
                               "Wln_mT"   : chi2_hadW.Wln_mT, #transverse mass
                               "Wqq_mass" : chi_square(events.qq_soft.mass,   76.84, 10.98), #hadronic W
                               "Hbb_dr"   : chi2_hadW.Hbb_dr  #delta
                                })


    chi2_hadW_soft["tot_4j"]  =  np.sqrt(chi2_hadW_soft.Hbb_mass**2 + chi2_hadW_soft.Wln_mT**2 + chi2_hadW_soft.Wqq_mass**2 + chi2_hadW_soft.Hbb_dr**2)
    # chi_sq_hadW_soft   = np.sqrt(chi2_hadW.Hbb_mass**2 + chi2_hadW.Wln_mT**2 + chi3_hadW_soft + chi2_hadW.Hbb_dr**2)
    min_chi_sq_hadW_soft= ak.argmin(chi2_hadW_soft.tot_4j, axis=1, keepdims = True) #index of the minimum chi square non-bjet pair
    chi2_hadW_soft = chi2_hadW_soft[min_chi_sq_hadW_soft]

    events["chi2_hadW_soft"] = chi2_hadW_soft

    chi_sq_hadW_soft_flat = ak.flatten(chi2_hadW_soft.tot_4j)
    chi_sq_hadW = ak.where(events.nominal_4j2b, events.chi2_hadW.tot_4j , events.chi2_hadW.tot_3j)
    chi_sq_hadW = ak.where(events.lowpt_4j2b,   chi_sq_hadW_soft_flat, chi_sq_hadW )


    #
    # ttbar reconstruction
    #

    chi2_tt_bjet_dr = chi_square(events.Hbb_cand.dr,     2.30,     0.81)  #delta R between b-jets

    chi2_tt = ak.zip( {"lepTop_mass" : chi_square(events.tt_sel.lepTop.mass,    165.55,    35.49), #leptonic top
                       "hadTop_mass" : chi_square(events.tt_sel.hadTop.mass,    171.55,    44.95), #hadronic top
                       "Wqq_mass"    : chi_square(events.Wqq_cand.mass,          73.9,     23.56), #hadronic W
                       "Hbb_dr"      : chi2_tt_bjet_dr
                        })

    chi2_tt["tot_4j"] = np.sqrt(chi2_tt.lepTop_mass**2 + chi2_tt.hadTop_mass**2 + chi2_tt.Wqq_mass**2 + chi2_tt.Hbb_dr**2)

    events['chi2_tt'] = chi2_tt

    # Soft

    mbqq1_soft = ak.pad_none((events.b_cands[:,0] + events.qq_soft).mass,3,axis=1) #hadronic candidate 1
    mbqq2_soft = ak.pad_none((events.b_cands[:,1] + events.qq_soft).mass,3,axis=1) #hadronic candidate 2

    mlvb1 = (events.Wlnu_cand + events.b_cands[:,0]).mass
    mlvb2 = (events.Wlnu_cand + events.b_cands[:,1]).mass

    tt1_soft = ak.cartesian({"t1":mlvb1, "t2":mbqq2_soft},axis=1)
    tt2_soft = ak.cartesian({"t1":mlvb2, "t2":mbqq1_soft},axis=1)

    b_sel_soft = abs(distance(tt1_soft.t1, tt1_soft.t2, 172.5, 172.5)) < abs(distance(tt2_soft.t1, tt2_soft.t2, 172.5, 172.5)) #pick pair closest to ttbar mass

    #final ttbar candidates
    tt_soft = ak.where(b_sel_soft, tt1_soft , tt2_soft)

    chi1_tt_soft = chi_square(tt_soft.t1,           165.55,    35.49, power=2) #leptonic top
    chi2_tt_soft = chi_square(tt_soft.t2,           171.55,    44.95, power=2) #hadronic top
    chi3_tt_soft = chi_square(events.qq_soft.mass,   73.9,     23.56, power=2) #hadronic W
    chi4_tt      = chi_square(events.Hbb_cand.dr,     2.30,     0.81, power=2) #hadronic W

    chi_sq_tt_soft = np.sqrt(chi1_tt_soft + chi2_tt_soft + chi3_tt_soft + chi2_tt_bjet_dr**2 )

    min_chi_sq_tt_soft = ak.argmin(chi_sq_tt_soft, axis=1, keepdims = True) #get index of the minimum chi square
    events['chi_sq_tt'] = ak.where(events.lowpt_4j2b, ak.firsts(chi_sq_tt_soft[min_chi_sq_tt_soft]), chi2_tt.tot_4j)

    # select jets with lower chi square across two signal regions
    qq_sel_index = ak.where(chi2_hadW_soft.tot_4j <= chi2_hadWs_soft.tot_4j, min_chi_sq_hadW_soft, min_chi_sq_hadWs_soft)
    events['qq_sel_mass'] = events.qq_soft[qq_sel_index].mass
    events['sr_boolean'] = ak.where(events.qq_sel_mass > 55.0, 1, 0)

    events['chi_sq_hadWs'] = chi_sq_hadWs
    events['chi_sq_hadW']  = chi_sq_hadW

    return events

# apply chi square cuts
def chi_sq_cut(events):
    #apply cuts on chi square calculation in leptonic W region and 4 jets hadronic W region
    events['passChiSqTT'] = ak.firsts(events.sr_boolean == 0) | (events.chi_sq_tt > 1.0) # hadronic W region cut
    events['passChiSqLepW'] = ak.firsts(events.sr_boolean == 1) | (events.chi_sq_hadWs < 2.0) #leptonic W region cut
    #events['passChiSqHadW'] = ak.where(events.nominal_3j2b, ak.firsts(events.chi_sq_hadW < 1.2), False)

    return events
