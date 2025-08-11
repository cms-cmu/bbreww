import awkward as ak
import numpy as np
from bbww.analysis.helpers.common import nu_pz, chi_square, met_reconstr

def chi_sq(events):
    e_clean = e_clean = events.Electron[events.Electron.isclean]
    leading_mu = ak.firsts(events.Muon[events.Muon.istight])
    leading_e = ak.firsts(e_clean[e_clean.istight])
    met = events.MET

    # hadronic W* signal reconstruction
    v_mu, v_e = met_reconstr(events, leading_e, leading_mu)
    # H -> lvqq with electrons and muons
    mevqq = (leading_e + v_e + events.qq ).mass
    mmuvqq = (leading_mu + v_mu + events.qq).mass

    l_mu = ~ak.is_none(leading_mu.pt)
    l_e = ~ak.is_none(leading_e.pt)
    muge = leading_mu.pt > leading_e.pt

    # 0 = electron, 1 = muon, -1 = neither
    events['lepton_choice'] = ak.where(l_mu & l_e,
                            ak.where(muge, 1, 0),  # both present: 1 if mu>e, else 0
                            ak.where(l_mu, 1,      # only mu: 1
                                    ak.where(l_e, 0, -1)))  # only e: 0, neither: -1


    ### TEMP: studying reconstructed MET pz resolution
    events['rec_met'] = ak.where(events.lepton_choice == 1, v_mu, v_e) #select leading lepton combination
    events['rec_W'] = ak.where(events.lepton_choice == 1, (leading_mu + v_mu).mass, (leading_e+v_e).mass)
    ####################

    mlvqq_hadWs = ak.where(events.lepton_choice==1, mmuvqq, mevqq) #select leading lepton combination
    events['bb_dr'] = events.j_bcand[:,0].delta_r(events.j_bcand[:,1])

    #individual chi squares for hadronic W* signal selection
    chi1_hadWs = chi_square(events.mbb,116.02, 45.04) # H -> bb            
    chi2_hadWs = chi_square(mlvqq_hadWs, 161.15, 34.23) # H -> lvqq
    chi3_hadWs = chi_square(events.qq.mass,39.13, 10.02) #hadronic W*   
    chi4_hadWs = chi_square(events.bb_dr,1.79, 0.79) #delta R between b-jets 

    #total chi square
    chi_sq_hadWs = np.sqrt(chi1_hadWs + chi2_hadWs + chi3_hadWs + chi4_hadWs)
    min_chi_sq_hadWs = ak.argmin(chi_sq_hadWs, axis=1, keepdims = True) #index of the minimum chi square non-bjet pair
    #events['mlvqq_hadWs'] = ak.firsts(mlvqq_hadWs[min_chi_sq_hadWs]) ## TEMP : comment out for now
    chi_sq_hadWs = chi_sq_hadWs[min_chi_sq_hadWs]

    #transverse mass
    mT = {
        'esr'  : np.sqrt(2*leading_e.pt*met.pt*(1-np.cos(met.delta_phi(leading_e)))),
        'msr'  : np.sqrt(2*leading_mu.pt*met.pt*(1-np.cos(met.delta_phi(leading_mu))))
    }

    events['mT_leading_lep'] = ak.where(events.lepton_choice==1, mT['msr'], mT['esr'])

    #individual chi squares for hadronic W signal selection
    chi1_hadW = chi_square(events.mbb,112.46, 46.61) # H -> bb
    chi2_hadW = chi_square(events.mT_leading_lep, 58.87, 37.35) #transverse mass             
    chi3_hadW = chi_square(events.qq.mass,76.84, 10.98) #hadronic W
    chi4_hadW = chi_square(events.bb_dr,1.76, 0.81) #delta R between b-jets

    chi_sq_hadW = np.sqrt(chi1_hadW + chi2_hadW + chi3_hadW + chi4_hadW)
    min_chi_sq_hadW= ak.argmin(chi_sq_hadW, axis=1, keepdims = True) #index of the minimum chi square non-bjet pair
    chi_sq_hadW = chi_sq_hadW[min_chi_sq_hadW]

    ## ttbar reconstruction

    #leptonic top with electrons
    mevb1 = (leading_e + v_e + ak.pad_none(events.j_bcand,2,axis=1)[:,0]).mass
    mevb2 = (leading_e + v_e + ak.pad_none(events.j_bcand,2,axis=1)[:,1]).mass

    #leptonic top with muons
    mmvb1 = (leading_mu + v_mu + ak.pad_none(events.j_bcand,2,axis=1)[:,0]).mass
    mmvb2 = (leading_mu + v_mu + ak.pad_none(events.j_bcand,2,axis=1)[:,1]).mass

    mlvb1 = ak.where(events.lepton_choice==1, mmvb1, mevb1) #leptonic candidate 1
    mlvb2 = ak.where(events.lepton_choice==1, mmvb2, mevb2) #leptonic candidate 2  

    mbqq1 = ak.pad_none((ak.pad_none(events.j_bcand,2,axis=1)[:,0] + events.qq).mass,3,axis=1) #hadronic candidate 1
    mbqq2 = ak.pad_none((ak.pad_none(events.j_bcand,2,axis=1)[:,1] + events.qq).mass,3,axis=1) #hadronic candidate 2

    def distance(x1,y1,x2,y2):
        return np.sqrt((x2-x1)**2+(y2-y1)**2)

    #ttbar candidates
    tt1 = ak.cartesian({"t1":mlvb1,"t2":mbqq2},axis=1)
    tt2 = ak.cartesian({"t1":mlvb2,"t2":mbqq1},axis=1)
    b_sel = abs(distance(tt1.t1,tt1.t2,172.5,172.5)) <  abs(distance(tt2.t1,tt2.t2,172.5,172.5)) #pick pair closest to ttbar mass

    #conditions to work around None values
    c1 = ~ak.is_none(distance(tt1.t1, tt1.t2,172.5,172.5))
    c2 = ~ak.is_none(distance(tt2.t1, tt2.t2,172.5,172.5))

    #final ttbar candidates
    tt = ak.pad_none(ak.where( c1 & c2, ak.where(b_sel, tt1 , tt2), ak.where(c1, tt1, tt2)),3,axis=1)

    chi1_tt = chi_square(tt.t1,165.55 , 35.49 ) #leptonic top
    chi2_tt = chi_square(tt.t2, 171.55, 44.95 ) #hadronic top
    chi3_tt = chi_square(events.qq.mass,73.9, 23.56) #hadronic W
    chi4_tt = chi_square(events.bb_dr,2.36, 0.81) #delta R between b-jets

    chi_sq_tt = np.sqrt(chi1_tt + chi2_tt + chi3_tt + chi4_tt)
    min_chi_sq_tt = ak.argmin(chi_sq_tt, axis=1, keepdims = True) #get index of the minimum chi square 
    events['chi_sq_tt'] = ak.firsts(chi_sq_tt[min_chi_sq_tt])

    # select jets with lower chi square across two signal regions
    events['qq_sel_index'] = ak.where(
        ak.fill_none(chi_sq_hadW,100.0) < ak.fill_none(chi_sq_hadWs,100.0) , 
        min_chi_sq_hadW, ak.where(~ak.is_none(chi_sq_hadWs),min_chi_sq_hadWs, -1))


    events['qq_sel_mass'] = ak.where(
        ak.fill_none(chi_sq_hadW,100.0) < ak.fill_none(chi_sq_hadWs,100.0) , 
        ak.firsts(events.qq[min_chi_sq_hadW].mass), 
        ak.where(~ak.is_none(chi_sq_hadWs),ak.firsts(events.qq[min_chi_sq_hadWs].mass), -1))

    events['sr_boolean'] = ak.where(events.qq_sel_mass > 55.0, 1, 
                                    ak.where(events.qq_sel_mass > 0, 0, 5))
    
    events['leading_lep'] = ak.where(events.lepton_choice == 0, 
                                     events.Electron[events.Electron.istight], 
                                     events.Muon[events.Muon.istight]) 
    
    events['e_region'] = events.Electron[(events.Electron.istight) & (events.lepton_choice ==0)]
    events['mu_region'] = events.Muon[(events.Muon.istight) & (events.lepton_choice ==1)]
    events['mlvqq_hadWs'] = ak.firsts(mlvqq_hadWs[min_chi_sq_hadWs]) ##( TEMP:  change to plot this variable before running chi square
    
    events['ak4_sel1'] = events.j_nonbcand[events.dijet_combs.j1[events.qq_sel_index]]
    events['ak4_sel2'] = events.j_nonbcand[events.dijet_combs.j2[events.qq_sel_index]]
    events['chi_sq_hadWs'] = ak.firsts(chi_sq_hadWs)
    events['chi_sq_hadW'] = ak.firsts(chi_sq_hadW)

    return events

# apply chi square cuts
def chi_sq_cut(events):
    events['passChiSqHadW'] = ak.fill_none(ak.where(events.chi_sq_hadW < 2, True, False ),False) 
    events['passChiSqLepW'] = ak.fill_none(ak.where(events.chi_sq_hadWs < 1.6, True, False ),False) 
    return events