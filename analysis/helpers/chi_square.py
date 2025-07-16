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

    mlvqq_hadWs = {
        'esr'  : mevqq,
        'msr'  : mmuvqq
    }
        
    mlvqq_hadWs = ak.where(l_mu & l_e,
                        ak.where(muge, mlvqq_hadWs['msr'], mlvqq_hadWs['esr']),
                        ak.where(l_mu, mlvqq_hadWs['msr'], mlvqq_hadWs['esr'])
                        ) #select leading lepton combination

    #individual chi squares for hadronic W* signal selection
    chi1_hadWs, mean1_hadWs, std1_hadWs = chi_square(events.mbb,116.02, 45.04) # H -> bb            
    chi2_hadWs, mean2_hadWs, std2_hadWs = chi_square(mlvqq_hadWs, 173.59, 48.67) # H -> lvqq
    chi3_hadWs, mean3_hadWs, std3_hadWs = chi_square(events.qq.mass,41.77, 14.92) #hadronic W*    

    #total chi square
    chi_sq_hadWs = np.sqrt(chi1_hadWs + chi2_hadWs + chi3_hadWs)
    min_chi_sq_hadWs = ak.argmin(chi_sq_hadWs, axis=1, keepdims = True) #index of the minimum chi square non-bjet pair
    events['mlvqq_hadWs'] = mlvqq_hadWs[min_chi_sq_hadWs]
    events['chi_sq_hadWs'] = chi_sq_hadWs[min_chi_sq_hadWs]

    #transverse mass
    mT = {
        'esr'  : np.sqrt(2*leading_e.pt*met.pt*(1-np.cos(met.delta_phi(leading_e)))),
        'msr'  : np.sqrt(2*leading_mu.pt*met.pt*(1-np.cos(met.delta_phi(leading_mu))))
    }


    mT_leading_lep = ak.where(l_mu & l_e,
                        ak.where(muge, mT['msr'], mT['esr']),
                        ak.where(l_mu, mT['msr'], mT['esr'])
                        )


    #individual chi squares for hadronic W signal selection
    chi1_hadW, mean1_hadW, std1_hadW = chi_square(events.mbb,115.33, 46.29) # H -> bb
    chi2_hadW, mean2_hadW, std2_hadW = chi_square(mT_leading_lep, 58.87, 37.35) #transverse mass             
    chi3_hadW, mean3_hadW, std3_hadW = chi_square(events.qq.mass,66.89, 10.98) #hadronic W

    chi_sq_hadW = np.sqrt(chi1_hadW + chi2_hadW + chi3_hadW)
    min_chi_sq_hadW= ak.argmin(chi_sq_hadW, axis=1, keepdims = True) #index of the minimum chi square non-bjet pair
    events['chi_sq_hadW'] = chi_sq_hadW[min_chi_sq_hadW]

    ## ttbar reconstruction

    #leptonic top with electrons
    mevb1 = (leading_e + v_e + ak.pad_none(events.j_bcand,2,axis=1)[:,0]).mass
    mevb2 = (leading_e + v_e + ak.pad_none(events.j_bcand,2,axis=1)[:,1]).mass

    #leptonic top with muons
    mmvb1 = (leading_mu + v_mu + ak.pad_none(events.j_bcand,2,axis=1)[:,0]).mass
    mmvb2 = (leading_mu + v_mu + ak.pad_none(events.j_bcand,2,axis=1)[:,1]).mass

    mlvb1 = ak.where(l_mu & l_e,
                        ak.where(muge, mmvb1, mevb1),
                        ak.where(l_mu, mmvb1, mevb1)
                        ) #leptonic candidate 1

    mlvb2 = ak.where(l_mu & l_e,
                        ak.where(muge, mmvb2, mevb2),
                        ak.where(l_mu, mmvb2, mevb2)
                        ) #leptonic candidate 2  

    mbqq1 = ak.pad_none((ak.pad_none(events.j_bcand,2,axis=1)[:,0] + ak.mask(events.qq, events.j_tt_mask)).mass,3,axis=1) #hadronic candidate 1
    mbqq2 = ak.pad_none((ak.pad_none(events.j_bcand,2,axis=1)[:,1] + ak.mask(events.qq, events.j_tt_mask)).mass,3,axis=1) #hadronic candidate 2

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

    qq_tt = ak.mask(events.qq, events.j_tt_mask) #select leading pT jet > 20 GeV for ttbar
    chi1_tt, mean1_tt, std1_tt = chi_square(tt.t1,194.93 , 47.59 ) #leptonic top
    chi2_tt, mean2_tt, std2_tt = chi_square(tt.t2, 171.55, 44.95 ) #hadronic top
    chi3_tt, mean3_tt, std3_tt = chi_square(qq_tt.mass,73.9, 23.56) #hadronic W

    chi_sq_tt = np.sqrt(chi1_tt + chi2_tt + chi3_tt)
    min_chi_sq_tt = ak.argmin(chi_sq_tt, axis=1, keepdims = True) #get index of the minimum chi square 
    events['chi_sq_tt'] = chi_sq_tt[min_chi_sq_tt]

    jj_sel_gen_mass_tt =  ak.fill_none(ak.firsts(events.qq_gen_mass[min_chi_sq_tt]),-1) #gen mass of the di jet pair with minimum chi square
    events['sr_boolean'] = ak.where(jj_sel_gen_mass_tt > 55.0, 1,
                                        ak.where(jj_sel_gen_mass_tt > 0, 0, 5))
    
    return events