from src.hist_tools import H, Template
from src.hist_tools.object import LorentzVector

class Chi2Hists(Template):
    tot_4j      = H((50, -0.1, 10, ('tot_4j', 'tot chi square 4j2b')))
    tot_3j      = H((50, -0.1, 10, ('tot_3j', 'tot chi square 3j2b')))
    Hbb_mass    = H((50, -0.1, 10, ('Hbb_mass',  'chi square for Hbb_mass')))
    Hww_mass    = H((50, -0.1, 10, ('Hww_mass',  'chi square for Hww_mass')))
    Wqq_mass    = H((50, -0.1, 10, ('Wqq_mass',  'chi square for Wqq_mass')))
    Wln_mT      = H((50, -0.1, 10, ('Wln_mT',    'chi square for Wln_mT')))
    Hbb_dr      = H((50, -0.1, 10, ('Hbb_dr',    'chi square for Hbb_dr')))
    lepTop_mass = H((50, -0.1, 10, ('lepTop_mass','chi square for lep top mass')))
    hadTop_mass = H((50, -0.1, 10, ('hadTop_mass','chi square for had top mass')))

class TTbarHists(Template):
    p      = LorentzVector.plot_pair(("...", R"$t\bar{t}$"), "p",  skip=["n","lead","subl","st"], bins={"mass": (100, 0, 1200)}, )
    lepTop = LorentzVector.plot_pair(("...", R"lepTop"), "lepTop", skip=["n","lead","subl","st"], bins={"mass": (100, 0, 400)}, )
    hadTop = LorentzVector.plot_pair(("...", R"hadTop"), "hadTop", skip=["n","lead","subl","st"], bins={"mass": (100, 0, 400)}, )

class regressionHists(Template):
    reg_px = H((50, -150, 150, ('nu_px', R"Regressed MET $p_x$")))
    reg_py = H((50, -150, 150, ('nu_py', R"Regressed MET $p_y$"))) 
    reg_pz = H((50, -150, 150, ('nu_pz', R"Regressed MET $p_z$ [GeV]")))
    sigma_pz_on = H((50, 0, 100, ('sigma_pz_on', R"on shell uncertainty")))
    sigma_pz_off = H((50, 0, 100, ('sigma_pz_off', R"off shell uncertainty")))

    p_onshell = H((10, 0, 1.1, ('p_onshell', "on shell probability")))
    
class SvBHists(Template):
    phh      = H((50, 0, 1, ('phh', "Regressed P(Signal)")))
    
    phh_variable_nominal = H(([2.84092705e-04, 2.98679943e-02, 5.16027159e-02, 7.18979611e-02,
                9.19469550e-02, 1.16690672e-01, 1.40041789e-01, 1.65460721e-01,
                1.90193765e-01, 2.15181096e-01, 2.42934471e-01, 2.67309617e-01,
                2.94463028e-01, 3.22157875e-01, 3.50515915e-01, 3.78820745e-01,
                4.06102498e-01, 4.32527262e-01, 4.58217502e-01, 4.83336654e-01,
                5.12493183e-01, 5.38735968e-01, 5.65131502e-01, 5.93076841e-01,
                6.17239177e-01, 6.41012574e-01, 6.62211168e-01, 6.82939761e-01,
                7.04404577e-01, 7.25850819e-01, 7.47014721e-01, 7.66755427e-01,
                7.86144962e-01, 8.03882648e-01, 8.20590594e-01, 8.38531715e-01,
                8.54030535e-01, 8.66325794e-01, 8.80648195e-01, 8.94177027e-01,
                9.06234559e-01, 9.17626855e-01, 9.27120019e-01, 9.35542504e-01,
                9.44127313e-01, 9.53574098e-01, 9.61449202e-01, 9.69035538e-01,
                9.77348578e-01, 9.85852016e-01, 9.98567104e-01],
            ('phh', "Regressed P(Signal) bin #")))
    
    phh_variable_lowpt = H(([3.34139768e-04, 2.55530765e-02, 4.83194350e-02, 6.68962127e-02,
                8.34158867e-02, 1.01090330e-01, 1.19578418e-01, 1.39260355e-01,
                1.59699197e-01, 1.80954290e-01, 2.01895898e-01, 2.22877205e-01,
                2.46693094e-01, 2.68002399e-01, 2.88320161e-01, 3.11062260e-01,
                3.34952262e-01, 3.60665545e-01, 3.81871782e-01, 4.11358301e-01,
                4.36647992e-01, 4.64942872e-01, 4.88358375e-01, 5.11337955e-01,
                5.33927419e-01, 5.57941654e-01, 5.82443227e-01, 6.10809675e-01,
                6.36339113e-01, 6.61362870e-01, 6.84831151e-01, 7.05213334e-01,
                7.26350972e-01, 7.49133295e-01, 7.67536638e-01, 7.89900031e-01,
                8.07856521e-01, 8.23239504e-01, 8.39564421e-01, 8.54804909e-01,
                8.69854110e-01, 8.85630450e-01, 9.00315340e-01, 9.12526443e-01,
                9.22520990e-01, 9.33639441e-01, 9.44395603e-01, 9.55872365e-01,
                9.65906348e-01, 9.77993890e-01, 9.96997237e-01],
            ('phh', "Regressed P(Signal) bin #")))
    
    ptt     = H((50, 0, 1, ('ptt', "Regressed P(tt)")))
    poth     = H((50, 0, 1, ('poth', "P(minor backgrounds)")))
    hh_vs_tt = H((50, 0, 1, ('hh_vs_tt', "P(hh) | TTbar")))
    hh_vs_oth = H((50, 0, 1, ('hh_vs_oth', "P(hh) | Minor Backgrounds")))
    tt_vs_oth = H((50, 0, 1, ('tt_vs_oth', "P(tt) | Minor Backgrounds")))
    #WW = H((100, 0, 20, ('WW', "WW classifier score")))

