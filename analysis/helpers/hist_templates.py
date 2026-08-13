
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

class JetHists(Template):
    chEmEF      = H((30, -0.1, 1, ('chEmEF', 'charged EM energy fraction')))
    chHEF      = H((30, -0.1, 1, ('chHEF', 'charged hadron energy fraction')))
    neEmEF      = H((30, -0.1, 1, ('neEmEF', 'neutral EM energy fraction')))
    neHEF      = H((30, -0.1, 1, ('neHEF', 'neutral hadron energy fraction')))
    
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
    phh      = H((30, 0, 1, ('phh', "Regressed P(Signal)")))
    
    phh_variable_nominal = H(([0.000000, 0.236000, 0.596000, 0.784000, 0.900000, 0.928000, 0.968000, 0.984000, 1.000000],
            ('phh', "Regressed P(Signal) bin #")))
    
    phh_variable_lowpt = H(([0.000000, 0.240000, 0.456000, 0.680000, 0.812000, 0.868000, 0.944000, 0.960000, 1.000000],
            ('phh', "Regressed P(Signal) bin #")))

    phh_variable_3j2b = H(([0.000000, 0.460000, 0.632000, 0.736000, 0.848000, 0.888000, 0.904000, 0.920000, 0.944000, 1.000000],
                           ('phh', "Regressed P(Signal) bin #")))
                                  
    ptt     = H((50, 0, 1, ('ptt', "Regressed P(tt)")))
    poth     = H((50, 0, 1, ('poth', "P(minor backgrounds)")))
    hh_vs_tt = H((50, 0, 1, ('hh_vs_tt', "P(hh) | TTbar")))
    hh_vs_oth = H((50, 0, 1, ('hh_vs_oth', "P(hh) | Minor Backgrounds")))
    tt_vs_oth = H((50, 0, 1, ('tt_vs_oth', "P(tt) | Minor Backgrounds")))

