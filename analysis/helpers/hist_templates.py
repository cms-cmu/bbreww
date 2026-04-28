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
    phh      = H((20, 0, 1, ('phh', "Regressed P(Signal)")))
    
    phh_variable_nominal = H(([0.000000, 0.015118, 0.025579, 0.035865, 0.047568, 0.059949, 0.074508, 0.088799, 0.108748, 0.126804, 0.147484, 0.170666, 0.197836, 0.224978, 0.249622, 0.283753, 0.320594, 0.352732, 0.393251, 0.427180, 0.457017, 0.497098, 0.532316, 0.571828, 0.612726, 0.646702, 0.678011, 0.714927, 0.746695, 0.776337, 0.804921, 0.830473, 0.855672, 0.876266, 0.897546, 0.920978, 0.938179, 0.957493, 0.970888, 0.983099, 1.000000],
            ('phh', "Regressed P(Signal) bin #")))
    
    phh_variable_lowpt = H(([0.000000, 0.011356, 0.019027, 0.025580, 0.032693, 0.040552, 0.048454, 0.056070, 0.067720, 0.079139, 0.092498, 0.108201, 0.125204, 0.142825, 0.159998, 0.177086, 0.203283, 0.229156, 0.262785, 0.289982, 0.319949, 0.348740, 0.378441, 0.412525, 0.452870, 0.491348, 0.532600, 0.562163, 0.608546, 0.647599, 0.696727, 0.732454, 0.772075, 0.807127, 0.835676, 0.866473, 0.891915, 0.913833, 0.939417, 0.968514, 1.000000],
            ('phh', "Regressed P(Signal) bin #")))
    
    ptt     = H((50, 0, 1, ('ptt', "Regressed P(tt)")))
    poth     = H((50, 0, 1, ('poth', "P(minor backgrounds)")))
    hh_vs_tt = H((50, 0, 1, ('hh_vs_tt', "P(hh) | TTbar")))
    hh_vs_oth = H((50, 0, 1, ('hh_vs_oth', "P(hh) | Minor Backgrounds")))
    tt_vs_oth = H((50, 0, 1, ('tt_vs_oth', "P(tt) | Minor Backgrounds")))
    #WW = H((100, 0, 20, ('WW', "WW classifier score")))

