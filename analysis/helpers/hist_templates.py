
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
    phh      = H((20, 0, 1, ('phh', "Regressed P(Signal)")))
    
    phh_variable_nominal = H(([0.000000, 0.012987, 0.023376, 0.034598, 0.045523, 0.057103, 0.069355, 0.085169, 0.101577, 0.119004, 0.139776, 0.162401, 0.185083, 0.208877, 0.235493, 0.264666, 0.290680, 0.319656, 0.356300, 0.388705, 0.424377, 0.458871, 0.499058, 0.539747, 0.581018, 0.619069, 0.653519, 0.690753, 0.730645, 0.767691, 0.799627, 0.829254, 0.855008, 0.879654, 0.900615, 0.922676, 0.941503, 0.957525, 0.971450, 0.984629, 1.000000],
            ('phh', "Regressed P(Signal) bin #")))
    
    phh_variable_lowpt = H(([0.000000, 0.011795, 0.020768, 0.028407, 0.036951, 0.045443, 0.053582, 0.063584, 0.074772, 0.088669, 0.105676, 0.122538, 0.138372, 0.155180, 0.175359, 0.194778, 0.219542, 0.245249, 0.274882, 0.306396, 0.334771, 0.370617, 0.398498, 0.437623, 0.477450, 0.516357, 0.560993, 0.599858, 0.636510, 0.673523, 0.710284, 0.757001, 0.793527, 0.828811, 0.861580, 0.887833, 0.910936, 0.932395, 0.955963, 0.977855, 1.000000],
            ('phh', "Regressed P(Signal) bin #")))

    phh_variable_3j2b = H(([0.000000, 0.020970, 0.034992, 0.049793, 0.067026, 0.089443, 0.114940, 0.144620, 0.178053, 0.214778, 0.253238, 0.296123, 0.341060, 0.397935, 0.461438, 0.532112, 0.600571, 0.666646, 0.734094, 0.796536, 0.848188, 0.899523, 0.934712, 0.969096, 1.000000],
                           ('phh', "Regressed P(Signal) bin #")))
                                  
    ptt     = H((50, 0, 1, ('ptt', "Regressed P(tt)")))
    poth     = H((50, 0, 1, ('poth', "P(minor backgrounds)")))
    hh_vs_tt = H((50, 0, 1, ('hh_vs_tt', "P(hh) | TTbar")))
    hh_vs_oth = H((50, 0, 1, ('hh_vs_oth', "P(hh) | Minor Backgrounds")))
    tt_vs_oth = H((50, 0, 1, ('tt_vs_oth', "P(tt) | Minor Backgrounds")))
    #WW = H((100, 0, 20, ('WW', "WW classifier score")))

