
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
    
    phh_variable_nominal = H(([0.000000, 0.012249, 0.020829, 0.031057, 0.041468, 0.050794, 0.061684, 0.074252, 0.087996, 0.103224, 0.119010, 0.137672, 0.157572, 0.177938, 0.199333, 0.223341, 0.247925, 0.273049, 0.299128, 0.325075, 0.358213, 0.388765, 0.420660, 0.454539, 0.485262, 0.526539, 0.560908, 0.596915, 0.631327, 0.662718, 0.697344, 0.732927, 0.767149, 0.795762, 0.823332, 0.848338, 0.868353, 0.890970, 0.911782, 0.930123, 0.946373, 0.961100, 0.974572, 0.985489, 1.000000],
               ('phh', "Regressed P(Signal) bin #")))

    phh_variable_lowpt = H(([0.000000, 0.010374, 0.018817, 0.024180, 0.029789, 0.037552, 0.044812, 0.050853, 0.058141, 0.067567, 0.076204, 0.088469, 0.101080, 0.114524, 0.129642, 0.141797, 0.156082, 0.172704, 0.187533, 0.203544, 0.226047, 0.247253, 0.270580, 0.297005, 0.320809, 0.346444, 0.376262, 0.396731, 0.427521, 0.459089, 0.490930, 0.527809, 0.562134, 0.594770, 0.623950, 0.656728, 0.681740, 0.714452, 0.754858, 0.784361, 0.813551, 0.841672, 0.866882, 0.889613, 0.907480, 0.926323, 0.946777, 0.960938, 0.980391, 1.000000],
               ('phh', "Regressed P(Signal) bin #")))

    phh_variable_3j2b = H(([0.000000, 0.015498, 0.023469, 0.032503, 0.041552, 0.054287, 0.068127, 0.085279, 0.102909, 0.122282, 0.149347, 0.175755, 0.208884, 0.245607, 0.289195, 0.340326, 0.396944, 0.460296, 0.533628, 0.597239, 0.666946, 0.730038, 0.795618, 0.855271, 0.909280, 0.955333, 1.000000],
                ('phh', "Regressed P(Signal) bin #")))

                                  
    ptt     = H((50, 0, 1, ('ptt', "Regressed P(tt)")))
    poth     = H((50, 0, 1, ('poth', "P(minor backgrounds)")))
    hh_vs_tt = H((50, 0, 1, ('hh_vs_tt', "P(hh) | TTbar")))
    hh_vs_oth = H((50, 0, 1, ('hh_vs_oth', "P(hh) | Minor Backgrounds")))
    tt_vs_oth = H((50, 0, 1, ('tt_vs_oth', "P(tt) | Minor Backgrounds")))
    #WW = H((100, 0, 20, ('WW', "WW classifier score")))

