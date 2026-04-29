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
    
    phh_variable_nominal = H(([0.000000, 0.028806, 0.045231, 0.066259, 0.085993, 0.111045, 0.135618, 0.161293, 0.188042, 0.219752, 0.250790, 0.285573, 0.319903, 0.351917, 0.387853, 0.427756, 0.468420, 0.510314, 0.545485, 0.579444, 0.613291, 0.648580, 0.683775, 0.712018, 0.741373, 0.767620, 0.794539, 0.818005, 0.841648, 0.862076, 0.884600, 0.901133, 0.917262, 0.933671, 0.948090, 0.962910, 0.973818, 0.984453, 1.000000],
            ('phh', "Regressed P(Signal) bin #")))
    
    phh_variable_lowpt = H(([0.000000, 0.014905, 0.025288, 0.036131, 0.045277, 0.056941, 0.068035, 0.078536, 0.092887, 0.108523, 0.122283, 0.142631, 0.163056, 0.183468, 0.202435, 0.226643, 0.251694, 0.271075, 0.299633, 0.331193, 0.362592, 0.394022, 0.425558, 0.457136, 0.489279, 0.519983, 0.556079, 0.587894, 0.622649, 0.655849, 0.682687, 0.719662, 0.750957, 0.784467, 0.809527, 0.836404, 0.858502, 0.878870, 0.899157, 0.920223, 0.936293, 0.955259, 0.973305, 1.000000],
            ('phh', "Regressed P(Signal) bin #")))
    
    ptt     = H((50, 0, 1, ('ptt', "Regressed P(tt)")))
    poth     = H((50, 0, 1, ('poth', "P(minor backgrounds)")))
    hh_vs_tt = H((50, 0, 1, ('hh_vs_tt', "P(hh) | TTbar")))
    hh_vs_oth = H((50, 0, 1, ('hh_vs_oth', "P(hh) | Minor Backgrounds")))
    tt_vs_oth = H((50, 0, 1, ('tt_vs_oth', "P(tt) | Minor Backgrounds")))
    #WW = H((100, 0, 20, ('WW', "WW classifier score")))

