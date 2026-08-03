
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
    
    phh_variable_nominal = H(([0.000000, 0.010739, 0.019511, 0.027474, 0.036143, 0.045502, 0.053547, 0.062518, 0.073783, 0.086441, 0.100942, 0.113705, 0.130105, 0.148809, 0.167569, 0.187389, 0.205826, 0.226714, 0.249788, 0.272803, 0.299040, 0.326414, 0.357124, 0.388597, 0.420917, 0.454213, 0.488288, 0.522594, 0.553884, 0.590739, 0.624224, 0.657734, 0.689970, 0.719688, 0.753752, 0.781984, 0.808506, 0.834749, 0.856509, 0.877911, 0.897315, 0.915683, 0.932493, 0.950130, 0.964949, 0.976870, 0.987069, 1.000000],
            ('phh', "Regressed P(Signal) bin #")))
    
    phh_variable_lowpt = H(([0.000000, 0.009896, 0.016455, 0.021468, 0.027140, 0.033540, 0.040083, 0.047281, 0.054754, 0.062322, 0.071149, 0.081699, 0.092290, 0.101865, 0.112235, 0.124673, 0.134822, 0.151259, 0.163837, 0.177734, 0.196130, 0.215505, 0.234917, 0.254861, 0.278192, 0.303797, 0.332879, 0.358769, 0.387854, 0.417824, 0.446696, 0.476567, 0.510414, 0.545622, 0.578927, 0.605171, 0.640114, 0.674326, 0.712038, 0.742302, 0.776153, 0.807328, 0.829853, 0.852909, 0.882838, 0.904168, 0.924889, 0.945927, 0.961698, 0.980718, 1.000000],
            ('phh', "Regressed P(Signal) bin #")))

    phh_variable_3j2b = H(([0.000000, 0.021943, 0.037813, 0.054027, 0.074112, 0.102263, 0.128707, 0.167344, 0.204439, 0.245433, 0.292952, 0.341060, 0.401908, 0.475643, 0.549103, 0.631600, 0.695617, 0.769567, 0.830903, 0.882506, 0.929049, 0.966786, 1.000000],
                           ('phh', "Regressed P(Signal) bin #")))
                                  
    ptt     = H((50, 0, 1, ('ptt', "Regressed P(tt)")))
    poth     = H((50, 0, 1, ('poth', "P(minor backgrounds)")))
    hh_vs_tt = H((50, 0, 1, ('hh_vs_tt', "P(hh) | TTbar")))
    hh_vs_oth = H((50, 0, 1, ('hh_vs_oth', "P(hh) | Minor Backgrounds")))
    tt_vs_oth = H((50, 0, 1, ('tt_vs_oth', "P(tt) | Minor Backgrounds")))

