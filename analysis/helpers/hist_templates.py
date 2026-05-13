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
    
    phh_variable_nominal = H(([0.000000, 0.012487, 0.021573, 0.030475, 0.040586, 0.049177, 0.060150, 0.073422, 0.086895, 0.102351, 0.122105, 0.137716, 0.157491, 0.174290, 0.194212, 0.215496, 0.238422, 0.262152, 0.287313, 0.315634, 0.343960, 0.374287, 0.404945, 0.432944, 0.466294, 0.497691, 0.526706, 0.559833, 0.587227, 0.616853, 0.645867, 0.676430, 0.703447, 0.728866, 0.753531, 0.779323, 0.803516, 0.825933, 0.846198, 0.865016, 0.883759, 0.901100, 0.914514, 0.928717, 0.943950, 0.955421, 0.967107, 0.976605, 0.985227, 0.991964, 1.000000],
            ('phh', "Regressed P(Signal) bin #")))
    
    phh_variable_lowpt = H(([0.000000, 0.011851, 0.017848, 0.023470, 0.029058, 0.036643, 0.043229, 0.053216, 0.062162, 0.070799, 0.079870, 0.093265, 0.107530, 0.120730, 0.136159, 0.152767, 0.172419, 0.195941, 0.222552, 0.247029, 0.270114, 0.296222, 0.325773, 0.358558, 0.391690, 0.432992, 0.464730, 0.493613, 0.522110, 0.559639, 0.597013, 0.626003, 0.660341, 0.702312, 0.733715, 0.764745, 0.797398, 0.818155, 0.842838, 0.867434, 0.891592, 0.915325, 0.938194, 0.954010, 0.971661, 0.984673, 1.000000],
            ('phh', "Regressed P(Signal) bin #")))

    phh_variable_3j2b = H(([0.000000, 0.009722, 0.015128, 0.020576, 0.026126, 0.032270, 0.038436, 0.045637, 0.054570, 0.064396, 0.073045, 0.085494, 0.097681, 0.111196, 0.125162, 0.143415, 0.160308, 0.179614, 0.201934, 0.224122, 0.249731, 0.278098, 0.305766, 0.343959, 0.380775, 0.421817, 0.462964, 0.501593, 0.549505, 0.595981, 0.642196, 0.684835, 0.728851, 0.769677, 0.810461, 0.844704, 0.886407, 0.920026, 0.945547, 0.969524, 1.000000],
                           ('phh', "Regressed P(Signal) bin #")))
    
    ptt     = H((50, 0, 1, ('ptt', "Regressed P(tt)")))
    poth     = H((50, 0, 1, ('poth', "P(minor backgrounds)")))
    hh_vs_tt = H((50, 0, 1, ('hh_vs_tt', "P(hh) | TTbar")))
    hh_vs_oth = H((50, 0, 1, ('hh_vs_oth', "P(hh) | Minor Backgrounds")))
    tt_vs_oth = H((50, 0, 1, ('tt_vs_oth', "P(tt) | Minor Backgrounds")))
    #WW = H((100, 0, 20, ('WW', "WW classifier score")))

