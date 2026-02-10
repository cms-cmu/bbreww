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

class SvBHists(Template):
    phh      = H((50, 0, 1, ('phh', "Regressed P(Signal)")))
    
    phh_variable_nominal      = H(([5.77115511e-09, 2.07861486e-03, 9.74256968e-03, 2.09019366e-02,
        3.59456162e-02, 5.49924561e-02, 7.47428291e-02, 9.52028170e-02,
        1.21005004e-01, 1.53258490e-01, 1.87231470e-01, 2.24898224e-01,
        2.69794477e-01, 3.18816095e-01, 3.64716995e-01, 4.13307336e-01,
        4.62918717e-01, 5.17182591e-01, 5.67307529e-01, 6.21275279e-01,
        6.74717878e-01, 7.20779525e-01, 7.61824812e-01, 8.08383809e-01,
        8.44516339e-01, 8.79015831e-01, 9.06797693e-01, 9.33613220e-01,
        9.55633926e-01, 9.74925759e-01, 9.99031663e-01],
            ('phh', "Regressed P(Signal) bin #")))
    
    phh_variable_lowpt      = H(([1.54415467e-08, 1.72320349e-03, 6.81343618e-03, 1.57521433e-02,
 2.76361583e-02, 4.10416707e-02, 5.80793296e-02, 7.46274552e-02,
 9.77299866e-02, 1.20794360e-01, 1.48998935e-01, 1.78634754e-01,
 2.19924739e-01, 2.59853094e-01, 3.02134047e-01, 3.50822357e-01,
 3.99033185e-01, 4.50288078e-01, 5.03829679e-01, 5.53545665e-01,
 6.06485643e-01, 6.56978902e-01, 7.11608144e-01, 7.56452737e-01,
 8.03913018e-01, 8.43366670e-01, 8.78862691e-01, 9.10970982e-01,
 9.38424080e-01, 9.62641380e-01, 9.94914055e-01],
            ('phh', "Regressed P(Signal) bin #")))
    
    ptt     = H((50, 0, 1, ('ptt', "Regressed P(tt)")))
    poth     = H((50, 0, 1, ('poth', "P(minor backgrounds)")))
    hh_vs_tt = H((50, 0, 1, ('hh_vs_tt', "P(hh) | TTbar")))
    hh_vs_oth = H((50, 0, 1, ('hh_vs_oth', "P(hh) | Minor Backgrounds")))
    tt_vs_oth = H((50, 0, 1, ('tt_vs_oth', "P(tt) | Minor Backgrounds")))
    #WW = H((100, 0, 20, ('WW', "WW classifier score")))

