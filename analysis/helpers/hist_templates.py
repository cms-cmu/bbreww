from src.hist_tools import H, Template

class SvBHists(Template):
    phh      = H((50, 0, 1, ('phh', "Regressed P(Signal)")))
    ptt     = H((50, 0, 1, ('ptt', "Regressed P(tt)")))
    poth     = H((50, 0, 1, ('poth', "P(minor backgrounds)")))
    hh_vs_tt = H((50, 0, 1, ('hh_vs_tt', "P(hh) | TTbar")))
    hh_vs_oth = H((50, 0, 1, ('hh_vs_oth', "P(hh) | Minor Backgrounds")))
    tt_vs_oth = H((50, 0, 1, ('tt_vs_oth', "P(tt) | Minor Backgrounds")))

