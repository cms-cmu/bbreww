from src.hist_tools import Collection, Fill
from src.hist_tools.object import Elec, Jet, LorentzVector, Muon, Lepton
import awkward as ak
import numpy as np
from bbreww.analysis.helpers.hist_templates import SvBHists, Chi2Hists, TTbarHists, regressionHists

def add_bbWW_common_hists(fill, hist, SvB: bool = False, MET_regression: bool = False):

    #
    #  Event Level
    #
    fill += hist.add("nPVs", (101, -0.5, 100.5, ("PV.npvs", "Number of Primary Vertices")))
    fill += hist.add("nPVsGood", (101, -0.5, 100.5, ("PV.npvsGood", "Number of Good Primary Vertices")))
    fill += hist.add("MET_pt", (50, -0.5, 250, ("MET.pt", R"MET $p_T$ [GeV]")))
    fill += hist.add("MET_phi", (50, -4, 4, ("MET.phi", R"MET $Phi$")))
    fill += hist.add("njets", (10, -0.5, 9.5, ("njets", "jet multiplicity")))
    fill += hist.add("btag_sf", (50, 0.5, 1.5, ("btag_sf", "b-tagging SF")))
    fill += hist.add("lepnu_deta", (50, -3, 3, ("Wlnu_cand.deta", "delta_eta between lepton and neutrino")))

    ## W-qq quark vs. gluon selection candidates
    fill += hist.add("qvgScore", (50, 0, 1.0, ("q_cands_soft.btagPNetQvG", "ak4 jets quark vs. gluon score")))
    fill += Jet.plot_pair( ("Wqq_soft", R"$W_{qq}$"), "q_cands_soft", bins={"mass": (120, 0, 200)}, )
    #
    # Hbb Candidate
    #
    fill += Jet.plot_pair( ("Hbb", R"$H_{bb}$"), "Hbb_cand", skip=["n"], bins={"mass": (120, 0, 200)}, )
    fill += hist.add("mbb_vs_bb_dr",
                    (50, 0, 250, ('Hbb_cand.mass', 'H->bb Candidate Mass [GeV]')),
                    (50, 0,   5, ('Hbb_cand.dr', r'$\Delta R$ between b-candidates')))

    fill += hist.add("mbb_vs_mqq",
                    (50, 0, 250, ('Hbb_cand.mass', 'H->bb Candidate Mass [GeV]')),
                     (30, 0, 150, ('mqq', r'$\Delta R$ between b-candidates')),
                     mqq= lambda events: ak.fill_none((events.sel_qq_l + events.sel_qq_sl).mass, np.nan)
                     )

    fill += hist.add("mbb_vs_qpt",
                    (50, 0, 250, ('Hbb_cand.mass', 'H->bb Candidate Mass [GeV]')),
                     (30, 0, 150, ('qpt', r'$leading non-bjet pT$')),
                     qpt= lambda events: ak.fill_none(events.q_cands_soft[:,0].pt, np.nan)
                     )


    #
    # Wlnu Candidate and reconstructed neutrino pz
    #
    fill += Lepton.plot_leptonMeT( ("Wlnu", R"$W_{lnu}$"), "Wlnu_cand", skip=["n"], bins={"mass": (120, 0, 200)})
    fill += hist.add("nu_pz1",   (80, -200, 200, ("diff", r'recontructed MET pz 1 (GeV)')),
                     diff= lambda events: (events.Wlnu_cand.pz_1 - events.Wlnu_cand.pz_2))
    fill += hist.add("nu_pz2",   (40, 0, 200, ("Wlnu_cand.pz_2", r'recontructed MET pz 2 (GeV)')))
    
    #
    # Leptons
    #
    fill += Elec.plot( ("Elec", R"$Elec$"), "sel_elec", skip=["n"], )
    fill += Muon.plot( ("Muon", R"$Muon$"), "sel_muon", skip=["n"], )

    #
    # Signal vs Backgrounds classifier scores hists
    if SvB:
        fill += SvBHists(("SvB", "SvB Classifier"), "SvB")

    ## distribution and resolution plots for MET regressor
    if MET_regression:
        fill += hist.add("genNu_pt", (50, 0, 200, ("genNu_pt", R"gen $\nu p_T$ [GeV]")))
        fill += hist.add("genNu_eta", (50, -5, 5, ("genNu_eta", R"gen $\nu \eta$")))
        fill += hist.add("genNu_pz", (80, -200, 200, ("genNu_pz", R"gen $\nu p_z$")))
        fill += hist.add("gen_lepW_mass", (30, 0, 150, ("gen_lepW_mass", R"gen leptonic $m_W$ [GeV]")))
        
        fill += hist.add("islepW", (2, 0, 1.1, ("isLepW", R"leptonic W on shell boolean")))

        fill += hist.add("reg_mW",
                          (30, 0, 150, ('reg_mW', R"Regressed leptonic W mass [GeV]")))
        fill += hist.add("reg_nu_pt_res",
                        (40, -100, 100, ("nu_pt_res", R"(True - Regressed) $\nu p_T$ [GeV]")),
                        nu_pt_res=lambda events: ak.fill_none(ak.mask(events.genNu_pt - events.reg_nu.pt, events.HWW_mass > 150.0), np.nan)
                        )
        fill += hist.add("reg_nu_pz_res",
                        (60, -200, 200, ("nu_pz_res", R"(True - Regressed) $\nu |p_z|$")),
                        nu_pz_res= lambda events: ak.fill_none(ak.mask(events.genNu_pz - events.reg_nu.pz, events.HWW_mass > 150.0), np.nan)
                        )
        fill += hist.add("reg_nu_eta_res",
                        (50, -5, 5, ("nu_eta_res", R"(True - Regressed) $\nu \eta$")),
                        nu_eta_res=lambda events: events.genNu_eta -events.reg_nu.eta
                        )
        fill += hist.add("mW_res",
                        (40, -100, 100, ("mW_res", R"(True - Regressed) leptonic $m_W$")),
                        mW_res=lambda events: ak.fill_none(events.gen_lepW_mass - (events.reg_nu + events.leading_lep).mass, np.nan)
                        )
        # reco resolutions before regression
        fill += hist.add("reco_nu_pt_res",
                         (40, -100, 100, ("nu_pt_res", R"(True - Reco) $\nu \ p_T$ [GeV]")),
                         nu_pt_res=lambda events: events.genNu_pt -events.MET.pt
                         )
        fill += hist.add("reco_nu_phi_res",
                         (50, -5, 5, ("nu_phi_res", R"(True - Reco) $\nu \ \Phi$")),
                         nu_phi_res=lambda events: events.genNu_phi -events.MET.phi
                         )
        
        fill += regressionHists(("met_regressor", "MET Regressor"), "met_regressor")
        fill += hist.add("HWW_mass", (50, 0, 250, ("HWW_mass", "H-> WW mass [GeV]")))

        
        fill += hist.add("mbb_vs_mWW",
                         (50, 0, 250, ('Hbb_cand.mass', 'H->bb Candidate Mass [GeV]')),
                         (50, 0, 250, ('HWW_mass', r'$\Delta R$ between b-candidates')))

    return fill, hist


def fill_histograms_nominal(
    events,
    processName: str = None,
    year: str = 'UL18',
    is_mc: bool = False,
    histCuts: list = ['preselection'],
    channel_list: list = ['hadronic_W','leptonic_W'],
    flavor_list: list = ['e', 'mu'],
    #region_list: list = ['SR', 'CR'],
    run_SvB: bool = False,
    run_MET_regression: bool = False,
):

    fill = Fill(
        process=processName,
        year=year,
        weight="weight")

    hist = Collection(
        process=[processName],
        year=[year],
        channel=channel_list,
        flavor = flavor_list,
        #region = region_list,
        **dict((s, ...) for s in histCuts)
    )


    #
    #  Common Histograms:  Hbb and leptons
    #
    fill, hist = add_bbWW_common_hists(fill, hist, run_SvB, run_MET_regression)

    # jet selection efficiencies
    fill += hist.add("true_jets_sublead.pt", (50, -0.5, 250, ("true_ak4_2", "pT [GeV]")))
    fill += hist.add("true_jets_lead.pt", (50, -0.5, 250, ("true_ak4_1", "pT [GeV]")))
    fill += hist.add("true_soft_jets_sel_sublead.pt", (50, -0.5, 250, ("q_soft_true_sublead", "pT[GeV]"))) # softjet 1
    fill += hist.add("true_soft_jets_sel_lead.pt", (50, -0.5, 250, ("q_soft_true_lead", "pT[GeV]"))) # softjet 2
    fill += hist.add("true_nom_jets_sel_sublead.pt", (50, -0.5, 250, ("q_nom_true_sublead", "pT[GeV]"))) # nominal jet 1
    fill += hist.add("true_nom_jets_sel_lead.pt", (50, -0.5, 250, ("q_nom_true_lead", "pT[GeV]"))) # nominal jet 2    
    fill += hist.add("true_ml_jets_sel_lead.pt", (50, -0.5, 250, ("q_ml_true_lead", "pT[GeV]"))) # ml both correct
    fill += hist.add("true_ml_jets_sel_sublead.pt", (50, -0.5, 250, ("q_ml_true_sublead", "pT[GeV]"))) # ml both correct
    fill += hist.add("ml_lead_denom.pt", (50, -0.5, 250, ("q_ml_lead_denom", "pT[GeV]"))) # >= 1 true jet
    fill += hist.add("ml_lead_numer.pt", (50, -0.5, 250, ("q_ml_lead_numer", "pT[GeV]"))) # ML lead correct
    fill += hist.add("ml_sublead_denom.pt", (50, -0.5, 250, ("q_ml_sublead_denom", "pT[GeV]"))) # >= 2 true jets
    fill += hist.add("ml_sublead_numer.pt", (50, -0.5, 250, ("q_ml_sublead_numer", "pT[GeV]"))) # ML sublead correct
    fill += hist.add("misclass_p_onshell", (20, 0, 1.1, ("misclass_p_onshell", "p(on-shell)"))) # misclassified & gen_lepW < 20
    fill += hist.add("misclass_sigma_pz_on", (100, 0, 300, ("misclass_sigma_pz_on", "on-shell uncertainty"))) # misclassified & gen_lepW < 20

    # fill += Chi2Hists(("chi2_hadWs",      "chi2 hadWs"),         "chi2_hadWs")
    # fill += Chi2Hists(("chi2_hadW",       "chi2 hadW"),          "chi2_hadW")
    # fill += Chi2Hists(("chi2_tt",         "chi2 tt"),            "chi2_tt")

    #
    # Wqq Candidate
    #
    fill += Jet.plot_pair( ("Wqq", R"$W_{qq}$"), "Wqq_cand", skip=["n"], bins={"mass": (120, 0, 200)}, )

    #
    #  HWW Candidate
    #
    fill += LorentzVector.plot_pair( ("HWW", R"$H_{WW}$"), "Hww_cand", skip=["n","lead","subl","st"], bins={"mass": (100, 100, 400)}, )
    fill += hist.add("HWW.lqq_dr", (50, -0.5, 10, ("Hww_cand.lqq_dr", "qq - lepton  delta R")))
    fill += hist.add("HWW.lqq_mass", (50, -0.5, 10, ("Hww_cand.lqq_mass", "(qq + lepton) mass ")))
    
    #  TTbar Candidate
    #
    fill += TTbarHists( ("tt", R"$t\bar{t}$"), "tt_sel" )

    # fill histograms
    fill(events, hist)

    return hist.to_dict(nonempty=False)

def fill_histograms(
    events,
    processName: str = None,
    year: str = 'UL18',
    is_mc: bool = False,
    histCuts: list = ['preselection'],
    channel_list: list = ['hadronic_W','leptonic_W'],
    flavor_list: list = ['e', 'mu'],
    #region_list: list = ['SR', 'CR'],
    run_SvB: bool = False,
    run_MET_regression: bool = False,
):

    fill = Fill(
        process=processName,
        year=year,
        weight="weight")

    hist = Collection(
        process=[processName],
        year=[year],
        channel=channel_list,
        flavor = flavor_list,
        #region = region_list,
        **dict((s, ...) for s in histCuts)
    )

    fill, hist = add_bbWW_common_hists(fill, hist, run_SvB, run_MET_regression)

    # jet selection efficiencies
    fill += hist.add("true_jets_sublead.pt", (20, 14.5, 30, ("true_ak4_2", "pT [GeV]")))
    fill += hist.add("true_jets_lead.pt", (20, 14.5, 30, ("true_ak4_1", "pT [GeV]")))
    fill += hist.add("true_soft_jets_sel_sublead.pt", (20, 14.5, 30, ("q_soft_true_sublead", "pT[GeV]"))) # softjet 1
    fill += hist.add("true_soft_jets_sel_lead.pt", (20, 14.5, 30, ("q_soft_true_lead", "pT[GeV]"))) # softjet 2
    fill += hist.add("true_nom_jets_sel_sublead.pt", (20, 14.5, 30, ("q_nom_true_sublead", "pT[GeV]"))) # nominal jet 1
    fill += hist.add("true_nom_jets_sel_lead.pt", (20, 14.5, 30, ("q_nom_true_lead", "pT[GeV]"))) # nominal jet 2
    fill += hist.add("true_ml_jets_sel_lead.pt", (20, 14.5, 30, ("q_ml_true_lead", "pT[GeV]"))) # ml both correct
    fill += hist.add("true_ml_jets_sel_sublead.pt", (20, 14.5, 30, ("q_ml_true_sublead", "pT[GeV]"))) # ml both correct
    fill += hist.add("ml_lead_denom.pt", (20, 14.5, 30, ("q_ml_lead_denom", "pT[GeV]"))) # >= 1 true jet
    fill += hist.add("ml_lead_numer.pt", (20, 14.5, 30, ("q_ml_lead_numer", "pT[GeV]"))) # ML lead correct
    fill += hist.add("ml_sublead_denom.pt", (20, 14.5, 30, ("q_ml_sublead_denom", "pT[GeV]"))) # >= 2 true jets
    fill += hist.add("ml_sublead_numer.pt", (20, 14.5, 30, ("q_ml_sublead_numer", "pT[GeV]"))) # ML sublead correct
    fill += hist.add("misclass_p_onshell", (10, 0, 1.1, ("misclass_p_onshell", "p(on-shell)"))) # misclassified & gen_lepW < 20
    fill += hist.add("misclass_sigma_pz_on", (50, 0, 100, ("misclass_sigma_pz_on", "on-shell uncertainty"))) # misclassified & gen_lepW < 20

    # fill += Chi2Hists(("chi2_hadWs", "chi2 hadWs"), "chi2_hadWs",
    #                   skip=["tot_4j", "Hww_mass", "Wqq_mass",]
    #                   )

    # fill += Chi2Hists(("chi2_hadW",  "chi2 hadW"),  "chi2_hadW",
    #                   skip=["tot_4j", "Hww_mass", "Wqq_mass",]
    #                   )

    # fill += Chi2Hists(("chi2_hadWs_soft", "chi2 hadWs soft"),    "chi2_hadWs_soft")
    # fill += Chi2Hists(("chi2_hadW_soft",  "chi2 hadW soft"),     "chi2_hadW_soft")
    # fill += Chi2Hists(("chi2_tt_soft",    "chi2 tt soft"),       "chi2_tt_soft")

    #
    #  HWW Candidate
    #
    fill += LorentzVector.plot_pair( ("HWW_soft", R"$H_{WW}$ (soft)"), "Hww_cand_soft", skip=["n","lead","subl","st"], bins={"mass": (100, 100, 400)}, )

    # fill += TTbarHists( ("tt_soft", R"$t\bar{t}$"), "tt_soft_minChi2" )

    # fill histograms
    fill(events, hist)

    return hist.to_dict(nonempty=False)
