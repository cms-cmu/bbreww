from src.hist_tools import Collection, Fill
from src.hist_tools.object import Elec, Jet, LorentzVector, Muon, Lepton

from bbreww.analysis.helpers.hist_templates import SvBHists, Chi2Hists, TTbarHists

def add_bbWW_common_hists(fill, hist):

    #
    #  Event Level
    #
    fill += hist.add("nPVs", (101, -0.5, 100.5, ("PV.npvs", "Number of Primary Vertices")))
    fill += hist.add("nPVsGood", (101, -0.5, 100.5, ("PV.npvsGood", "Number of Good Primary Vertices")))
    fill += hist.add("MET", (50, -0.5, 250, ("MET.pt", "MET pT [GeV]")))
    fill += hist.add("njets", (10, -0.5, 9.5, ("njets", "jet multiplicity")))

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

    #
    # Wlnu Candidate and reconstructed neutrino pz
    #
    fill += Lepton.plot_leptonMeT( ("Wlnu", R"$W_{lnu}$"), "Wlnu_cand", skip=["n"], bins={"mass": (120, 0, 200)})
    fill += hist.add("nu_pz1",   (40, 0, 200, ("Wlnu_cand.pz_1", r'recontructed MET pz 1 (GeV)')))
    fill += hist.add("nu_pz2",   (40, 0, 200, ("Wlnu_cand.pz_2", r'recontructed MET pz 2 (GeV)')))
    
    #
    # Leptons
    #
    fill += Elec.plot( ("Elec", R"$Elec$"), "sel_elec", skip=["n"], )
    fill += Muon.plot( ("Muon", R"$Muon$"), "sel_muon", skip=["n"], )

    #
    #  From before
    #
    #fill += hist.add("bjets_genjets_dr",   (30, -0.5, 5, ("bjets_genjets_dr", r'$\Delta$ R between b-candidates (genjets)')))
    #fill += hist.add("bjets_genjets_mass", (50, -0.5, 250, ("bjets_genjets_mass", "H-> bb candidate (genjets) mass[GeV]")))


    #fill += hist.add("genjets_mbb_vs_bb_dr",
    #                (50, 0, 250, ('bjets_genjets_mass', 'H->bb Candidate (genjets) Mass [GeV]')),
    #                (50, 0, 5, ('bjets_genjets_dr', r'$\Delta R$ between b-candidates (genjets)')))

    #fill += hist.add("genW_mass_vs_subl_jet_pt",
    #            (50, 0, 250, ('gen_hadW.mass', 'W->qq gen mass [GeV]')),
    #            (50, 0, 250, ('true_ak4_2', r'W->qq subleading jet $p_T$')))

    return fill, hist


def fill_histograms_nominal(
    events,
    processName: str = None,
    year: str = 'UL18',
    is_mc: bool = False,
    histCuts: list = ['preselection'],
    #channel_list: list = ['hadronic_W','leptonic_W'],
    flavor_list: list = ['e', 'mu'],
    region_list: list = ['SR', 'CR'],
    run_SvB: bool = False
):

    fill = Fill(
        process=processName,
        year=year,
        weight="weight")

    hist = Collection(
        process=[processName],
        year=[year],
        #channel=channel_list,
        flavor = flavor_list,
        region = region_list,
        **dict((s, ...) for s in histCuts)
    )


    #
    #  Common Histograms:  Hbb and leptons
    #
    fill, hist = add_bbWW_common_hists(fill, hist)

    # jet selection efficiencies
    fill += hist.add("true_jets_sublead.pt", (50, -0.5, 250, ("true_ak4_2", "pT [GeV]")))
    fill += hist.add("true_jets_lead.pt", (50, -0.5, 250, ("true_ak4_1", "pT [GeV]")))
    fill += hist.add("true_soft_jets_sel_sublead.pt", (50, -0.5, 250, ("q_soft_true_sublead", "pT[GeV]"))) # softjet 1
    fill += hist.add("true_soft_jets_sel_lead.pt", (50, -0.5, 250, ("q_soft_true_lead", "pT[GeV]"))) # softjet 2
    fill += hist.add("true_nom_jets_sel_sublead.pt", (50, -0.5, 250, ("q_nom_true_sublead", "pT[GeV]"))) # nominal jet 1
    fill += hist.add("true_nom_jets_sel_lead.pt", (50, -0.5, 250, ("q_nom_true_lead", "pT[GeV]"))) # nominal jet 2    
    fill += hist.add("true_ml_jets_sel_lead.pt", (50, -0.5, 250, ("q_ml_true_lead", "pT[GeV]"))) # ml classifier jet 1
    fill += hist.add("true_ml_jets_sel_sublead.pt", (50, -0.5, 250, ("q_ml_true_sublead", "pT[GeV]"))) # ml classifier jet 2        
    
    fill += Chi2Hists(("chi2_hadWs",      "chi2 hadWs"),         "chi2_hadWs")
    fill += Chi2Hists(("chi2_hadW",       "chi2 hadW"),          "chi2_hadW")
    fill += Chi2Hists(("chi2_tt",         "chi2 tt"),            "chi2_tt")

    #
    # Wqq Candidate
    #
    fill += Jet.plot_pair( ("Wqq", R"$W_{qq}$"), "Wqq_cand", skip=["n"], bins={"mass": (120, 0, 200)}, )

    #
    #  HWW Candidate
    #
    fill += LorentzVector.plot_pair( ("HWW", R"$H_{WW}$"), "Hww_cand", skip=["n","lead","subl","st"], bins={"mass": (100, 100, 400)}, )
    fill += hist.add("HWW.lqq_dr", (50, -0.5, 10, ("Hww_cand.lqq_dr", "qq - lepton  delta R")))

    fill += hist.add("mbb_vs_lep_qq_dr",
            (50, 0, 250, ('Hbb_cand.mass', 'H->bb Candidate Mass [GeV]')),
            (50, 0, 5, ('Hww_cand.lqq_dr', r'$\Delta R$ between leading lepton and selected qq')))
    
    #  TTbar Candidate
    #
    fill += TTbarHists( ("tt", R"$t\bar{t}$"), "tt_sel" )

    #
    # Signal vs Backgrounds classifier scores hists
    if run_SvB:
        fill += SvBHists(("SvB", "SvB Classifier"), "SvB")

    # fill histograms
    fill(events, hist)

    return hist.to_dict(nonempty=False)

def fill_histograms(
    events,
    processName: str = None,
    year: str = 'UL18',
    is_mc: bool = False,
    histCuts: list = ['preselection'],
    #channel_list: list = ['hadronic_W','leptonic_W'],
    flavor_list: list = ['e', 'mu'],
    region_list: list = ['SR', 'CR'],
    run_SvB: bool = False,
):

    fill = Fill(
        process=processName,
        year=year,
        weight="weight")

    hist = Collection(
        process=[processName],
        year=[year],
        #channel=channel_list,
        flavor = flavor_list,
        region = region_list,
        **dict((s, ...) for s in histCuts)
    )

    fill, hist = add_bbWW_common_hists(fill, hist)

    # jet selection efficiencies
    fill += hist.add("true_jets_sublead.pt", (10, 14.5, 30, ("true_ak4_2", "pT [GeV]")))
    fill += hist.add("true_jets_lead.pt", (10, 14.5, 30, ("true_ak4_1", "pT [GeV]")))
    fill += hist.add("true_soft_jets_sel_sublead.pt", (10, 14.5, 30, ("q_soft_true_sublead", "pT[GeV]"))) # softjet 1
    fill += hist.add("true_soft_jets_sel_lead.pt", (10, 14.5, 30, ("q_soft_true_lead", "pT[GeV]"))) # softjet 2
    fill += hist.add("true_nom_jets_sel_sublead.pt", (10, 14.5, 30, ("q_nom_true_sublead", "pT[GeV]"))) # nominal jet 1
    fill += hist.add("true_nom_jets_sel_lead.pt", (10, 14.5, 30, ("q_nom_true_lead", "pT[GeV]"))) # nominal jet 2
    fill += hist.add("true_ml_jets_sel_lead.pt", (10, 14.5, 30, ("q_ml_true_lead", "pT[GeV]"))) # ml classifier jet 1
    fill += hist.add("true_ml_jets_sel_sublead.pt", (10, 14.5, 30, ("q_ml_true_sublead", "pT[GeV]"))) # ml classifier jet 2        
    
    fill += Chi2Hists(("chi2_hadWs", "chi2 hadWs"), "chi2_hadWs",
                      skip=["tot_4j", "Hww_mass", "Wqq_mass",]
                      )

    fill += Chi2Hists(("chi2_hadW",  "chi2 hadW"),  "chi2_hadW",
                      skip=["tot_4j", "Hww_mass", "Wqq_mass",]
                      )

    fill += Chi2Hists(("chi2_hadWs_soft", "chi2 hadWs soft"),    "chi2_hadWs_soft")
    fill += Chi2Hists(("chi2_hadW_soft",  "chi2 hadW soft"),     "chi2_hadW_soft")
    fill += Chi2Hists(("chi2_tt_soft",    "chi2 tt soft"),       "chi2_tt_soft")

    #
    #  HWW Candidate
    #
    fill += LorentzVector.plot_pair( ("HWW_soft", R"$H_{WW}$ (soft)"), "Hww_cand_soft", skip=["n","lead","subl","st"], bins={"mass": (100, 100, 400)}, )

    fill += TTbarHists( ("tt_soft", R"$t\bar{t}$"), "tt_soft_minChi2" )

    # Signal vs Backgrounds classifier scores hists
    if run_SvB:
        fill += SvBHists(("SvB", "SvB Classifier"), "SvB")

    # fill histograms
    fill(events, hist)

    return hist.to_dict(nonempty=False)
