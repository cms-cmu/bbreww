from src.hist_tools import Collection, Fill
from src.hist_tools.object import Elec, Jet, LorentzVector, Muon, Lepton


def add_bbWW_common_hists(fill, hist):

    #
    #  Event Level
    #
    fill += hist.add("nPVs", (101, -0.5, 100.5, ("PV.npvs", "Number of Primary Vertices")))
    fill += hist.add("nPVsGood", (101, -0.5, 100.5, ("PV.npvsGood", "Number of Good Primary Vertices")))
    fill += hist.add("MET", (50, -0.5, 250, ("MET.pt", "MET pT [GeV]")))
    fill += hist.add("njets", (10, -0.5, 9.5, ("njets", "jet multiplicity")))

    #
    # Hbb Candidate
    #
    fill += Jet.plot_pair( ("Hbb", R"$H_{bb}$"), "Hbb_cand", skip=["n"], bins={"mass": (120, 0, 200)}, )
    fill += hist.add("mbb_vs_bb_dr",
                    (50, 0, 250, ('Hbb_cand.mass', 'H->bb Candidate Mass [GeV]')),
                    (50, 0,   5, ('Hbb_cand.dr', r'$\Delta R$ between b-candidates')))


    #
    # Wlnu Candidate
    #
    fill += Lepton.plot_leptonMeT( ("Wlnu", R"$W_{lnu}$"), "Wlnu_cand", skip=["n"], bins={"mass": (120, 0, 200)}, )


    #
    # Leptons
    #
    fill += Elec.plot( ("Elec", R"$Elec$"), "sel_elec", skip=["n"], )
    fill += Muon.plot( ("Muon", R"$Muon$"), "sel_muon", skip=["n"], )


    return fill, hist


def fill_histograms_nominal(
    events,
    processName: str = None,
    year: str = 'UL18',
    is_mc: bool = False,
    histCuts: list = ['preselection'],
    channel_list: list = ['hadronic_W','leptonic_W'],
    flavor_list: list = ['e', 'mu'],
    #region_list: list = ['e_region', 'mu_region']
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
    fill, hist = add_bbWW_common_hists(fill, hist)


    #
    # Wqq Candidate
    #
    fill += Jet.plot_pair( ("Wqq", R"$W_{qq}$"), "Wqq_cand", skip=["n"], bins={"mass": (120, 0, 200)}, )

    fill += hist.add("bjets_genjets_dr", (30, -0.5, 5, ("bjets_genjets_dr", r'$\Delta$ R between b-candidates (genjets)')))
    fill += hist.add("bjets_genjets_mass", (50, -0.5, 250, ("bjets_genjets_mass", "H-> bb candidate (genjets) mass[GeV]")))

    fill += hist.add("chi_sq_hadW", (30, -0.5, 6, ("chi_sq_hadW", "hadronic W region chi square")))
    fill += hist.add("chi_sq_hadWs", (30, -0.5, 6, ("chi_sq_hadWs", "leptonic W region chi square")))
    fill += hist.add("chi_sq_tt", (30, -0.5, 6, ("chi_sq_tt", "ttbar chi square")))

    fill += hist.add("genjets_mbb_vs_bb_dr",
                    (50, 0, 250, ('bjets_genjets_mass', 'H->bb Candidate (genjets) Mass [GeV]')),
                    (50, 0, 5, ('bjets_genjets_dr', r'$\Delta R$ between b-candidates (genjets)')))

    fill += hist.add("lep_qq_pt_dr",
                (50, 0, 250, ('leading_lep.pt', 'leading lepton pT [GeV]')),
                (50, 0, 5, ('lep_qq_dr', r'$\Delta R$ between leading lepton and selected qq')))

    #fill += hist.add("Hbb_vs_HWW",
    #                (50, 0, 250, ('mbb', 'H->bb Candidate Mass [GeV]')),
    #                (50, 0, 250, ('mlvqq_hadWs', 'H->WW Candidate Mass [GeV]')))
    #
    #fill += hist.add("chiSq_vs_mbb",
    #            (50, 0, 250, ('mbb','H->bb Candidate Mass [GeV]')),
    #            (50, 0, 5, ('chi_sq_hadW', 'hadronic W region chi square')))


    # fill histograms
    fill(events, hist)

    return hist.to_dict(nonempty=True)




def fill_histograms(
    events,
    processName: str = None,
    year: str = 'UL18',
    is_mc: bool = False,
    histCuts: list = ['preselection'],
    channel_list: list = ['hadronic_W','leptonic_W'],
    flavor_list: list = ['e', 'mu'],
    #region_list: list = ['e_region', 'mu_region']
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

    fill, hist = add_bbWW_common_hists(fill, hist)


    #fill += hist.add("gen_bb", (60, -0.5, 250, ("gen_bb.mass", "gen bb mass [GeV]")))
    #fill += hist.add("genjet_from_b", (60, -0.5, 250, ("genjet_from_b.mass", "gen bb mass [GeV]")))
    #fill += hist.add("mass_reco_b_gen_match", (60, -0.5, 250, ("mass_reco_b_gen_match.mass", "gen bb mass [GeV]")))
    fill += hist.add("bjets_genjets_dr", (30, -0.5, 5, ("bjets_genjets_dr", r'$\Delta$ R between b-candidates (genjets)')))
    fill += hist.add("bjets_genjets_mass", (50, -0.5, 250, ("bjets_genjets_mass", "H-> bb candidate (genjets) mass[GeV]")))

    fill += hist.add("chi_sq_hadW", (30, -0.5, 6, ("chi_sq_hadW", "hadronic W region chi square")))
    fill += hist.add("chi_sq_hadWs", (30, -0.5, 6, ("chi_sq_hadWs", "leptonic W region chi square")))
    fill += hist.add("chi_sq_tt", (30, -0.5, 6, ("chi_sq_tt", "ttbar chi square")))

    fill += hist.add("genjets_mbb_vs_bb_dr",
                    (50, 0, 250, ('bjets_genjets_mass', 'H->bb Candidate (genjets) Mass [GeV]')),
                    (50, 0, 5, ('bjets_genjets_dr', r'$\Delta R$ between b-candidates (genjets)')))
    fill += hist.add("lep_qq_pt_dr",
                (50, 0, 250, ('leading_lep.pt', 'leading lepton pT [GeV]')),
                (50, 0, 5, ('lep_qq_dr', r'$\Delta R$ between leading lepton and selected qq')))



    # fill histograms
    fill(events, hist)



    return hist.to_dict(nonempty=True)
