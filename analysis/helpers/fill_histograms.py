from src.hist_tools import Collection, Fill
from src.hist_tools.object import Elec, Jet, LorentzVector, Muon, Lepton
from src.hist_tools import H, Template

class Chi2Hists(Template):
    tot_4j   = H((50, -0.1, 6, ('tot_4j', 'tot chi square 4j2b')))
    tot_3j   = H((50, -0.1, 6, ('tot_3j', 'tot chi square 3j2b')))
    Hbb_mass = H((50, -0.1, 6, ('Hbb_mass',  'chi square for Hbb_mass')))
    Hww_mass = H((50, -0.1, 6, ('Hww_mass',  'chi square for Hww_mass')))
    Wqq_mass = H((50, -0.1, 6, ('Wqq_mass',  'chi square for Wqq_mass')))
    Wln_mT   = H((50, -0.1, 6, ('Wln_mT',    'chi square for Wln_mT')))
    Hbb_dr   = H((50, -0.1, 6, ('Hbb_dr',    'chi square for Hbb_dr')))



def add_bbWW_common_hists(fill, hist):

    #
    #  Event Level
    #
    fill += hist.add("nPVs", (101, -0.5, 100.5, ("PV.npvs", "Number of Primary Vertices")))
    fill += hist.add("nPVsGood", (101, -0.5, 100.5, ("PV.npvsGood", "Number of Good Primary Vertices")))
    fill += hist.add("MET", (50, -0.5, 250, ("MET.pt", "MET pT [GeV]")))
    fill += hist.add("njets", (10, -0.5, 9.5, ("njets", "jet multiplicity")))



    fill += hist.add("chi_sq_hadW",  (30, -0.5, 6, ("chi_sq_hadW",  "hadronic W region chi square")))
    fill += hist.add("chi_sq_hadWs", (30, -0.5, 6, ("chi_sq_hadWs", "leptonic W region chi square")))
    fill += hist.add("chi_sq_tt",    (30, -0.5, 6, ("chi_sq_tt",    "ttbar chi square")))


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

    fill += Chi2Hists(("chi2_hadWs", "chi2 hadWs"), "chi2_hadWs")
    fill += Chi2Hists(("chi2_hadW",  "chi2 hadW"),  "chi2_hadW")


    #
    # Wqq Candidate
    #
    fill += Jet.plot_pair( ("Wqq", R"$W_{qq}$"), "Wqq_cand", skip=["n"], bins={"mass": (120, 0, 200)}, )

    #
    #  HWW Candidate
    #
    fill += LorentzVector.plot_pair( ("HWW", R"$H_{WW}$"), "Hww_cand", skip=["n","lead","subl","st"], bins={"mass": (100, 100, 400)}, )


    fill += hist.add("bjets_genjets_dr",   (30, -0.5, 5, ("bjets_genjets_dr", r'$\Delta$ R between b-candidates (genjets)')))
    fill += hist.add("bjets_genjets_mass", (50, -0.5, 250, ("bjets_genjets_mass", "H-> bb candidate (genjets) mass[GeV]")))

    fill += hist.add("genjets_mbb_vs_bb_dr",
                     (50, 0, 250, ('bjets_genjets_mass', 'H->bb Candidate (genjets) Mass [GeV]')),
                     (50, 0, 5, ('bjets_genjets_dr', r'$\Delta R$ between b-candidates (genjets)')))

    fill += hist.add("lep_qq_pt_dr",
                (50, 0, 250, ('leading_lep.pt', 'leading lepton pT [GeV]')),
                (50, 0, 5, ('lep_qq_dr', r'$\Delta R$ between leading lepton and selected qq')))


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

    fill += Chi2Hists(("chi2_hadWs", "chi2 hadWs"), "chi2_hadWs",
                      skip=["tot_4j", "Hww_mass", "Wqq_mass",]
                      )


    fill += Chi2Hists(("chi2_hadW",  "chi2 hadW"),  "chi2_hadW",
                      skip=["tot_4j", "Hww_mass", "Wqq_mass",]
                      )

    fill += hist.add("bjets_genjets_dr",   (30, -0.5, 5, ("bjets_genjets_dr", r'$\Delta$ R between b-candidates (genjets)')))
    fill += hist.add("bjets_genjets_mass", (50, -0.5, 250, ("bjets_genjets_mass", "H-> bb candidate (genjets) mass[GeV]")))


    fill += hist.add("genjets_mbb_vs_bb_dr",
                    (50, 0, 250, ('bjets_genjets_mass', 'H->bb Candidate (genjets) Mass [GeV]')),
                    (50, 0, 5, ('bjets_genjets_dr', r'$\Delta R$ between b-candidates (genjets)')))

    fill += hist.add("lep_qq_pt_dr",
                (50, 0, 250, ('leading_lep.pt', 'leading lepton pT [GeV]')),
                (50, 0, 5, ('lep_qq_dr', r'$\Delta R$ between leading lepton and selected qq')))



    # fill histograms
    fill(events, hist)



    return hist.to_dict(nonempty=True)
