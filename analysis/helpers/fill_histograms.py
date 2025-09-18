from src.hist import Collection, Fill
from src.hist.object import Elec, Jet, LorentzVector, Muon

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

    fill += hist.add("nPVs", (101, -0.5, 100.5, ("PV.npvs", "Number of Primary Vertices")))
    fill += hist.add("nPVsGood", (101, -0.5, 100.5, ("PV.npvsGood", "Number of Good Primary Vertices")))

    ### gen studies plots ###
    #fill += hist.add("W_mass_res", (50, -100, 100, ("W_mass_res", "leptonic W mass resolution [GeV]")))
    #fill += hist.add("genW_mass", (50, 30, 150, ("genW_mass", "leptonic W mass resolution [GeV]")))
    #fill += hist.add("mlvqq_hadWs", (50, 50, 300, ("mlvqq_hadWs", "reconstructed H->lvqq mass [GeV]")))

    ### these histograms are just to study the quark vs. gluon selection efficiency
    #fill += hist.add("Wjets_pre_lead",    (15, 15 , 30,    ("Wjets_pre_lead.pt", r"$jet pT$[GeV]")))
    #fill += hist.add("Wjets_post_lead",   (15, 15 , 30,   ("Wjets_post_lead.pt", r"$jet pT$[GeV]")))
    #fill += hist.add("Wjets_pre_sublead", (15, 15 , 30, ("Wjets_pre_sublead.pt", r"$jet pT$[GeV]")))
    #fill += hist.add("Wjets_post_sublead",(15, 15 , 30,("Wjets_post_sublead.pt", r"$jet pT$[GeV]")))
    #fill += hist.add("dijets_pre_lead",    (15, 15 , 30,  ("dijets_pre_lead.pt", r"$jet pT$[GeV]")))
    #fill += hist.add("dijets_post_lead",   (15, 15 , 30, ("dijets_post_lead.pt", r"$jet pT$[GeV]")))
    #fill += hist.add("dijets_pre_sublead", (15, 15 , 30,  ("dijets_pre_sublead.pt", r"$jet pT$[GeV]")))
    #fill += hist.add("dijets_post_sublead",(15, 15 , 30, ("dijets_post_sublead.pt", r"$jet pT$[GeV]")))
    ###########

    #
    # Nominal Plots
    #
    fill += Jet.plot_pair( ("Hbb", R"$H_{bb}$"), "Hbb_cand", skip=["n"], bins={"mass": (120, 0, 200)}, )

    # print("Wqq_cand", events.wqq_cand[0:10].pt.tolist(),"\n")
    # print("q_cands_nom 0", events.q_cands_nom[0:10,0].pt.tolist(),"\n")
    # print("q_cands_nom 1", events.q_cands_nom[0:10,1].pt.tolist(),"\n")
    # print("njets", events.njets[0:10].tolist(),"\n")
    # fill += Jet.plot_pair( ("Wqq", R"$W_{qq}$"), "Hqq_cand", skip=["n"], bins={"mass": (120, 0, 200)}, )

    #fill += hist.add("gen_bb", (60, -0.5, 250, ("gen_bb.mass", "gen bb mass [GeV]")))
    #fill += hist.add("genjet_from_b", (60, -0.5, 250, ("genjet_from_b.mass", "gen bb mass [GeV]")))
    #fill += hist.add("mass_reco_b_gen_match", (60, -0.5, 250, ("mass_reco_b_gen_match.mass", "gen bb mass [GeV]")))
    fill += hist.add("bjets_genjets_dr", (30, -0.5, 5, ("bjets_genjets_dr", r'$\Delta$ R between b-candidates (genjets)')))
    fill += hist.add("bjets_genjets_mass", (50, -0.5, 250, ("bjets_genjets_mass", "H-> bb candidate (genjets) mass[GeV]")))
    fill += hist.add("nonbjet_pt_lead", (50, -0.5, 250, ("j_nonbcand_nom_lead_pt", "leading non-bjet pT [GeV]")))
    fill += hist.add("nonbjet_pt_sublead", (50, -0.5, 250, ("j_nonbcand_nom_sublead_pt", "subleading non-bjet pT [GeV]")))
    fill += hist.add("nonbjets_pt", (50, -0.5, 250, ("j_nonbcand_nom.pt", "non-bjets pT [GeV]")))

    fill += hist.add("qq_mass", (50, -0.5, 250, ("qq_mass", "non-bjets pT [GeV]")))
    fill += hist.add("mT", (60, -0.5, 250, ("mT_leading_lep", "transverse mass W->lv [GeV]")))
    fill += hist.add("leading_e", (50, -0.5, 250, ("leading_e.pt", "electron pT [GeV]")))
    fill += hist.add("leading_mu", (50, -0.5, 250, ("leading_mu.pt", "muon pT [GeV]")))
    fill += hist.add("MET", (50, -0.5, 250, ("MET.pt", "MET pT [GeV]")))
    fill += hist.add("njets", (10, -0.5, 9.5, ("njets", "jet multiplicity")))

    fill += hist.add("chi_sq_hadW", (30, -0.5, 6, ("chi_sq_hadW", "hadronic W region chi square")))
    fill += hist.add("chi_sq_hadWs", (30, -0.5, 6, ("chi_sq_hadWs", "leptonic W region chi square")))
    fill += hist.add("chi_sq_tt", (30, -0.5, 6, ("chi_sq_tt", "ttbar chi square")))

    fill += hist.add("mbb_vs_bb_dr",
                    (50, 0, 250, ('mbb', 'H->bb Candidate Mass [GeV]')),
                    (50, 0, 5, ('bb_dr', r'$\Delta R$ between b-candidates')))
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

    #### THIS IS JUST A COPY OF THE PREVIOUS FILL HISTOGRAMS FUNCTION
    # fill += hist.add("hT", (50, 0, 1500, ("hT", "h_{T} [GeV]")))

    # Jets
    # skip_jet_list = ['energy', 'deepjet_c']
    # fill += Jet.plot(("selJets", "Selected Jets"), "selJet", skip=skip_jet_list, bins={"mass": (50, 0, 100)})
    # fill += Jet.plot(("canJets", "Higgs Candidate Jets"), "canJet", skip=skip_jet_list, bins={"mass": (50, 0, 100)})
    # fill += Jet.plot(("othJets", "Other Jets"), "notCanJet_coffea", skip=skip_jet_list, bins={"mass": (50, 0, 100)})
    # fill += Jet.plot(("tagJets", "Tag Jets"), "tagJet", skip=skip_jet_list, bins={"mass": (50, 0, 100)})

    # # Leptons
    # skip_muons = ["charge"] + Muon.skip_detailed_plots
    # if not is_mc:
    #     skip_muons += ["genPartFlav"]
    # fill += Muon.plot(("selMuons", "Selected Muons"), "selMuon", skip=skip_muons)

    # if "Elec" in events.fields:
    #     skip_elecs = ["charge"] + Elec.skip_detailed_plots
    #     if not is_mc:
    #         skip_elecs += ["genPartFlav"]
    #     fill += Elec.plot(("selElecs", "Selected Elecs"), "selElec", skip=skip_elecs)


    # fill histograms
    fill(events, hist)

    return hist.to_dict(nonempty=True)
