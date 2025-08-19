from src.hist import Collection, Fill
from src.hist.object import Elec, Jet, LorentzVector, Muon

def fill_histograms(
    events, 
    processName: str = None,
    year: str = 'UL18',
    is_mc: bool = False,
    selection_list: list = ['basic_selection', 'preselection'],
    channel_list: list = ['hadronic_W','leptonic_W', 'null']
):

    fill = Fill(
        process=processName, 
        year=year, 
        weight="weight")
    
    hist = Collection(
        process=[processName],
        year=[year],
        channel=channel_list,
        selection=selection_list,
    )

    fill += hist.add("nPVs", (101, -0.5, 100.5, ("PV.npvs", "Number of Primary Vertices")))
    fill += hist.add("nPVsGood", (101, -0.5, 100.5, ("PV.npvsGood", "Number of Good Primary Vertices")))

    ### gen studies plots ###
    #fill += hist.add("met_pt_res", (50, -100, 100, ("met_pt_res", r"MET $p_T$ [GeV]")))
    #fill += hist.add("met_pz_res", (50, -250, 250, ("met_pz_res", r"MET $p_z$ [GeV] ")))
    #fill += hist.add("W_mass_res", (50, -100, 100, ("W_mass_res", "leptonic W mass resolution [GeV]")))
    #fill += hist.add("genW_mass", (50, 30, 150, ("genW_mass", "leptonic W mass resolution [GeV]")))
    #fill += hist.add("rec_W", (50, 30, 150, ("rec_W", "leptonic W mass resolution [GeV]")))
    #fill += hist.add("mlvqq_hadWs", (50, 50, 300, ("mlvqq_hadWs", "reconstructed H->lvqq mass [GeV]")))
    
    #fill += hist.add("Wjets_pre_lead",    (15, 15 , 30,    ("Wjets_pre_lead.pt", r"$jet pT$[GeV]")))
    #fill += hist.add("Wjets_post_lead",   (15, 15 , 30,   ("Wjets_post_lead.pt", r"$jet pT$[GeV]")))
    #fill += hist.add("Wjets_pre_sublead", (15, 15 , 30, ("Wjets_pre_sublead.pt", r"$jet pT$[GeV]")))    
    #fill += hist.add("Wjets_post_sublead",(15, 15 , 30,("Wjets_post_sublead.pt", r"$jet pT$[GeV]")))
    #fill += hist.add("dijets_pre_lead",    (15, 15 , 30,  ("dijets_pre_lead.pt", r"$jet pT$[GeV]")))
    #fill += hist.add("dijets_post_lead",   (15, 15 , 30, ("dijets_post_lead.pt", r"$jet pT$[GeV]")))
    #fill += hist.add("dijets_pre_sublead", (15, 15 , 30,  ("dijets_pre_sublead.pt", r"$jet pT$[GeV]")))
    #fill += hist.add("dijets_post_sublead",(15, 15 , 30, ("dijets_post_sublead.pt", r"$jet pT$[GeV]")))
    #fill += hist.add("gen_bb", (60, -0.5, 250, ("gen_bb.mass", "gen bb mass [GeV]")))
    #fill += hist.add("genjet_from_b", (60, -0.5, 250, ("genjet_from_b.mass", "gen bb mass [GeV]")))
    #fill += hist.add("mass_reco_b_gen_match", (60, -0.5, 250, ("mass_reco_b_gen_match.mass", "gen bb mass [GeV]")))
    #fill += hist.add("bb_dr", (30, 0.5, 8, ("bb_dr", "delta r between two b-candidates")))
    #fill += hist.add("mbb", (60, -0.5, 250, ("mbb", "H-> bb candidate mass[GeV]")))
    #fill += hist.add("qq", (50, -0.5, 140, ("j_nonbcand.pt", "non-bjets pT [GeV]")))
    #fill += hist.add("mT", (60, -0.5, 250, ("mT_leading_lep", "transverse mass W->lv [GeV]")))

    #fill += hist.add("chi_sq_hadW", (30, -0.5, 6, ("chi_sq_hadW", "hadronic W region chi square")))
    #fill += hist.add("chi_sq_hadWs", (30, -0.5, 6, ("chi_sq_hadWs", "leptonic W region chi square")))
    #fill += hist.add("chi_sq_tt", (30, -0.5, 6, ("chi_sq_tt", "ttbar chi square")))

    #fill += hist.add("Hbb_vs_HWW", 
    #                (50, 0, 250, ('mbb', 'H->bb Candidate Mass [GeV]')),
    #                (50, 0, 250, ('mlvqq_hadWs', 'H->WW Candidate Mass [GeV]')))
    
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