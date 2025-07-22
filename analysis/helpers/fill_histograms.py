from base_class.hist import Collection, Fill
from base_class.hist.object import Elec, Jet, LorentzVector, Muon
import logging

def fill_histograms(
    events, 
    processName: str = None,
    year: str = 'UL18',
    is_mc: bool = False,
    selection_list: list = ['basic_selection', 'preselection'],
    channel_list: list = ["hadronic_W","leptonic_W"]
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