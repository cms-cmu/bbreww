import awkward as ak
import src.data_formats.awkward as akext
import numpy as np
from src.data_formats.root import Chunk, Friend
from src.storage.eos import PathLike
from src.friendtrees.dump_friend import dump_friend, _build_cutflow

_NAMING = "{path1}/{name}_{uuid}_{start}_{stop}_{path0}"


def dump_input_friend(
    events: ak.Array,
    output: PathLike,
    name: str,
    *selections: ak.Array,
    bcand: str = "b_cands",
    nonbcand: str = "q_cands_nom",
    lepton: str = "leading_lep",
    met: str = "MET",
    weight: str = "weight",
    dump_naming: str = _NAMING,
):
    selection = _build_cutflow(*selections)
    padded = akext.pad.selected()
    data = ak.Array(
        {

            "bJetCand": padded(
                ak.zip(
                    {
                        "pt": events[bcand].pt,
                        "eta": events[bcand].eta,
                        "phi": events[bcand].phi,
                        "mass": events[bcand].mass,
                        "btagScore": events[bcand].btagScore
                    }
                ),
                selection,
            ),
            "nonbJetCand": padded(
                ak.zip(
                    {
                        "pt":   events[nonbcand].pt,
                        "eta":  events[nonbcand].eta,
                        "phi":  events[nonbcand].phi,
                        "mass": events[nonbcand].mass,
                    }
                ),
                selection,
            ),
            "leadingLep": padded(
                ak.zip(
                    {
                        "pt":   events[lepton].pt,
                        "eta":  events[lepton].eta,
                        "phi":  events[lepton].phi,
                        "mass": events[lepton].mass,
                        "isE" : events.flavor.e,
                        "isM" : events.flavor.mu
                    }
                ),
                selection,
            ),
            "MET": padded(
                ak.zip(
                    {
                        "pt":   events[met].pt,
                        "phi":  events[met].phi,
                    }
                ),
                selection,
            ),
        }
        | akext.to_numpy(
            padded(
                events["region"][
                    [
                        "SR",
                        "CR",
                    ]
                ],
                selection,
            )
        )
        | akext.to_numpy(
            padded(
                events[
                    [
                        "njets",
                        "nsoftjets",
                        "HT",
                        "HTsoft"
		    ]	
		],
                selection,
            )
	)
        | {"weight": padded(events[weight], selection)}
    )
    return dump_friend(
        events=events,
        output=output,
        name=name,
        data=data,
        dump_naming=dump_naming,
    )
