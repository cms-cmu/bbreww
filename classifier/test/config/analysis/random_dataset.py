from __future__ import annotations
from typing import TYPE_CHECKING, Callable
from src.classifier.config.setting import IO
from src.classifier.config.setting.cms import CollisionData
from bbreww.classifier.config.setting.bbWW import InputBranch
from src.classifier.task import Analysis, ArgParser, converter

if TYPE_CHECKING:
    import awkward as ak

def _generate_dataset(seed: int, nevents: int, name: str, output: str):
    import awkward as ak
    import numpy as np
    from src.data_formats.root import TreeWriter
    from bbreww.classifier.test.physics.random_object import typed_uniform

    rng = np.random.Generator(np.random.PCG64(seed))
    n = nevents

    # Generate bJetCand (2 jets per event)
    bJetCand = ak.zip({
        "pt": typed_uniform(rng, 40, 200, n * 2, np.float32),
        "eta": typed_uniform(rng, -2.5, 2.5, n * 2, np.float32),
        "phi": typed_uniform(rng, -np.pi, np.pi, n * 2, np.float32),
        "mass": typed_uniform(rng, 5, 25, n * 2, np.float32),
        "btagScore": typed_uniform(rng, 0.5, 1.0, n * 2, np.float32),
    })
    bJetCand = ak.unflatten(bJetCand, 2)

    # Generate nonbJetCand (4 jets per event)
    nonbJetCand = ak.zip({
        "pt": typed_uniform(rng, 30, 150, n * 4, np.float32),
        "eta": typed_uniform(rng, -2.5, 2.5, n * 4, np.float32),
        "phi": typed_uniform(rng, -np.pi, np.pi, n * 4, np.float32),
        "mass": typed_uniform(rng, 5, 20, n * 4, np.float32),
        "attn_score": typed_uniform(rng, 0.0, 1.0, n * 4, np.float32),
    })
    nonbJetCand = ak.unflatten(nonbJetCand, 4)

    # Generate leadingLep (1 lepton per event)
    leadingLep = ak.zip({
        "pt": typed_uniform(rng, 25, 120, n, np.float32),
        "eta": typed_uniform(rng, -2.4, 2.4, n, np.float32),
        "phi": typed_uniform(rng, -np.pi, np.pi, n, np.float32),
        "mass": typed_uniform(rng, 0.1, 0.1, n, np.float32),
        "isE": rng.integers(0, 2, n).astype(np.int32),
        "isM": rng.integers(0, 2, n).astype(np.int32),
    })

    # Generate regressed_nu
    regressed_nu = ak.zip({
        "px": typed_uniform(rng, -50, 50, n, np.float32),
        "py": typed_uniform(rng, -50, 50, n, np.float32),
        "pz": typed_uniform(rng, -100, 100, n, np.float32),
        "E": typed_uniform(rng, 10, 200, n, np.float32),
    })

    # Generate flat and ancillary features
    true_nbjet_flat = ak.zip({
        "0": typed_uniform(rng, 0.0, 1.0, n, np.float32),
        "1": typed_uniform(rng, 0.0, 1.0, n, np.float32),
        "2": typed_uniform(rng, 0.0, 1.0, n, np.float32),
        "3": typed_uniform(rng, 0.0, 1.0, n, np.float32),
    })

    weight = typed_uniform(rng, 0.5, 1.5, n, np.float32)

    data = {
        "HT": typed_uniform(rng, 100, 800, n, np.float32),
        "njets": rng.integers(4, 8, n).astype(np.int32),
        "nsoftjets": rng.integers(0, 4, n).astype(np.int32),
        "year": np.full(n, 2018, dtype=np.int32),
        "weight": weight,
        "label": rng.integers(0, 2, n).astype(np.int32),
    }

    # Flatten zip structures with proper naming
    for field in ["pt", "eta", "phi", "mass", "btagScore"]:
        for i in range(2):
            data[f"bJetCand_{field}_{i}"] = bJetCand[field][:, i]
    for field in ["pt", "eta", "phi", "mass", "attn_score"]:
        for i in range(4):
            data[f"nonbJetCand_{field}_{i}"] = nonbJetCand[field][:, i]
    for field in ["pt", "eta", "phi", "mass", "isE", "isM"]:
        data[f"leadingLep_{field}"] = leadingLep[field]
    for field in ["px", "py", "pz", "E"]:
        data[f"regressed_nu_{field}"] = regressed_nu[field]
    for field in ["0", "1", "2", "3"]:
        data[f"true_nbjet_flat_{field}"] = true_nbjet_flat[field]

    writer = TreeWriter(output, name)
    writer.write(data)


class _Runner:
    def __init__(self, seed: int, nevents: int, name: str, output: str):
        self.seed = seed
        self.nevents = nevents
        self.name = name
        self.output = output

    def __call__(self):
        _generate_dataset(self.seed, self.nevents, self.name, self.output)


class JetsbbWW(Analysis):
    argparser = ArgParser()
    argparser.add_argument("--seed", type=int, default=0)
    argparser.add_argument("--nevents", type=converter.int_pos, default=1)
    argparser.add_argument("--name", type=str, default="dataset")

    def analyze(self, results):
        return [_Runner(self.opts.seed, self.opts.nevents, self.opts.name, IO.output)]
