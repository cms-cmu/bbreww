import os
import sys
import ROOT
import argparse
import logging
import json
import yaml
from copy import deepcopy

# Allow running from the barista root or from within bbreww/stats_analysis/
_barista_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _barista_root not in sys.path:
    sys.path.insert(0, _barista_root)
from src.tools.convert_json_to_root import json_to_TH1

import CombineHarvester.CombineTools.ch as ch
ROOT.gROOT.SetBatch(True)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

# bbreww sample names to merge, keyed by the metadata background label
MERGE_SAMPLES = {
    'tt':        ('TTToSemiLeptonic', 'TTTo2L2Nu', 'TTToHadronic'),
    'wjets':     ('WtoLNu-2Jets_0J', 'WtoLNu-2Jets_1J', 'WtoLNu-2Jets_2J'),
    'singletop': ('TbarWplustoLNu2Q', 'TbarWplusto2L2Nu',
                  'TWminustoLNu2Q',   'TWminusto2L2Nu',
                  'TBbarQ', 'TbarBQ', 'TBbartoLplusNuBbar', 'TbarBtoLminusNuB'),
}
_ALL_MERGE_SAMPLES = {s for samples in MERGE_SAMPLES.values() for s in samples}

def _bin_name(flavor, year):
    """Compose a combine bin name from (flavor, year). Combine bins cannot
    start with a digit, hence the 'y' prefix on the year."""
    return f"{flavor}_y{year}"


def create_combine_root_file(file_to_convert,
                             rebin,
                             var,
                             output_dir,
                             systematics_file,
                             flavors,
                             metadata_file='bbreww/stats_analysis/metadata/bbWW.yml',
                             stat_only=False):
    """Build ROOT shapes file and CombineHarvester datacards for bbWW.

    JSON axes order: [histogram][process][year][flavor]
    One combine bin is created per (flavor, year) — typical Run3 run gives
    2 flavors × 2 years = 4 bins.
    """

    logging.info(f"Reading {metadata_file}")
    metadata = yaml.safe_load(open(metadata_file, 'r'))
    metadata['processes']['all'] = {**metadata['processes']['signal'],
                                    **metadata['processes']['background']}

    logging.info(f"Reading {file_to_convert}")
    with open(file_to_convert, 'r') as f:
        coffea_hists = json.load(f)

    if systematics_file:
        logging.info(f"Reading {systematics_file}")
        with open(systematics_file, 'r') as f:
            coffea_hists_syst = json.load(f)

    # Years present in the JSON for the first signal process
    first_signal = next(iter(metadata['processes']['signal']))
    years_in_file = list(coffea_hists[var][first_signal].keys())
    logging.info(f"Years in file: {years_in_file}")
    logging.info(f"Flavors to use:  {flavors}")

    root_hists = {}      # bin_name → {process → TH1F or {variation → TH1F}}
    mcSysts = []
    bin_names = []

    for iyear in years_in_file:
        for iflavor in flavors:
            bname = _bin_name(iflavor, iyear)
            bin_names.append(bname)
            root_hists[bname] = {}

            for iprocess in coffea_hists[var].keys():
                try:
                    leaf = coffea_hists[var][iprocess][iyear][iflavor]
                except KeyError:
                    logging.warning(f"Missing leaf for {iprocess}/{iyear}/{iflavor}")
                    continue

                if iprocess in _ALL_MERGE_SAMPLES:
                    root_hists[bname][iprocess] = json_to_TH1(
                        leaf, f'{iprocess}_{bname}', rebin)
                elif iprocess in metadata['processes']['signal']:
                    root_hists[bname][iprocess] = {
                        'nominal': json_to_TH1(leaf, f'{iprocess}_{bname}', rebin)
                    }
                else:
                    logging.debug(f"Skipping {iprocess} (not signal/known background)")

            if systematics_file:
                for iprocess in metadata['processes']['signal']:
                    root_hists[bname][iprocess] = {}
                    if stat_only:
                        root_hists[bname][iprocess]['nominal'] = json_to_TH1(
                            coffea_hists_syst[var][iprocess][iyear]['nominal'][iflavor],
                            f'{iprocess}_nominal_{bname}', rebin)
                    else:
                        for ivar in coffea_hists_syst[var][iprocess][iyear].keys():
                            namevar = ivar.replace('_Up', 'Up').replace('_Down', 'Down')
                            for stat in ['hfstats1', 'hfstats2', 'lfstats1', 'lfstats2']:
                                if stat in namevar:
                                    year_tag = iyear.replace('_', '')
                                    namevar = namevar.replace(stat, f'{stat}_{year_tag}')
                                    break
                            tmpvar = namevar.replace('Up', '').replace('Down', '')
                            if tmpvar not in mcSysts and 'nominal' not in tmpvar:
                                mcSysts.append(tmpvar)
                            root_hists[bname][iprocess][namevar] = json_to_TH1(
                                coffea_hists_syst[var][iprocess][iyear][ivar][iflavor],
                                f'{iprocess}_{ivar}_{bname}', rebin)

    # Symmetrise one-sided signal shape systematics
    for bname in root_hists.keys():
        for ip in root_hists[bname].keys():
            if ip not in metadata['processes']['signal']:
                continue
            entries = root_hists[bname][ip]
            if not isinstance(entries, dict):
                continue
            for iv in list(entries.keys()):
                if 'Up' in iv or 'nominal' in iv:
                    continue
                up_key = iv.replace('Down', 'Up')
                if up_key not in entries:
                    continue
                nominal  = entries['nominal']
                Up_var   = entries[up_key]
                Down_var = entries[iv]
                for ibin in range(Up_var.GetNbinsX()):
                    up_bin   = Up_var.GetBinContent(ibin + 1)
                    down_bin = Down_var.GetBinContent(ibin + 1)
                    nom_bin  = nominal.GetBinContent(ibin + 1)
                    if ((up_bin < nom_bin) and (down_bin < nom_bin)) or \
                       ((up_bin > nom_bin) and (down_bin > nom_bin)):
                        max_diff = max(abs(up_bin - nom_bin), abs(down_bin - nom_bin))
                        Up_var.SetBinContent(ibin + 1, nom_bin + max_diff)
                        Down_var.SetBinContent(ibin + 1, nom_bin - max_diff)
                    if nom_bin > 0 and max(up_bin, down_bin) > nom_bin * 1.5:
                        tmp_nom  = nominal.GetBinContent(ibin)
                        tmp_up   = Up_var.GetBinContent(ibin)
                        tmp_down = Down_var.GetBinContent(ibin)
                        Down_var.SetBinContent(ibin + 1, nom_bin - (tmp_nom - tmp_down))
                        Up_var.SetBinContent(ibin + 1, nom_bin + (tmp_up - tmp_nom))

    # Merge background sample groups and rename signals to combine labels
    for bname in root_hists.keys():
        for bkg_key, sample_names in MERGE_SAMPLES.items():
            if bkg_key not in metadata['processes']['background']:
                continue
            merged_label = metadata['processes']['background'][bkg_key]['label']
            merged_hist  = None
            for ip in list(root_hists[bname].keys()):
                if ip in sample_names:
                    if merged_hist is None:
                        merged_hist = root_hists[bname][ip].Clone(merged_label)
                        merged_hist.SetTitle(f'{merged_label}_{bname}')
                    else:
                        merged_hist.Add(root_hists[bname][ip])
                    del root_hists[bname][ip]
            if merged_hist is not None:
                root_hists[bname][merged_label] = merged_hist

        for ip in list(root_hists[bname].keys()):
            if ip in metadata['processes']['signal']:
                label = metadata['processes']['signal'][ip]['label']
                root_hists[bname][label] = deepcopy(root_hists[bname][ip])
                if isinstance(root_hists[bname][label], ROOT.TH1F):
                    root_hists[bname][label].SetName(label)
                    root_hists[bname][label].SetTitle(f'{label}_{bname}')
                else:
                    for ivar in root_hists[bname][label]:
                        if 'nominal' in ivar:
                            root_hists[bname][label][ivar].SetName(label)
                            root_hists[bname][label][ivar].SetTitle(f'{label}_{bname}')
                        else:
                            root_hists[bname][label][ivar] = \
                                root_hists[bname][label][ivar].Clone(f'{label}_{ivar}')
                            root_hists[bname][label][ivar].SetTitle(
                                f'{label}_{ivar}_{bname}')
                del root_hists[bname][ip]
            elif ip not in metadata['processes']['all']:
                logging.debug(f"{ip} not in metadata processes, removing.")
                del root_hists[bname][ip]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file = "shapes.root"
    output = os.path.join(output_dir, output_file)

    # Fake data_obs from sum of backgrounds (CombineHarvester needs it for blind/Asimov)
    bkg_label_set = {metadata['processes']['background'][p]['label']
                     for p in metadata['processes']['background']}
    for bname in root_hists.keys():
        data_obs = None
        for ih_name, ih in root_hists[bname].items():
            if ih_name not in bkg_label_set:
                continue
            hist = ih if isinstance(ih, ROOT.TH1F) else ih.get('nominal')
            if hist is None:
                continue
            if data_obs is None:
                data_obs = hist.Clone('data_obs')
                data_obs.SetTitle('data_obs')
            else:
                data_obs.Add(hist)
        if data_obs is not None:
            root_hists[bname]['data_obs'] = data_obs

    root_file = ROOT.TFile(output, 'recreate')
    for bname in root_hists.keys():
        root_file.cd()
        try:
            directory = root_file.Get(bname)
            directory.IsZombie()
        except ReferenceError:
            directory = root_file.mkdir(bname)
        root_file.cd(bname)
        for ih_name, ih in root_hists[bname].items():
            if isinstance(ih, dict):
                for _, ih2 in ih.items():
                    ih2.Write()
            else:
                ih.Write()
    root_file.Close()
    logging.info(f"\n File {output} created.")

    # Build one datacard per combine bin
    bkg_labels = [metadata['processes']['background'][p]['label']
                  for p in metadata['processes']['background']]
    sig_labels = [metadata['processes']['signal'][p]['label']
                  for p in metadata['processes']['signal']]

    for i, ibin in enumerate(bin_names):
        cb = ch.CombineHarvester()
        cb.SetVerbosity(3)

        cats = [(i, ibin)]
        cb.AddObservations(['*'], [''], ['13p6TeV'], ['*'], cats)
        cb.AddProcesses(['*'], [''], ['13p6TeV'], ['*'], bkg_labels, cats, False)
        cb.AddProcesses(['*'], [''], ['13p6TeV'], ['*'], sig_labels, cats, True)

        if stat_only:
            cb.cp().backgrounds().ExtractShapes(output, '$BIN/$PROCESS', '')
            cb.cp().signals().ExtractShapes(output, '$BIN/$PROCESS', '')
        else:
            btagSysts, psfsrSysts, othersSysts = [], [], []
            for nuisance in mcSysts:
                era_tags = ['2022_preEE', '2022_EE', '2023_preBPix', '2023_BPix']
                era_match = next((e for e in era_tags if e in nuisance), None)
                if era_match and era_match not in ibin:
                    continue
                if era_match:
                    cb.cp().signals().AddSyst(
                        cb, nuisance, 'shape', ch.SystMap('bin')([ibin], 1.0))
                else:
                    cb.cp().signals().AddSyst(cb, nuisance, 'shape', ch.SystMap()(1.0))
                if 'btag' in nuisance:
                    btagSysts.append(nuisance)
                elif 'ps_fsr' in nuisance:
                    psfsrSysts.append(nuisance)
                else:
                    othersSysts.append(nuisance)
            cb.SetGroup("ps_fsr", psfsrSysts)
            cb.SetGroup("btag",   btagSysts)

            for isyst in metadata['uncertainty']:
                othersSysts.append(isyst)
                syst_years = metadata['uncertainty'][isyst]['years']
                # Match year by substring (e.g. 'y2022' in bin name 'hadW_mu_y2022')
                year_tag = next((yt for yt in syst_years if yt in ibin), None)
                if year_tag is None:
                    continue
                cb.cp().signals().AddSyst(
                    cb, isyst, metadata['uncertainty'][isyst]['type'],
                    ch.SystMap('bin')([ibin], syst_years[year_tag]))
                cb.cp().backgrounds().AddSyst(
                    cb, isyst, metadata['uncertainty'][isyst]['type'],
                    ch.SystMap('bin')([ibin], syst_years[year_tag]))

            cb.SetGroup("others", othersSysts)
            cb.SetGroup("signal_norm_xsbr", ['pdf_Higgs_ggHH', 'BR_hbb', 'BR_hww'])
            cb.SetGroup("signal_norm_xs",   ['pdf_Higgs_ggHH'])

            cb.cp().backgrounds().ExtractShapes(
                output, '$BIN/$PROCESS', '$BIN/$PROCESS_$SYSTEMATIC')
            cb.cp().signals().ExtractShapes(
                output, '$BIN/$PROCESS', '$BIN/$PROCESS_$SYSTEMATIC')
            cb.cp().SetAutoMCStats(cb, 0, 1, 1)

        cb.PrintAll()
        cb.WriteDatacard(f"{output_dir}/datacard_{ibin}.txt",
                         f"{output_dir}/{ibin}_{output_file}")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Create combine inputs for bbWW',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-o', '--output_dir', dest="output_dir",
                        default="./datacards/", help='Output directory.')
    parser.add_argument('--var', dest="variable",
                        default="SvB.phh", help='Variable to make histograms.')
    parser.add_argument('-f', '--file', dest='file_to_convert',
                        default="histos/histAll.json", help="File with coffea hists")
    parser.add_argument('-r', '--rebin', dest='rebin', type=int,
                        default=1, help="Rebin factor")
    parser.add_argument('-s', '--syst_file', dest='systematics_file',
                        default='', help="File with systematic variations")
    parser.add_argument('-m', '--metadata', dest='metadata',
                        default='bbreww/stats_analysis/metadata/bbWW.yml',
                        help="Metadata yaml with processes, bins, uncertainties")
    parser.add_argument('--flavors', dest='flavors', nargs='+',
                        default=['e', 'mu'],
                        help='Lepton flavor axis values to use (one bin per flavor × year).')
    parser.add_argument('--stat_only', dest='stat_only', action="store_true",
                        default=False, help="Create stat-only inputs (no shape systematics)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logging.info("\nRunning with these parameters: ")
    logging.info(args)

    create_combine_root_file(
        args.file_to_convert,
        args.rebin,
        args.variable,
        args.output_dir,
        args.systematics_file,
        args.flavors,
        metadata_file=args.metadata,
        stat_only=args.stat_only,
    )
