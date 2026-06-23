# bbreww uses the generic converter from src/tools.
# Run from the barista root:
#
#   ./run_container python src/tools/convert_coffea_to_json.py \
#       -i bbreww/analysis/hists/histAll.coffea \
#       -o bbreww/stats_analysis/histos/histAll.json \
#       --histos SvB.phh
#
# bbreww histograms have axes: process, year, channel, flavor, <Boolean cut axes>
# Boolean axes (e.g. preselection cuts) are summed over by default.
# Use --select to fix any of them, e.g. --select passPreSel=True

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.tools.convert_coffea_to_json import convert_histogram, hist_to_json  # noqa: F401

if __name__ == '__main__':
    # Delegate entirely to the generic tool
    import runpy
    runpy.run_module('src.tools.convert_coffea_to_json', run_name='__main__')
