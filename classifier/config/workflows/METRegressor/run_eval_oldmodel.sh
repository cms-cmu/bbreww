# Cross-check: evaluate the OLD (frozen baseline) model over the NEW (post-JEC-update)
# input friends. MODEL points at the old checkpoint area (read-only); the eval friend
# trees are written to the _test area, overwriting whatever is there — back it up first.
# change these vars #
export LPCUSER="akhanal"
export CERNUSER="a/akhanal"
export MODEL="root://cmseos.fnal.gov//store/user/${LPCUSER}/HHbbWW_MET_regressor/classifier/Regressor"
export MET_FRIEND="root://cmseos.fnal.gov//store/user/${LPCUSER}/HHbbWW_MET_regressor_test/friend/met_regressor/"
export PLOT="root://eosuser.cern.ch//eos/user/${CERNUSER}/HHbbWW_MET_regressor/"
export CLASSIFIER_CONFIG_PATHS="bbreww"
#####################
export WFS="bbreww/classifier/config/workflows/METRegressor"

# the first argument can be a port
if [ -z "$1" ]; then
    port=10201
else
    port=$1
fi

# evaluate 4-jet region (default) with evaluate.yml and common.yml configs
./src/pyml.py \
    template "{model: ${MODEL}, SvB: ${MET_FRIEND}}" $WFS/evaluate.yml \
    -from $WFS/common.yml \
    -setting Monitor "address: :${port}"
