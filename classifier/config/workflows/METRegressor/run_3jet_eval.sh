# change these vars #
export LPCUSER="akhanal"
export CERNUSER="a/akhanal"
export BASE="root://cmseos.fnal.gov//store/user/${LPCUSER}/HHbbWW_MET_regressor"
export MODEL="${BASE}/classifier/Regressor"
export MET_FRIEND="${BASE}/friend/met_regressor/"
export MET_FRIEND_3JET="${BASE}/friend/met_regressor_3jet/"
export PLOT="root://eosuser.cern.ch//eos/user/${CERNUSER}/HHbbWW_MET_regressor/"
export CLASSIFIER_CONFIG_PATHS="bbreww"
#####################
export WFS="bbreww/classifier/config/workflows/METRegressor"

# the first argument can be a port
if [ -z "$1" ]; then
    port=10200
else
    port=$1
fi

# evaluate 3-jet region with evaluate_3jet.yml; outputs go to a separate
# friend-tree directory (MET_FRIEND_3JET) to avoid overwriting 4-jet outputs
./src/pyml.py \
    template "{model: ${MODEL}, SvB: ${MET_FRIEND_3JET}}" $WFS/evaluate_3jet.yml \
    -from $WFS/common.yml \
    -setting Monitor "address: :${port}"
