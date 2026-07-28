# change these vars #
export LPCUSER="akhanal"
export CERNUSER="a/akhanal"
export BASE="root://cmseos.fnal.gov//store/user/${LPCUSER}/HHbbWW_MET_regressor_test"
export MODEL="${BASE}/classifier/Regressor"
export MET_FRIEND="${BASE}/friend/met_regressor/"
export MET_FRIEND_3JET="${BASE}/friend/met_regressor_3jet/"
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
