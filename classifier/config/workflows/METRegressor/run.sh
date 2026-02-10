# change these vars #
export LPCUSER="akhanal"
export CERNUSER="a/akhanal"
export BASE="root://cmseos.fnal.gov//store/user/${LPCUSER}/HHbbWW_MET_regressor"
export MODEL="${BASE}/classifier/Regressor"
export MET_FRIEND="${BASE}/friend/met_regressor/"
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

# train with train.yml and common.yml configs
./src/pyml.py \
    template "{model: ${MODEL}}" $WFS/train.yml \
    -from $WFS/common.yml \
    -setting Monitor "address: :${port}" \
    -flag debug # use debug flag

# TODO: add regression-specific analysis (loss curves, resolution plots)
# ./src/pyml.py analyze \
#     --results ${MODEL}/result.json \
#     -analysis ... \
#     -setting IO "output: ${PLOT}" \
#     -setting Monitor "address: :${port}"

# evaluate with evaluate.yml and common.yml configs
./src/pyml.py \
    template "{model: ${MODEL}, SvB: ${MET_FRIEND}}" $WFS/evaluate.yml \
    -from $WFS/common.yml \
    -setting Monitor "address: :${port}"
