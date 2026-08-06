# change these vars #
export LPCUSER="akhanal"
export CERNUSER="a/akhanal"
export BASE="root://cmseos.fnal.gov//store/user/${LPCUSER}/HHbbWW_classifier_lowpt"
export MODEL="${BASE}/classifier/bbWWBase/dvtt/"
export DvTT="${BASE}/friend/dvtt/"
export PLOT="root://eosuser.cern.ch//eos/user/${CERNUSER}/HHbbWW_classifier_v1/dvtt/"
#####################

export CLASSIFIER_CONFIG_PATHS="bbreww" 
export WFS="bbreww/classifier/config/workflows/bbWW_lowpt/dvtt"

# the first argument can be a port
if [ -z "$1" ]; then
    port=10200
else
    port=$1
fi

# train with train.yml and common.yml configs
./src/pyml.py \
    template "model: ${MODEL}" $WFS/train.yml \
    -from $WFS/common.yml \
    -setting Monitor "address: :${port}" \
    -flag debug # use debug flag

# plot the AUC and ROC
./src/pyml.py analyze \
    --results ${MODEL}/result.json \
    -analysis bbWW.LossROC \
    -setting IO "output: ${PLOT}" \
    -setting IO "report: FvT" \
    -setting Monitor "address: :${port}"

# evaluate with evaluate.yml and common.yml configs
./src/pyml.py \
    template "{model: ${MODEL}, DvTT: ${DvTT}}" $WFS/evaluate.yml \
    -from $WFS/common.yml \
    -setting Monitor "address: :${port}"
