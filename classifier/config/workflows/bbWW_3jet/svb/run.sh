# change these vars #
export LPCUSER="akhanal"
export CERNUSER="a/akhanal"
export BASE="root://cmseos.fnal.gov//store/user/${LPCUSER}/ML/HHbbWW_classifier_3jet"
export MODEL="${BASE}/classifier/bbWW_3jet/SvB/"
export FvT="${BASE}/friend/FvT/"
export SvB="${BASE}/friend/SvB/"
export PLOT="root://eosuser.cern.ch//eos/user/${CERNUSER}/HHbbWW_classifier_3jet/"
export CLASSIFIER_CONFIG_PATHS="bbreww"
#####################
export WFS="bbreww/classifier/config/workflows/bbWW_3jet/svb"

# the first argument can be a port
if [ -z "$1" ]; then
    port=10201
else
    port=$1
fi

# train with train.yml and common.yml configs
./src/pyml.py \
    template "{model: ${MODEL}, FvT: ${FvT}}" $WFS/train.yml \
    -from $WFS/common.yml \
    -setting Monitor "address: :${port}" \
    -flag debug

# evaluate with evaluate.yml and common.yml configs
./src/pyml.py \
    template "{model: ${MODEL}, SvB: ${SvB}}" $WFS/evaluate.yml \
    -from $WFS/common.yml \
    -setting Monitor "address: :${port}"
