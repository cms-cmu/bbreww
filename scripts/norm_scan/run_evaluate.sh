# Evaluate only (split from run.sh for orchestrator parallelization).
# Reads the trained model from ${MODEL}/SvB/ and writes friend trees to ${SvB}/.

# change these vars #
export LPCUSER="akhanal"
export CERNUSER="a/akhanal"
export BASE="root://cmseos.fnal.gov//store/user/${LPCUSER}/HHbbWW_classifier_lowpt"
export MODEL="${BASE}/classifier/bbWWBase/SvB/"
export FvT="${BASE}/friend/FvT/"
export SvB="${BASE}/friend/SvB/"
export PLOT="root://eosuser.cern.ch//eos/user/${CERNUSER}/HHbbWW_classifier_lowpt/"
export CLASSIFIER_CONFIG_PATHS="bbreww"
#####################
export WFS="bbreww/classifier/config/workflows/bbWW_lowpt/svb"

if [ -z "$1" ]; then
    port=10201
else
    port=$1
fi

# evaluate with evaluate.yml and common.yml configs
python -m src.classifier.task.main \
    template "{model: ${MODEL}, SvB: ${SvB}}" $WFS/evaluate.yml \
    -from $WFS/common.yml \
    -setting Monitor "address: :${port}"
