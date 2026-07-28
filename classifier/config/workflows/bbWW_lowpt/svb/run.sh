# change these vars #
export LPCUSER="akhanal"
export CERNUSER="a/akhanal"
# CAMPAIGN iter1_dD16: outputs go to the campaign EOS area so the production
# model/friends under HHbbWW_classifier_lowpt/ are never touched.
export ITER="iter2_dD20"
export BASE="root://cmseos.fnal.gov//store/user/${LPCUSER}/HHbbWW_classifier_lowpt_campaign/${ITER}"
export MODEL="${BASE}/classifier/bbWWBase/SvB/"
export FvT="root://cmseos.fnal.gov//store/user/${LPCUSER}/HHbbWW_classifier_lowpt/friend/FvT/"
export SvB="${BASE}/friend/SvB_lowpt/"
export PLOT="root://eosuser.cern.ch//eos/user/${CERNUSER}/HHbbWW_classifier_lowpt/"
export CLASSIFIER_CONFIG_PATHS="bbreww" 
#####################
export WFS="bbreww/classifier/config/workflows/bbWW_lowpt/svb"

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
    -flag debug # use debug flag

# plot the AUC and ROC
# CAMPAIGN: skipped — plots go to CERN EOS (separate auth) and are not used for
# the keep/revert decision, which is based only on the final combine limits.
# ./src/pyml.py analyze \
#     --results ${MODEL}/result.json \
#     -analysis bbWW.LossROC \
#     -setting IO "output: ${PLOT}" \
#     -setting IO "report: FvT" \
#     -setting Monitor "address: :${port}"

# evaluate with evaluate.yml and common.yml configs
# CAMPAIGN iter2: TRAIN-ONLY submission — evaluation deferred until the
# processor pipeline is not running (evaluate's EOS I/O competes with the
# processors). Re-enable and resubmit for the evaluate step.
# ./src/pyml.py \
#     template "{model: ${MODEL}, SvB: ${SvB}}" $WFS/evaluate.yml \
#     -from $WFS/common.yml \
#     -setting Monitor "address: :${port}"
