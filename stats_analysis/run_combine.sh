#!/bin/bash

# Function to handle argument parsing
parse_arguments() {

  # Check if folder argument is provided
  if [ -z "$1" ]; then
    echo "Missing folder argument"
  fi
  datacard_folder="$1"

  # Set defaults for flags
  limits=false
  impacts=false
  postfit=false
  systbreakdown=false
  unblind=false
  gof=false

  # Process arguments
  while [[ $# -gt 1 ]]; do
    case "$2" in
      --limits)
        limits=true
        shift
        ;;
      --impacts)
        impacts=true
        shift
        ;;
      --postfit)
        postfit=true
        shift
        ;;
      --systbreakdown)
        systbreakdown=true
        shift
        ;;
      --unblind)
        unblind=true
        shift
        ;;
      --gof)
        gof=true
        shift
        ;;
      *)
        echo "Invalid argument: '$2'"
        ;;
    esac
  done
}
# Parse arguments
parse_arguments "$@"

echo "Running combine script with arguments: $@"

currentDir=$PWD
signallabel="ggHH_kl_1_kt_1_13p6TeV_hbbhww"

if [ "$unblind" = true ]; then
    echo "Running in unblind mode"
    blind_label="_unblinded"
    limit_blind=""
    asymov_data=""
    signal_parameter=""
    freeze_parameters=""
else
    echo "Running in blind mode"
    blind_label=""
    limit_blind="--run blind"
    asymov_data="-t -1"
    signal_parameter="r${signallabel}=1,"
    freeze_parameters="r${signallabel},"
fi

other_signals="rggHH_kl_0_kt_1_13p6TeV_hbbhww=0,rggHH_kl_2p45_kt_1_13p6TeV_hbbhww=0,rggHH_kl_5_kt_1_13p6TeV_hbbhww=0"
freeze_other="rggHH_kl_0_kt_1_13p6TeV_hbbhww,rggHH_kl_2p45_kt_1_13p6TeV_hbbhww,rggHH_kl_5_kt_1_13p6TeV_hbbhww"


run_limits() {
  local datacard=$1
  local signallabel=$2
  local iclass=$3

    text2workspace.py ${datacard}.txt \
        -P HiggsAnalysis.CombinedLimit.PhysicsModel:multiSignalModel --PO verbose \
        --PO "map=.*/${signallabel}:r${signallabel}[1,-10,10]" \
        --PO "map=.*/ggHH_kl_0_kt_1_13p6TeV_hbbhww:rggHH_kl_0_kt_1_13p6TeV_hbbhww[1,-10,10]" \
        --PO "map=.*/ggHH_kl_2p45_kt_1_13p6TeV_hbbhww:rggHH_kl_2p45_kt_1_13p6TeV_hbbhww[1,-10,10]" \
        --PO "map=.*/ggHH_kl_5_kt_1_13p6TeV_hbbhww:rggHH_kl_5_kt_1_13p6TeV_hbbhww[1,-10,10]"

    combine -M AsymptoticLimits ${datacard}.root --redefineSignalPOIs r${signallabel} \
        -n _${iclass}${blind_label} ${limit_blind} \
        --setParameters ${other_signals} \
        --freezeParameters ${freeze_other} \
        > limits_${datacard}_${iclass}${blind_label}.txt
    cat limits_${datacard}_${iclass}${blind_label}.txt
    combineTool.py -M CollectLimits higgsCombine_${iclass}${blind_label}.AsymptoticLimits.mH120.root -o limits_${datacard}_${iclass}${blind_label}.json

    combine -M Significance ${datacard}.root --redefineSignalPOIs r${signallabel} \
        -n _${iclass}${blind_label} ${asymov_data} \
        --setParameters ${other_signals} \
        --freezeParameters ${freeze_other} \
        > significance_${datacard}_${iclass}${blind_label}.txt
    cat significance_${datacard}_${iclass}${blind_label}.txt

}

for iclass in SvB;
do
    datacard="datacard"
    cd ${datacard_folder}/

    if [ "$limits" = true ]; then

        combineCards.py \
            y2022=datacard_y2022.txt \
            y2023=datacard_y2023.txt \
            > ${datacard}.txt
        run_limits $datacard $signallabel $iclass

        # Per-year limits (uncomment as needed):
        # run_limits datacard_y2022 $signallabel $iclass
        # run_limits datacard_y2023 $signallabel $iclass

    elif [ "$impacts" = true ]; then

        if [ -f "${datacard}.root" ]; then

            combineTool.py -M Impacts -d ${datacard}.root --doInitialFit \
            --robustFit 1 -n ${iclass} -m 125 ${asymov_data} \
            --setParameterRanges r${signallabel}=-10,10:rggHH_kl_0_kt_1_13p6TeV_hbbhww=0,0:rggHH_kl_2p45_kt_1_13p6TeV_hbbhww=0,0:rggHH_kl_5_kt_1_13p6TeV_hbbhww=0,0 \
            --setParameters ${signal_parameter}${other_signals}

            combineTool.py -M Impacts -d ${datacard}.root --doFits \
            --robustFit 1 -m 125 --parallel 4 -n ${iclass} ${asymov_data} \
            --setParameterRanges r${signallabel}=-10,10:rggHH_kl_0_kt_1_13p6TeV_hbbhww=0,0:rggHH_kl_2p45_kt_1_13p6TeV_hbbhww=0,0:rggHH_kl_5_kt_1_13p6TeV_hbbhww=0,0 \
            --setParameters ${signal_parameter}${other_signals}

            combineTool.py -M Impacts -d ${datacard}.root -o impacts_combine_${iclass}_exp.json -m 125 -n ${iclass}

            if [[ ! -d "${currentDir}/stats_analysis" ]]; then
                tmpDir=${currentDir}/bbreww/stats_analysis
            else
                tmpDir=${currentDir}/stats_analysis
            fi
            plotImpacts.py -i impacts_combine_${iclass}_exp.json -o impacts_combine_${iclass}_exp_HH --POI r${signallabel} --per-page 20 --left-margin 0.3 --height 400 --label-size 0.04 --translate ${tmpDir}/nuisance_names.json --blind
            mkdir -p impacts/
            mv higgsCombine*Fit* impacts/

        else
            echo "File ${datacard}.root does not exist."
        fi

    elif [ "$gof" = true ]; then

        if [ -f "${datacard}.root" ]; then

            echo "Running goodness of fit test on data"
            combine -M GoodnessOfFit ${datacard}.root --algo saturated \
                -n _${iclass}${blind_label}_gof_data \
                --setParameters ${signal_parameter}${other_signals} \
                > gof_data_${datacard}_${iclass}${blind_label}.txt
            cat gof_data_${datacard}_${iclass}${blind_label}.txt

            combine -M GoodnessOfFit ${datacard}.root --algo saturated \
                -n _${iclass}${blind_label}_gof_toys --toysFrequentist -t 1000 \
                > gof_toys_${datacard}_${iclass}${blind_label}.txt
            cat gof_toys_${datacard}_${iclass}${blind_label}.txt

            combineTool.py -M CollectGoodnessOfFit \
                --input higgsCombine_${iclass}${blind_label}_gof_data.GoodnessOfFit.mH120.root \
                higgsCombine_${iclass}${blind_label}_gof_toys.GoodnessOfFit.mH120.123456.root \
                -o gof_${datacard}_${iclass}${blind_label}.json

            plotGof.py gof_${datacard}_${iclass}${blind_label}.json \
                --statistic saturated --mass 120.0 \
                --output gof_${datacard}_${iclass}${blind_label}

        else
            echo "File ${datacard}.root does not exist."
        fi

    elif [ "$postfit" = true ]; then

        if [ -f "${datacard}.root" ]; then

            echo "Running postfit b-only"
            combine -M FitDiagnostics ${datacard}.root --redefineSignalPOIs r${signallabel} \
                -n _${iclass}${blind_label}_prefit_bonly ${asymov_data} \
                --setParameters r${signallabel}=0,${other_signals} \
                --freezeParameters ${freeze_other} \
                > fitDiagnostics_${datacard}_${iclass}${blind_label}_prefit_bonly.txt
            cat fitDiagnostics_${datacard}_${iclass}${blind_label}_prefit_bonly.txt

            python /home/cmsusr/CMSSW_11_3_4/src/HiggsAnalysis/CombinedLimit/test/diffNuisances.py \
                -p r${signallabel} \
                -a fitDiagnostics_${iclass}${blind_label}_prefit_bonly.root \
                -g diffNuisances_${datacard}_${iclass}${blind_label}_prefit_bonly.root

            echo "Running postfit s+b"
            combine -M FitDiagnostics ${datacard}.root --redefineSignalPOIs r${signallabel} \
                -n _${iclass}${blind_label}_prefit_sb ${asymov_data} --saveShapes --saveWithUncertainties --plots \
                --setParameters ${signal_parameter}${other_signals} \
                --freezeParameters ${freeze_other} \
                > fitDiagnostics_${datacard}_${iclass}${blind_label}_prefit_sb.txt
            cat fitDiagnostics_${datacard}_${iclass}${blind_label}_prefit_sb.txt
            mkdir -p fitDiagnostics_sb/
            mv *th1x* fitDiagnostics_sb/
            mv covariance* fitDiagnostics_sb/

            python /home/cmsusr/CMSSW_11_3_4/src/HiggsAnalysis/CombinedLimit/test/diffNuisances.py \
                -p r${signallabel} \
                -a fitDiagnostics_${iclass}${blind_label}_prefit_sb.root \
                -g diffNuisances_${datacard}_${iclass}${blind_label}_prefit_sb.root

        else
            echo "File ${datacard}.root does not exist."
        fi

    elif [ "$systbreakdown" = true ]; then

        if [ -f "${datacard}.root" ]; then

            rMin=-20
            rMax=10
            points=50

            combine -M MultiDimFit -n _${iclass}_systbreakdown_postfit \
                --saveWorkspace -d ${datacard}.root --robustFit 1 ${asymov_data} \
                --setParameters ${signal_parameter}${other_signals} \
                --freezeParameters ${freeze_parameters}${freeze_other}

            combine -M MultiDimFit -n _${iclass}_systbreakdown_freeze_all \
                -P r${signallabel} ${asymov_data} --snapshotName MultiDimFit \
                --rMin ${rMin} --rMax ${rMax} --algo grid --points ${points} --alignEdges 1 \
                --setParameters ${signal_parameter}${other_signals} \
                --freezeParameters ${freeze_parameters}${freeze_other},allConstrainedNuisances \
                -d higgsCombine_${iclass}_systbreakdown_postfit.MultiDimFit.mH120.root

            scan_cmd="combine -M MultiDimFit \
                -P r${signallabel} ${asymov_data} --snapshotName MultiDimFit \
                --rMin ${rMin} --rMax ${rMax} --algo grid --points ${points} --alignEdges 1 \
                --setParameters ${signal_parameter}${other_signals} \
                --freezeParameters ${freeze_parameters}${freeze_other} \
                -d higgsCombine_${iclass}_systbreakdown_postfit.MultiDimFit.mH120.root"

            ${scan_cmd} -n _${iclass}_systbreakdown_total
            ${scan_cmd} -n _${iclass}_systbreakdown_freeze_btag --freezeNuisanceGroups btag
            ${scan_cmd} -n _${iclass}_systbreakdown_freeze_ps_fsr --freezeNuisanceGroups btag,ps_fsr

            plot1DScan.py higgsCombine_${iclass}_systbreakdown_total.MultiDimFit.mH120.root \
                --main-label "Total uncert." --others \
                higgsCombine_${iclass}_systbreakdown_freeze_btag.MultiDimFit.mH120.root:"b-tagging":4 \
                higgsCombine_${iclass}_systbreakdown_freeze_ps_fsr.MultiDimFit.mH120.root:"PS FSR":3 \
                higgsCombine_${iclass}_systbreakdown_freeze_all.MultiDimFit.mH120.root:"Stat. only":5 \
                --breakdown "btag,ps_fsr,others,stat" \
                --POI r${signallabel} -o systbreakdown_${iclass}_breakdown

            mkdir -p syst_breakdown/
            mv *_systbreakdown* syst_breakdown/

        else
            echo "File ${datacard}.root does not exist."
        fi
    fi

    cd $currentDir

done
