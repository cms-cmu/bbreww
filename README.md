# NO_NAME MC Production branch

**THIS BRANCH IS USED FOR NANOAOD PRODUCTION**. If you need the analysis code, please switch to the master branch.

## Setup Instructions

This branch must be included within a CMSSW release. Follow the steps below to set up the environment:

```bash
cmsrel CMSSW_15_0_2
cd CMSSW_15_0_2/src
cmsenv
git cms-init
git cms-merge-topic nurfikri89:from1502_NanoV15_RunIIULMiniAODv2_puppiRecluster
scram b -j4
cmsenv
git clone ssh://git@gitlab.cern.ch:7999/cms-cmu/my-awesome-project.git -b nano_prod MCProduction
cd MCProduction/
```

## Producing Private NanoAOD Datasets

To produce a nanoAOD dataset, you need to execute a Python configuration file tailored to your specific requirements. These configuration files are named `Prod_NanoAOD_**.py` in this branch. To test a configuration file, run the following command:

```bash
cmsRun Prod_NanoAOD_MCUL18.py
```

- For simplicity, and to make sure these files contain the correct configurations, these files are provided in this branch. If you want to know more about how to produce this files, please refer to [section below](#how-to-produce-config-files).
- NanoAOD production requires miniAOD datasets as input. These datasets are typically stored on `T2` sites worldwide. To process them, jobs must be submitted to the grid using CRAB.
- If you are new to CRAB, refer to the [official documentation](https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideCrab#Documentation_for_beginner_users). The [CRAB3 Tutorial](https://twiki.cern.ch/twiki/bin/view/CMSPublic/CRAB3Tutorial) is particularly helpful for setting up your environment and understanding the basics of job submission.

### Automating Job Submission with `crabby.py`

Managing a full production campaign can be complex. To simplify this process, you can use the `crabby.py` script, which allows you to submit multiple CRAB jobs simultaneously by specifying common configurations and datasets in a YAML file. For more details, visit the [crabby.py documentation](https://github.com/cms-btv-pog/btvnano-prod/tree/NanoAODv12_22Sep2023?tab=readme-ov-file).

In this branch, YAML configuration files are located in the [config_yml](./config_yml/) directory. Open one of these files to find detailed instructions on what needs to be modified for your specific use case. Then:

```bash
# This disable publish config and produce single file per dataset
python3 crabby.py -c config_yml/XXX --make --submit --test True
# Only generate configuration
python3 crabby.py -c config_yml/XXX --make 
# make the submit python file and submit
python3 crabby.py -c config_yml/XXX --make --submit 
```

**It is always a good idea to submit a test job**. If it is the first job submission, or some configuration changed, please submit a test job first.

### How to monitor the jobs

To check the individual status of the jobs, one can use directly the crab commands. For example:

```bash
crab status -d JMEnano/NAME_OF_THE_FOLDER_CONTAINING_THE_CRAB_JOB/
```

This command will spit some instructions and useful links on where to check the status of the jobs. You can access the online dashboard (Grafana) to monitor multiple datasets at the time.

There are some tools to monitor crab jobs in your command line, like [mrCrabs](https://github.com/JanvanderLinden/mrCrabs/tree/main). At the end of the day, you can use whatever makes your life easier.

## How to keep track of the datasets produced

We will need to produce several datasets, and therefore we need a way to keep track of them. For that we can use this [google sheet](https://docs.google.com/spreadsheets/d/1UTGNfjYcZp62a7J32YEW9eH6r4VG_mCbCSJGU4Y-bI8/edit?usp=sharing). 


## How to produce config files

We can produce the python config files in CMS with the help of the `cmsDriver.py` command. 

* An example of MC:
```

cmsDriver.py \
--eventcontent NANOAODSIM \
--customise Configuration/DataProcessing/Utils.addMonitoring \
--datatier NANOAODSIM \
--conditions auto:phase1_2018_realistic \
--step NANO \
--scenario pp \
--era Run2_2018,run2_nanoAOD_106Xv2 \
--python_filename Prod_NanoAOD_MCUL18.py \
--filein /store/mc/RunIISummer20UL18MiniAODv2/QCD_Pt-15to7000_TuneCP5_Flat2018_13TeV_pythia8/MINIAODSIM/106X_upgrade2018_realistic_v16_L1v1-v2/2520000/D089DD44-7E7D-954B-9ACA-4412C447146D.root \
--fileout file:NanoAOD_MCUL18_QCD_Flat2018.root \
--no_exec \
--mc \
-n 100
```

* An example for data:
```

cmsDriver.py \
--eventcontent NANOAOD \
--customise Configuration/DataProcessing/Utils.addMonitoring \
--datatier NANOAOD \
--conditions auto:run2_data \
--step NANO \
--scenario pp \
--era Run2_2018,run2_nanoAOD_106Xv2 \
--python_filename Prod_NanoAOD_DataUL18.py \
--filein /store/data/Run2018D/JetHT/MINIAOD/UL2018_MiniAODv2_GT36-v1/2820000/52A33486-2E2D-7746-841D-BFF94C975226.root \
--fileout file:NanoAOD_DataUL18_EraD_JetHT.root \
--no_exec \
--data \
-n 100
```