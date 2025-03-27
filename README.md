# NO_NAME MC Production branch

**THIS BRANCH IS USED FOR NANOAOD PRODUCTION**, if you need the analysis code go to the master branch.

## Setup

This branch requires to be included inside a CMSSW release. Therefore follow the steps below:

```
cmsrel CMSSW_15_0_2
cd CMSSW_15_0_2/src
cmsenv
git cms-init
git cms-merge-topic nurfikri89:from1502_NanoV15_RunIIULMiniAODv2_puppiRecluster
scram b -j4
```