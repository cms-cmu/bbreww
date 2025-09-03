# bbreww

This is a temporary repository for bbreww analysis code, which is implemented on top of the coffea4bees framework. The code is under development.

## Installation

To run the analysis, you need to have the `coffea4bees` framework installed. This can be done by cloning the repository and installing the required dependencies.
To install the necessary dependencies, run:

```bash
git clone ssh://git@gitlab.cern.ch:7999/cms-cmu/coffea4bees.git
cd coffea4bees/
git clone ssh://git@gitlab.cern.ch:7999/cms-cmu/bbreww.git
```

## Contributing

If you want to contribute, please make a merge request by pushing your changes to branch different from `master` and then creating a merge request.

## Gitlab CI tests

The repository includes a Gitlab CI configuration file (`.gitlab-ci.yml`) that runs tests on the code. These tests are run on every commit and merge request to ensure that the code is working correctly.

To run the tests locally, you can use the snakemake workflow defined in `bbreww/workflows/Snakemake_CI`. You can run the tests by executing the following command:

```bash
./run_container snakemake -s bbreww/workflows/Snakemake_CI --use-apptainer XXXXX
```

Replace `XXXXX` with the name of the job you want to run from the gitlab CI configuration file. For example, to run the `analysis_test` job, you can use:

```bash
./run_container snakemake -s bbreww/workflows/Snakemake_CI --use-apptainer analysis_test
```

The output of the tests will be saved in the `CI_output/` directory.
