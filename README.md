# vlm_benchmark
Code for benchmarking contrastive VLMs on zero and few-shot activity recognition. This codebase is used for evaluating methods in [our paper](https://arxiv.org/abs/2406.01662): `Few-Shot Classification of Interactive
Activities of Daily Living (InteractADL)`.  

## InteractADL Dataset
For information on how to download InteractADL, please see the [official InteractADL page](https://homeactiongenome.org/interactadl.html).  

## Other datasets
The setup instructions for our method evaluation suite (this codebase) and other activity recognition datasets can be found in `SETUP.md`.

## Issues
If you create a GitHub issue in this repo, we will do our best to help you resolve it!

## Information

The main entry point for our experiments is in `hyperparam_search.py`. We launch a run with a given VLM, classifier, dataset and number of shots with:
`python hyperparam_search.py <VLM> <classifier> --dataset <dataset> --n_shots <number_of_shots>`. This is how we report all results in our paper, unless specified otherwise.

By default for all classifiers and datasets, we find the optimal set of hyperparameters for the validation set, and report results from a fresh run of the same hyperparameters on the test set to avoid spurious results from testing various hyperparameter settings for our methods. We include our explicit hyperparameter tuning process in our codebase for transparency and reproducbility, and note that we only search over a small number of hyperparameters (<5 for each classifier) for fair comparison. We use default hyperparameter settings from papers and codebases whenever they are available.
