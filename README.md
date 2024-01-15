# vlm_benchmark
Code for benchmarking contrastive VLMs on zero and few-shot activity recognition. Setup instructions can be found in `SETUP.md`.

The main entry point for our experiments is in `hyperparam_search.py`. We launch a run with a given VLM, classifier, dataset and number of shots with:
`python hyperparam_search.py <VLM> <classifier> --dataset <dataset> --n_shots <number_of_shots>`.  

By default for all classifiers and datasets, we find the optimal set of hyperparameters for the validation set, and report results from a fresh run of the same hyperparameters on the test set to avoid spurious results from testing various hyperparameter settings for our methods.  We include our explicit hyperparameter tuning process in our codebase for transparency and reproducbility, and note that we only search over a small number of hyperparameters (<5 for each classifier) for fair comparison.  We use default hyperparameter settings from papers and codebases when available.

This README has been modified to remove some details to comply with double-blind review policies. Upon acceptance, we will revert some aspects of this README to its original state. 
