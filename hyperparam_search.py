import numpy as np
import os, sys
import importlib
from tqdm.autonotebook import tqdm, trange
import pandas as pd
import json
import itertools
import skopt

from FewShotTestHandler import FewShotTestHandler, optimize_hyperparameters, find_hyperparameters, test_already_stored, filter_test_results
from dataset import DatasetHandler
from similarity_metrics import Similarity
from plotting_utils import plot

VLM_ARG = sys.argv[1]
CLASSIFIER_ARG = sys.argv[2]

N_HYPERPARAM_SEARCH_CALLS = 64 # Max number of hyperparam values tested for each dataset/n_shot combo
SEARCH_METHOD = "grid" # gp, forest, random
USE_VAL_TUNING = True


'''
Test Setup
'''
# Parameters which will be iterated over.
# Each product will receive individual hyperparam optimization over vlm and classifier params
test_params_dict = {}

# Dataset Params - dataset.____ keys are passed into DatasetHandler constructor
test_params_dict["dataset.name"] = ["smsm"]#, "kinetics_100", "moma_act", "moma_sact"]
test_params_dict["dataset.split_type"] = ["video"]

# Few-Shot Test Params - test.____ keys are passed into few-shot test call
test_params_dict["test.n_way"] = [None] # None value gets manually converted to the max size for each dataset
test_params_dict["test.n_support"] = [1, 2, 4, 8, 16]
test_params_dict["test.n_query"] = [None]
test_params_dict["test.n_episodes"] = [4]



'''
VLM Setup
'''
fixed_vlm_kwargs = {} # VLM keyword parameters to pass to constructor
vlm_hyperparams = [] # Hyperparameter spaces in skopt format
if VLM_ARG == "clip":
    from CLIP.CLIPVLM import ClipVLM as VLM
    fixed_vlm_kwargs["num_frames"] = 10
elif VLM_ARG == "miles":
    from MILES.wrapper import MILES_SimilarityVLM as VLM
elif VLM_ARG == "videoclip":
    from video_clip.video_clip import VideoClipVLM as VLM
    fixed_vlm_kwargs["num_seconds"] = 4
    fixed_vlm_kwargs["sample_strat"] = "spread"
    fixed_vlm_kwargs["use_cuda"] = True
elif VLM_ARG == "univl":
    from UNIVL.wrapper import UniVL_SimilarityVLM as VLM
elif VLM_ARG == "vttwins":
    from VTTWINS.wrapper import VTTWINS_SimilarityVLM as VLM
else:
    raise NotImplementedError

'''
Classifier Setup
'''
fixed_classifier_kwargs = {} # Classifier keyword parameters to pass to constructor
classifier_hyperparams = [] # Hyperparameter spaces in skopt format
if CLASSIFIER_ARG == "vl_proto":
    from classifier import WeightedTextFewShotClassifier as Classifier
    classifier_hyperparams.append(skopt.space.Real(
        1e-2, 1000,
        name="text_weight", prior="log-uniform"
    ))
elif CLASSIFIER_ARG == "hard_prompt_weighted_text":
    from classifier import HardPromptFewShotClassifier as Classifier
elif CLASSIFIER_ARG == "nearest_neighbor":
    from classifier import NearestNeighborFewShotClassifier as Classifier
    classifier_hyperparams.append(skopt.space.Integer(
        1, 32,
        name="neighbor_count", prior="log-uniform"
    ))
    classifier_hyperparams.append(skopt.space.Categorical(
        ["uniform", "distance"],
        name="neighbor_weights"
    ))
elif CLASSIFIER_ARG == "gaussian_proto":
    from classifier import GaussianFewShotClassifier as Classifier
    classifier_hyperparams.append(skopt.space.Real(
        1e-2, 1000,
        name="text_weight", prior="log-uniform"
    ))
    classifier_hyperparams.append(skopt.space.Integer(
        0, 100,
        name="prior_count", prior="uniform"
    ))
    classifier_hyperparams.append(skopt.space.Real(
        1, 100,
        name="prior_var", prior="log-uniform"
    ))
elif CLASSIFIER_ARG == "subvideo":
    from classifier import SubVideoAverageFewShotClassifier as Classifier
elif CLASSIFIER_ARG == "tip_adapter":
    from classifier import TipAdapterFewShotClassifier as Classifier
    fixed_classifier_kwargs["finetune_epochs"] = 20
    fixed_classifier_kwargs["random_augment"] = False
    
    classifier_hyperparams.append(skopt.space.Categorical(
        [1e0, 1e1, 1e2, 1e3],
        name="alpha"
    ))
    classifier_hyperparams.append(skopt.space.Categorical(
        [4.0, 5.5, 7.0],
        name="beta"
    ))
    classifier_hyperparams.append(skopt.space.Categorical(
        [1e-4, 1e-3, 1e-2],
        name="finetune_lr"
    ))
elif CLASSIFIER_ARG == "smsm_object_oracle":
    from classifier.smsm_object_oracle import SmsmObjectOracleFewShotClassifier as Classifier
elif CLASSIFIER_ARG == "coop":
    from classifier.coop import CoopFewShotClassifier as Classifier
    fixed_classifier_kwargs["random_augment"] = False
    fixed_classifier_kwargs["batch_size"] = 8
    fixed_classifier_kwargs["optimizer"] = "sgd"
    fixed_classifier_kwargs["epochs"] = 50
    
    ORIG_COOP_BATCH_SIZE = 32
    ORIG_COOP_LR = 2e-3
    equiv_lr = ORIG_COOP_LR / ORIG_COOP_BATCH_SIZE * fixed_classifier_kwargs["batch_size"]
    
    classifier_hyperparams.append(skopt.space.Categorical(
        [0.5 * equiv_lr, equiv_lr, 2 * equiv_lr, 4 * equiv_lr, 8 * equiv_lr],
        name="lr"
    ))
    
    '''
    classifier_hyperparams.append(skopt.space.Real(
        1e-4, 1e-1,
        name="lr", prior="log-uniform"
    ))
    '''
    '''
    classifier_hyperparams.append(skopt.space.Categorical(
        [5, 10, 20],
        name="epochs"
    ))
    classifier_hyperparams.append(skopt.space.Categorical(
        [True, False],
        name="random_augment"
    ))
    classifier_hyperparams.append(skopt.space.Categorical(
        [1, 8],
        name="batch_size", prior=[0.1, 0.9]
    ))
    '''
elif CLASSIFIER_ARG == "cona":
    from classifier.cona import CoNaFewShotClassifier as Classifier
    fixed_classifier_kwargs["random_augment"] = False
    fixed_classifier_kwargs["batch_size"] = 8
    fixed_classifier_kwargs["optimizer"] = "sgd"
    fixed_classifier_kwargs["epochs"] = 50
    
    ORIG_COOP_BATCH_SIZE = 32
    ORIG_COOP_LR = 2e-3
    equiv_lr = ORIG_COOP_LR / ORIG_COOP_BATCH_SIZE * fixed_classifier_kwargs["batch_size"]
    
    classifier_hyperparams.append(skopt.space.Categorical(
        [0.5 * equiv_lr, equiv_lr, 2 * equiv_lr, 8 * equiv_lr],
        name="lr"
    ))
    classifier_hyperparams.append(skopt.space.Categorical(
        [1e4, 1e6, 1e8],
        name="name_regularization"
    ))
    '''
    classifier_hyperparams.append(skopt.space.Categorical(
        [5, 10, 20],
        name="epochs"
    ))
    classifier_hyperparams.append(skopt.space.Categorical(
        [True, False],
        name="random_augment"
    ))
    classifier_hyperparams.append(skopt.space.Categorical(
        [1, 8],
        name="batch_size", prior=[0.1, 0.9]
    ))
    '''
else:
    raise ValueError("Unrecognized classifier arg")


VAL_RESULTS_CSV = f"hyperparam_search_val.{Classifier.__name__}.{VLM.__name__}.csv"
TEST_RESULTS_CSV = f"hyperparam_search_test.{Classifier.__name__}.{VLM.__name__}.csv"
val_run_handler = FewShotTestHandler(VAL_RESULTS_CSV)
test_run_handler = FewShotTestHandler(TEST_RESULTS_CSV)



'''
Hyperparameter Search
'''
# Combine vlm and classifier hyperparams
for vlm_hyper in vlm_hyperparams:
    vlm_hyper.name = f"vlm.{vlm_hyper.name}"
for classifier_hyper in classifier_hyperparams:
    classifier_hyper.name = f"classifier.{classifier_hyper.name}"
hyperparam_space = vlm_hyperparams + classifier_hyperparams

train_dataset = None
val_dataset = None
test_dataset = None
cur_dataset_kwargs = None

vlm = None
cur_vlm_kwargs = None

pbar = tqdm(list(itertools.product(*test_params_dict.values())))
for test_params in pbar:
    test_params = dict(zip(test_params_dict.keys(), test_params))
    pbar.set_postfix(test_params)

    dataset_kwargs = {key[8:]: val for key, val in test_params.items() if key.startswith("dataset.")}
    test_kwargs = {key[5:]: val for key, val in test_params.items() if key.startswith("test.")}

    # Update dataset
    if val_dataset is None or cur_dataset_kwargs != dataset_kwargs:
        train_dataset = DatasetHandler(**dataset_kwargs, split="train")
        val_dataset = DatasetHandler(**dataset_kwargs, split="val")
        test_dataset = DatasetHandler(**dataset_kwargs, split="test")
        
        cur_dataset_kwargs = dataset_kwargs
        
    # Convert n_way = None into n_way = max-ways
    if test_kwargs["n_way"] is None:
        test_kwargs["n_way"] = train_dataset.category_count()
        
    # Skip if matching final run already exists in test results csv
    matching_test_run_results = filter_test_results(
        test_run_handler.results,
        dict(
            test_kwargs,
            query_dataset=test_dataset.id(),
            support_dataset=train_dataset.id(),
            val_tuning_dataset=val_dataset.id() if USE_VAL_TUNING else None,
            vlm_class=VLM.__name__,
            **{f"vlm.{key}": val for key, val in fixed_vlm_kwargs.items()},
            classifier_class=Classifier.__name__,
            **{f"classifier.{key}": val for key, val in fixed_classifier_kwargs.items()}
        )
    )
    if len(matching_test_run_results):
        print(f"Skipping hyperparam search which already has test results.")
        print(f"Dataset: {test_dataset.id()}")
        print(f"Test kwargs:\n{json.dumps(test_kwargs, indent=2)}")
        continue
    
    
    
    '''
    Hyperparameter search in given dataset split
    '''
    
    # skopt loss function
    @skopt.utils.use_named_args(hyperparam_space)
    def val_neg_accuracy(**hyperparam_kwargs):
        hyperparam_kwargs = dict(hyperparam_kwargs)
        vlm_kwargs = {key[4:]: val for key, val in hyperparam_kwargs.items() if key.startswith("vlm.")}
        classifier_kwargs = {key[11:]: val for key, val in hyperparam_kwargs.items() if key.startswith("classifier.")}

        # Update vlm if necessary (allow reuse if unchanging)
        global vlm, cur_vlm_kwargs
        if vlm is None or cur_vlm_kwargs != vlm_kwargs:
            vlm = VLM(**fixed_vlm_kwargs, **vlm_kwargs)
            cur_vlm_kwargs = vlm_kwargs
            
        # Update classifier
        classifier = Classifier(vlm, **fixed_classifier_kwargs, **classifier_kwargs)
        
        accuracy = val_run_handler.run_few_shot_test(classifier, val_dataset, train_dataset, **test_kwargs, val_tuning_dataset=val_dataset if USE_VAL_TUNING else None)
        return -1 * accuracy
    
    # Callback function for progress bar
    skopt_pbar = None
    def skopt_callback(current_skopt_results):
        best_run_ind = np.argmin(current_skopt_results.func_vals)
        postfix_dict = {
            "best_acc": round(-1 * current_skopt_results.func_vals[best_run_ind], 5)
        }
        for i, param_space in enumerate(hyperparam_space):
            key = param_space.name
            val = current_skopt_results.x_iters[best_run_ind][i]
            if isinstance(val, float):
                val = round(val, 5)
            postfix_dict[key] = val
        skopt_pbar.update(1)
        skopt_pbar.set_postfix(postfix_dict)
        
    # Find any previous val runs which shall be fed into skopt hyperparam search alg
    # Possible since hyperparameter spaces are named to match names in results csvs, which cover all vlm and classifier parameters
    # Only used for skopt search methods
    prev_val_run_results = filter_test_results(
        val_run_handler.results,
        dict(
            test_kwargs,
            query_dataset=val_dataset.id(),
            support_dataset=train_dataset.id(),
            val_tuning_dataset=val_dataset.id() if USE_VAL_TUNING else None,
            vlm_class=VLM.__name__,
            **{f"vlm.{key}": val for key, val in fixed_vlm_kwargs.items()},
            classifier_class=Classifier.__name__,
            **{f"classifier.{key}": val for key, val in fixed_classifier_kwargs.items()}
        )
    ).reset_index(drop=True)
    if len(prev_val_run_results):
        x0, y0 = [], []
        for i in range(len(prev_val_run_results)):
            x0.append(tuple(prev_val_run_results.loc[i, hyper.name] for hyper in hyperparam_space))
            y0.append(-1 * prev_val_run_results.loc[i, "accuracy"])
    else:
        x0, y0 = None, None
    
    # Run skopt process
    skopt_pbar = tqdm(total=N_HYPERPARAM_SEARCH_CALLS)
    
    if SEARCH_METHOD == "gp":
        skopt_results = skopt.gp_minimize(val_neg_accuracy, hyperparam_space, n_calls=N_HYPERPARAM_SEARCH_CALLS, callback=skopt_callback, x0=x0, y0=y0)
    elif SEARCH_METHOD == "forest":
        skopt_results = skopt.forest_minimize(val_neg_accuracy, hyperparam_space, n_calls=N_HYPERPARAM_SEARCH_CALLS, callback=skopt_callback, x0=x0, y0=y0)
    elif SEARCH_METHOD == "random":
        for _ in range(N_HYPERPARAM_SEARCH_CALLS):
            val_neg_accuracy([hyper.rvs(1)[0] for hyper in hyperparam_space])
            skopt_pbar.update(1)
    elif SEARCH_METHOD == "grid":
        categorical_hyperparams = [hyper for hyper in hyperparam_space if type(hyper) is skopt.space.space.Categorical]
        other_hyperparams = [hyper for hyper in hyperparam_space if type(hyper) is not skopt.space.space.Categorical]
        
        # Grid must iterate over all selected categories
        runs_per_category_choice = N_HYPERPARAM_SEARCH_CALLS
        for hyper in categorical_hyperparams:
            runs_per_category_choice = runs_per_category_choice // len(hyper.categories)
        
        if runs_per_category_choice == 0:
            raise ValueError(f"Too many categorical hyperparameters to iterate over all choices without exceeding {N_HYPERPARAM_SEARCH_CALLS} runs.")
        
        if len(other_hyperparams) == 0:
            discretized_hyperparam_space = [hyper.categories for hyper in hyperparam_space]
        else:
            samples_per_cont_hyper = int(np.power(runs_per_category_choice, 1 / len(other_hyperparams)))
            
            if samples_per_cont_hyper == 0:
                raise ValueError(f"Too many hyperparameters to iterate over all categories and still choose multiple values per continuous space, without exceeding {N_HYPERPARAM_SEARCH_CALLS} runs.")
            
            discretized_hyperparam_space = []
            for hyper in hyperparam_space:
                if type(hyper) is skopt.space.space.Categorical:
                    discretized_hyperparam_space.append(hyper.categories)
                elif type(hyper) in [skopt.space.space.Real, skopt.space.space.Integer]:
                    if hyper.prior == "log-uniform":
                        hyper_samples = np.logspace(np.log10(hyper.low), np.log10(hyper.high), num=samples_per_cont_hyper, endpoint=True)
                    else:
                        hyper_samples = np.linspace(hyper.low, hyper.high, num=samples_per_cont_hyper, endpoint=True)
                    
                    if type(hyper) is skopt.space.space.Integer:
                        hyper_samples = np.round(hyper_samples)
                        
                    discretized_hyperparam_space.append(hyper_samples)
                else:
                    raise NotImplementedError
            
        hyperparam_value_iter = list(itertools.product(*discretized_hyperparam_space))
        skopt_pbar.total = len(hyperparam_value_iter)
        for i, hyperparam_values in enumerate(hyperparam_value_iter):
            val_neg_accuracy(hyperparam_values)
            skopt_pbar.update(1)
        
    else:
        raise NotImplementedError
    
    
    '''
    Test run with best hyperparams
    '''
    # Select best hyperparameter values from val split
    best_hyperparam_values = find_hyperparameters(
        val_run_handler.results,
        hyperparam_cols=[col for col in val_run_handler.results if col.startswith("classifier.") or col.startswith("vlm.")]
    )
    matching_hyperparam_values = filter_test_results(
        best_hyperparam_values,
        dict(
            test_kwargs,
            query_dataset=val_dataset.id(),
            support_dataset=train_dataset.id(),
            val_tuning_dataset=val_dataset.id() if USE_VAL_TUNING else None,
            vlm_class=VLM.__name__,
            **{f"vlm.{key}": val for key, val in fixed_vlm_kwargs.items()},
            classifier_class=Classifier.__name__,
            **{f"classifier.{key}": val for key, val in fixed_classifier_kwargs.items()}
        )
    ).reset_index(drop=True)
    
    vlm_kwargs = {}
    classifier_kwargs = {}
    for col in matching_hyperparam_values.columns:
        if col.startswith("vlm."):
            if col[4:] in fixed_vlm_kwargs.keys():
                continue
            val = matching_hyperparam_values.loc[0, col]
            # NaN values indicate they aren't applicable for this vlm/classifier
            if not pd.isna(val):
                # Replace np types with native python types
                if type(val).__module__ == np.__name__:
                    val = val.item()
                vlm_kwargs[col[4:]] = val
                
        if col.startswith("classifier."):
            if col[11:] in fixed_classifier_kwargs.keys():
                continue
            val = matching_hyperparam_values.loc[0, col]
            # NaN values indicate they aren't applicable for this vlm/classifier
            if not pd.isna(val):
                # Replace np types with native python types
                if type(val).__module__ == np.__name__:
                    val = val.item()
                if col != "classifier.metric":
                    classifier_kwargs[col[11:]] = val
                else:
                    classifier_kwargs[col[11:]] = Similarity[val]
                
    # Update vlm if necessary (allow reuse if unchanging)
    if vlm is None or cur_vlm_kwargs != vlm_kwargs:
        vlm = VLM(**fixed_vlm_kwargs, **vlm_kwargs)
        cur_vlm_kwargs = vlm_kwargs
        
    # Update classifier
    classifier = Classifier(vlm, **fixed_classifier_kwargs, **classifier_kwargs)
    
    test_acc = test_run_handler.run_few_shot_test(classifier, test_dataset, train_dataset, **test_kwargs, val_tuning_dataset=val_dataset if USE_VAL_TUNING else None)
    print(f"Test Run Complete!")
    print(f"Accuracy: {test_acc}")
    print(f"Dataset: {test_dataset.id()}")
    print(f"Test: {json.dumps(test_kwargs, indent=2)}")
    print(f"VLM: {json.dumps(vlm_kwargs, indent=2)}")
    print(f"Classifier: {json.dumps(classifier_kwargs, indent=2)}")
    
    