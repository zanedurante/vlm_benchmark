import numpy as np
import os, sys
import importlib
from tqdm.autonotebook import tqdm, trange
import pandas as pd
import json
import itertools
import skopt
import random
from glob import glob
from classifier.coop import SharedContextCoopModule
import torch

from FewShotTestHandler import FewShotTestHandler, optimize_hyperparameters, find_hyperparameters, test_already_stored, filter_test_results
from dataset import DatasetHandler
from similarity_metrics import Similarity
from plotting_utils import plot

VLM_ARG = sys.argv[1]
CLASSIFIER_ARG = sys.argv[2]

N_HYPERPARAM_SEARCH_CALLS = 1
USE_BEST_VAL = True

'''
Test Setup
'''
# Parameters which will be iterated over.
# Each product will receive individual hyperparam optimization over vlm and classifier params
test_params_dict = {}

# Dataset Params - dataset.____ keys are passed into DatasetHandler constructor
test_params_dict["dataset.name"] = ["smsm"]
test_params_dict["dataset.split"] = ["val"]
test_params_dict["dataset.split_type"] = ["video"]

# Few-Shot Test Params - test.____ keys are passed into few-shot test call
test_params_dict["test.n_way"] = [100]
test_params_dict["test.n_support"] = [1, 2, 4, 8, 16, 32, 64]
test_params_dict["test.n_query"] = [None]
test_params_dict["test.n_episodes"] = [1] # Was 3



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
    classifier_hyperparams.append(skopt.space.Real(
        1, 1000,
        name="alpha", prior="log-uniform"
    ))
    classifier_hyperparams.append(skopt.space.Real(
        1, 50,
        name="beta", prior="log-uniform"
    ))
    classifier_hyperparams.append(skopt.space.Real(
        1e-5, 1e-1,
        name="finetune_lr", prior="log-uniform"
    ))
    classifier_hyperparams.append(skopt.space.Integer(
        1, 20,
        name="finetune_epochs", prior="log-uniform"
    ))
    classifier_hyperparams.append(skopt.space.Real(
        1e-5, 1e-1,
        name="weight_decay", prior="log-uniform"
    ))
elif CLASSIFIER_ARG == "smsm_object_oracle":
    from classifier.smsm_object_oracle import SmsmObjectOracleFewShotClassifier as Classifier
elif CLASSIFIER_ARG == "coop":
    from classifier.coop import CoopFewShotClassifier as Classifier
    classifier_hyperparams.append(skopt.space.Real(
        1e-4, 1e-1,
        name="lr", prior="log-uniform"
    ))
    classifier_hyperparams.append(skopt.space.Categorical(
        [2], # Change to 10 or 15
        name="epochs"
    ))
    classifier_hyperparams.append(skopt.space.Categorical(
        [False], # Changed from [True, False]
        name="random_augment"
    ))
    classifier_hyperparams.append(skopt.space.Categorical(
        [USE_BEST_VAL],
        name="use_best_query"
    ))
    classifier_hyperparams.append(skopt.space.Categorical(
        [1], # Changed from [1, 8]
        name="batch_size"
    ))
elif CLASSIFIER_ARG == "cona":
    from classifier.cona import CoNaFewShotClassifier as Classifier
    classifier_hyperparams.append(skopt.space.Real(
        1e-3, 1e-2,
        name="lr", prior="log-uniform"
    ))
    classifier_hyperparams.append(skopt.space.Real(
        1e6, 1e9,
        name="name_regularization", prior="log-uniform"
    ))
    classifier_hyperparams.append(skopt.space.Categorical(
        [3, 4, 5],
        name="epochs"
    ))
    classifier_hyperparams.append(skopt.space.Categorical(
        [False],
        name="random_augment"
    ))
    classifier_hyperparams.append(skopt.space.Categorical(
        [1],
        name="batch_size"#, prior=[0.1, 0.9]
    ))
else:
    raise ValueError("Unrecognized classifier arg")


VAL_RESULTS_CSV = f"random_hyperparam_search_val.{Classifier.__name__}.{VLM.__name__}.csv"
TEST_RESULTS_CSV = f"random_hyperparam_search_test.{Classifier.__name__}.{VLM.__name__}.csv"
val_run_handler = FewShotTestHandler(VAL_RESULTS_CSV, use_best_query=True, dataset_name=test_params_dict["dataset.name"])
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

query_dataset = None
support_dataset = None
test_dataset = None
cur_dataset_kwargs = None

vlm = None
cur_vlm_kwargs = None

pbar = tqdm(list(itertools.product(*test_params_dict.values())))
for test_params in pbar:
    test_params = dict(zip(test_params_dict.keys(), test_params))
    print("Testing params:", test_params)
    pbar.set_postfix(test_params)

    dataset_kwargs = {key[8:]: val for key, val in test_params.items() if key.startswith("dataset.")}
    test_kwargs = {key[5:]: val for key, val in test_params.items() if key.startswith("test.")}

    # Update dataset
    if query_dataset is None or cur_dataset_kwargs != dataset_kwargs:
        query_dataset = DatasetHandler(**dataset_kwargs)
        support_dataset_kwargs = dict(dataset_kwargs, split="train")
        support_dataset = DatasetHandler(**support_dataset_kwargs)
        test_dataset_kwargs = dict(dataset_kwargs, split="test")
        test_dataset = DatasetHandler(**test_dataset_kwargs)
        
        cur_dataset_kwargs = dataset_kwargs
        
    # Skip if matching final run already exists in test results csv
    matching_test_run_results = filter_test_results(
        test_run_handler.results,
        dict(
            test_kwargs,
            query_dataset=test_dataset.id(),
            support_dataset=support_dataset.id(),
            vlm_class=VLM.__name__,
            **{f"vlm.{key}": val for key, val in fixed_vlm_kwargs.items()},
            classifier_class=Classifier.__name__,
            **{f"classifier.{key}": val for key, val in fixed_classifier_kwargs.items()}
        )
    )
    if len(matching_test_run_results):
        print(f"Skipping hyperparam search which already has test results.")
        print(f"Dataset: {query_dataset.id()}")
        print(f"Test kwargs:\n{json.dumps(test_kwargs, indent=2)}")
        continue
    
    def get_accuracy(**hyperparam_kwargs):
        hyperparam_kwargs = dict(hyperparam_kwargs)
        print("Trying hyper params:", hyperparam_kwargs)
        vlm_kwargs = {key[4:]: val for key, val in hyperparam_kwargs.items() if key.startswith("vlm.")}
        classifier_kwargs = {key[11:]: val for key, val in hyperparam_kwargs.items() if key.startswith("classifier.")}

        # Update vlm if necessary (allow reuse if unchanging)
        global vlm, cur_vlm_kwargs
        if vlm is None or cur_vlm_kwargs != vlm_kwargs:
            vlm = VLM(**fixed_vlm_kwargs, **vlm_kwargs)
            cur_vlm_kwargs = vlm_kwargs
            
        # Update classifier
        classifier = Classifier(vlm, **classifier_kwargs)
        
        accuracy = val_run_handler.run_few_shot_test(classifier, query_dataset, support_dataset, **test_kwargs)
        return accuracy
    
    
    '''
    Hyperparameter search in given dataset split
    '''
        
    # Find any previous val runs which shall be fed into skopt hyperparam search alg
    # Possible since hyperparameter spaces are named to match names in results csvs, which cover all vlm and classifier parameters
    prev_val_run_results = filter_test_results(
        val_run_handler.results,
        dict(
            test_kwargs,
            query_dataset=query_dataset.id(),
            support_dataset=support_dataset.id(),
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
    #skopt_pbar = tqdm(total=N_HYPERPARAM_SEARCH_CALLS)
    #skopt_results = skopt.gp_minimize(val_neg_accuracy, hyperparam_space, n_calls=N_HYPERPARAM_SEARCH_CALLS, callback=skopt_callback, x0=x0, y0=y0)
    
    # Run randomized optimization with N_HYPERPARAM_SEARCH_CALLS random values
    for _ in range(N_HYPERPARAM_SEARCH_CALLS):
        random_param_vals = []
        for param in hyperparam_space:
            if type(param) == skopt.space.space.Real:
                random_param_vals.append((param.name, np.random.uniform(param.low, param.high)))
            elif type(param) == skopt.space.space.Categorical:
                random_param_vals.append((param.name, random.choice(param.categories)))

        acc = get_accuracy(**dict(random_param_vals))
        print("Parameters:", random_param_vals)
        print("Accuracy:", acc)
    
    
    
    '''
    Test run with best hyperparams
    '''
    
    # TODO: Run tests by loading model if use_best_val
    
    if USE_BEST_VAL:
        best_val = 0
        num_episodes = test_params_dict["test.n_episodes"][0]
        # Load the best models and perform inference
        # TODO: Add multi-method support
        configs = sorted(glob("fewshot_models/{}/{}/{}_way/{}_shots/best_param*".format(CLASSIFIER_ARG,test_params["dataset.name"], test_params["test.n_way"], test_params["test.n_support"])))
        # Every num_episodes should be the same config
        start_idx = 0
        best_idx = 0
        for _ in range(N_HYPERPARAM_SEARCH_CALLS):
            total_val_acc = 0
            for config_path in configs[start_idx:start_idx+num_episodes]:
                config = torch.load(config_path)
                total_val_acc += config['val_acc']
            
            if total_val_acc > best_val:
                best_idx = start_idx
                best_val = total_val_acc
            
            start_idx += num_episodes
                
        # Report test results for best_idx
        for config_path in configs[best_idx:best_idx+num_episodes]:
            # Load classifier from params
            params = torch.load(config_path)
            del params['val_acc']
            if CLASSIFIER_ARG == "coop":                
                model = SharedContextCoopModule(vlm, category_names=val_run_handler.category_names, context_len=params['context_len']) #category_names
            else:
                raise NotImplementedError()
            model.load_state_dict(torch.load(config_path.replace("best_param", "best_models")))
            
            # TODO: Do best way to get the predictions on the test set!
            import pdb
            pdb.set_trace()
            
        
        exit()
        
    
    # Select best hyperparameter values from val split
    best_hyperparam_values = find_hyperparameters(
        val_run_handler.results,
        hyperparam_cols=[col for col in val_run_handler.results if col.startswith("classifier.") or col.startswith("vlm.")]
    )
    matching_hyperparam_values = filter_test_results(
        best_hyperparam_values,
        dict(
            test_kwargs,
            query_dataset=query_dataset.id(),
            support_dataset=support_dataset.id(),
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
                classifier_kwargs[col[11:]] = val
                
    # Update vlm if necessary (allow reuse if unchanging)
    if vlm is None or cur_vlm_kwargs != vlm_kwargs:
        vlm = VLM(**fixed_vlm_kwargs, **vlm_kwargs)
        cur_vlm_kwargs = vlm_kwargs
        
    # Update classifier
    classifier = Classifier(vlm, **fixed_classifier_kwargs, **classifier_kwargs)
    
    test_acc = test_run_handler.run_few_shot_test(classifier, test_dataset, support_dataset, **test_kwargs)
    print(f"Test Run Complete!")
    print(f"Accuracy: {test_acc}")
    print(f"Dataset: {test_dataset.id()}")
    print(f"Test: {json.dumps(test_kwargs, indent=2)}")
    print(f"VLM: {json.dumps(vlm_kwargs, indent=2)}")
    print(f"Classifier: {json.dumps(classifier_kwargs, indent=2)}")
    
    