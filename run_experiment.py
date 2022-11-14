import numpy as np
import os, sys
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm, trange
import pandas as pd
import json
import itertools
import skopt
import skopt.plots
import argparse
from functools import reduce

from FewShotTestHandler import FewShotTestHandler, optimize_hyperparameters, find_hyperparameters, test_already_stored, filter_test_results
from dataset import DatasetHandler
from similarity_metrics import Similarity
from plotting_utils import plot

argparser = argparse.ArgumentParser()
argparser.add_argument("vlm", choices=["clip", "miles", "videoclip"],
                       help="VLM to run. Requires corresponding conda environment.")
argparser.add_argument("classifier", choices=["vl_proto", "hard_prompt_weighted_text", "nearest_neighbor", "gaussian_proto",
                                              "subvideo", "tip_adapter", "coop", "cona", "cona_tip"],
                       help="Classifier to run.")
argparser.add_argument("-d", "--dataset", nargs="+", default=["smsm", "moma_sact", "kinetics_100", "moma_act"],
                       help="Which dataset name to run on.")
argparser.add_argument("--dataset_split", default="val", choices=["val", "test"],
                       help="Which dataset split to evaluate on.")
argparser.add_argument("-s", "--n_shots", nargs="+", type=int, default=[1,2,4,8,16],
                       help="Number of shots to run on.")
argparser.add_argument("--n_episodes", type=int, default=4,
                       help="Number of support set samples to repeat every test over.")
argparser.add_argument("--val_tuning", type=bool, default=True,
                       help="Whether or not the final trained classifier is reloaded from the epoch with the best val performance")
argparser.add_argument("-f", "--file", default=None,
                       help="Optional specific filename to save results csv to")
args, unknown_args_list = argparser.parse_known_args()

# Attempt to parse unknown args as vlm/classifier parameter overrides, like "--classifier.epochs 5 10 20"
cur_key = None
cur_key_vals = []
unknown_args = {}
for unknown_arg in unknown_args_list:
    if unknown_arg.startswith("--"):
        if cur_key is not None:
            unknown_args[cur_key] = cur_key_vals
        cur_key = unknown_arg[2:]
        cur_key_vals = []
    else:
        cur_key_vals.append(unknown_arg)
if cur_key is not None:
    unknown_args[cur_key] = cur_key_vals

'''
Test Setup
'''
# Parameters which will be iterated over.
# Primarily iterates over dataset and test params, but can also iterate over vlm and classifier params,
# overriding any hardcoded parameters set below
params_dict = {}

# Dataset Params - dataset.____ keys are passed into DatasetHandler constructor
params_dict["dataset.name"] = args.dataset
params_dict["dataset.split"] = [args.dataset_split]
params_dict["dataset.split_type"] = ["video"]

# Few-Shot Test Params - test.____ keys are passed into few-shot test call
params_dict["test.n_way"] = [None] # None value gets manually converted to the max size for each dataset
params_dict["test.n_support"] = args.n_shots
params_dict["test.n_query"] = [None]
params_dict["test.n_episodes"] = [args.n_episodes]



'''
VLM Setup
'''
fixed_vlm_kwargs = {} # VLM keyword parameters to pass to constructor
if args.vlm == "clip":
    from CLIP.CLIPVLM import ClipVLM as VLM
    fixed_vlm_kwargs["num_frames"] = 10
elif args.vlm == "miles":
    from MILES.wrapper import MILES_SimilarityVLM as VLM
elif args.vlm == "videoclip":
    from video_clip.video_clip import VideoClipVLM as VLM
    fixed_vlm_kwargs["num_seconds"] = 4
    fixed_vlm_kwargs["sample_strat"] = "spread"
    fixed_vlm_kwargs["use_cuda"] = True
elif args.vlm == "univl":
    from UNIVL.wrapper import UniVL_SimilarityVLM as VLM
elif args.vlm == "vttwins":
    from VTTWINS.wrapper import VTTWINS_SimilarityVLM as VLM
else:
    raise NotImplementedError

'''
Classifier Setup
'''
fixed_classifier_kwargs = {} # Classifier keyword parameters to pass to constructor
if args.classifier == "vl_proto":
    from classifier import WeightedTextFewShotClassifier as Classifier
elif args.classifier == "hard_prompt_weighted_text":
    from classifier import HardPromptFewShotClassifier as Classifier
elif args.classifier == "nearest_neighbor":
    from classifier import NearestNeighborFewShotClassifier as Classifier
elif args.classifier == "gaussian_proto":
    from classifier import GaussianFewShotClassifier as Classifier
elif args.classifier == "subvideo":
    from classifier import SubVideoAverageFewShotClassifier as Classifier
elif args.classifier == "tip_adapter":
    from classifier import TipAdapterFewShotClassifier as Classifier
    fixed_classifier_kwargs["finetune_epochs"] = 20
    fixed_classifier_kwargs["random_augment"] = False
    fixed_classifier_kwargs["beta"] = 5.5
elif args.classifier == "smsm_object_oracle":
    from classifier.smsm_object_oracle import SmsmObjectOracleFewShotClassifier as Classifier
elif args.classifier == "coop":
    from classifier.coop import CoopFewShotClassifier as Classifier
    fixed_classifier_kwargs["random_augment"] = False
    fixed_classifier_kwargs["batch_size"] = 8
    fixed_classifier_kwargs["optimizer"] = "sgd"
    fixed_classifier_kwargs["epochs"] = 20
    fixed_classifier_kwargs["lr"] = 2e-3
elif args.classifier == "cona":
    from classifier.cona import CoNaFewShotClassifier as Classifier
    fixed_classifier_kwargs["random_augment"] = False
    fixed_classifier_kwargs["batch_size"] = 8
    fixed_classifier_kwargs["optimizer"] = "adamw"
    fixed_classifier_kwargs["epochs"] = 20
    fixed_classifier_kwargs["lr"] = 4e-4
    fixed_classifier_kwargs["name_regularization"] = 1
    fixed_classifier_kwargs["context_len"] = 16
elif args.classifier == "cona_tip":
    from classifier.cona_tip_adapter import CoNaTipAdapterFewShotClassifier as Classifier
    fixed_classifier_kwargs["random_augment"] = False
    fixed_classifier_kwargs["batch_size"] = 8
    fixed_classifier_kwargs["optimizer"] = "adamw"
    fixed_classifier_kwargs["epochs"] = 20
    fixed_classifier_kwargs["lr"] = 1e-3
    fixed_classifier_kwargs["adapter_lr_multiplier"] = 1e-2
    fixed_classifier_kwargs["name_regularization"] = 20
    fixed_classifier_kwargs["adapter_regularization"] = 1e-2
    fixed_classifier_kwargs["context_len"] = 4
else:
    raise ValueError("Unrecognized classifier arg")
    
# Insert any overrides from command line into params dict
for key, vals in unknown_args.items():
    if key.startswith("classifier.") or key.startswith("vlm."):
        params_dict[key] = vals


if args.file is None:
    run_handler = FewShotTestHandler(f"experiment.{Classifier.__name__}.{VLM.__name__}.csv")
else:
    run_handler = FewShotTestHandler(args.file)

query_dataset = None
support_dataset = None
cur_dataset_kwargs = None

vlm = None
cur_vlm_kwargs = None

pbar = tqdm(list(itertools.product(*params_dict.values())))
for params in pbar:
    params = dict(zip(params_dict.keys(), params))
    pbar.set_postfix(params)

    dataset_kwargs = {key[8:]: val for key, val in params.items() if key.startswith("dataset.")}
    test_kwargs = {key[5:]: val for key, val in params.items() if key.startswith("test.")}
    vlm_kwargs = {key[4:]: val for key, val in params.items() if key.startswith("vlm.")}
    classifier_kwargs = {key[11:]: val for key, val in params.items() if key.startswith("classifier.")}

    # Update dataset
    if query_dataset is None or cur_dataset_kwargs != dataset_kwargs:
        query_dataset = DatasetHandler(**dataset_kwargs)
        support_dataset = DatasetHandler(**dict(dataset_kwargs, split="train"))
        val_tuning_dataset = DatasetHandler(**dict(dataset_kwargs, split="val"))
        
        cur_dataset_kwargs = dataset_kwargs
        
    # Convert n_way = None into n_way = max-ways
    if test_kwargs["n_way"] is None:
        test_kwargs["n_way"] = query_dataset.category_count()
        
    # Update VLM
    if vlm is None or cur_vlm_kwargs != vlm_kwargs:
        vlm = VLM(**dict(fixed_vlm_kwargs, **vlm_kwargs))
        cur_vlm_kwargs = vlm_kwargs
        
    # Update classifier
    classifier = Classifier(vlm, **dict(fixed_classifier_kwargs, **classifier_kwargs))
    
    accuracy = run_handler.run_few_shot_test(classifier, query_dataset, support_dataset, **test_kwargs, val_tuning_dataset=val_tuning_dataset if args.val_tuning else None)

