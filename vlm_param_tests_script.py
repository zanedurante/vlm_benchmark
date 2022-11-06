import numpy as np
import os, sys
import importlib
from tqdm.autonotebook import tqdm, trange
import pandas as pd
import json
import itertools

from FewShotTestHandler import FewShotTestHandler, optimize_hyperparameters, find_hyperparameters, test_already_stored
from dataset import DatasetHandler
from similarity_metrics import Similarity
from plotting_utils import plot

VLM_ARG = sys.argv[1]
CLASSIFIER_ARG = sys.argv[2]



params_dict = {}

# Dataset Params - dataset.____ keys are passed into DatasetHandler constructor
params_dict["dataset.name"] = ["smsm"]
params_dict["dataset.split"] = ["val"]
params_dict["dataset.split_type"] = ["video"]

# Few-Shot Test Params - test.____ keys are passed into few-shot test call
params_dict["test.n_way"] = [100]
params_dict["test.n_support"] = [1, 2, 4, 8, 16, 32, 64]
params_dict["test.n_query"] = [None]
params_dict["test.n_episodes"] = [4]

# VLM Params - vlm.____ keys are passed into VLM constructor
if VLM_ARG == "clip":
    from CLIP.CLIPVLM import ClipVLM as VLM
    params_dict["vlm.num_frames"] = [10]
elif VLM_ARG == "miles":
    from MILES.wrapper import MILES_SimilarityVLM as VLM
elif VLM_ARG == "videoclip":
    from video_clip.video_clip import VideoClipVLM as VLM
    params_dict["vlm.num_seconds"] = [4]
    params_dict["vlm.sample_strat"] = ["spread"]
    params_dict["vlm.use_cuda"] = [True]
elif VLM_ARG == "univl":
    from UNIVL.wrapper import UniVL_SimilarityVLM as VLM
elif VLM_ARG == "vttwins":
    from VTTWINS.wrapper import VTTWINS_SimilarityVLM as VLM
else:
    raise NotImplementedError

# Classifier Params - classifier.____ keys are passed into classifier constructor
if CLASSIFIER_ARG == "vl_proto":
    from classifier import WeightedTextFewShotClassifier as Classifier
    params_dict["classifier.text_weight"] = [0, 0.1, 1.0, 4.0, 10.0, 100.0]
    #params_dict["classifier.metric"] = [Similarity.COSINE, Similarity.DOT, Similarity.EUCLID]
elif CLASSIFIER_ARG == "hard_prompt_weighted_text":
    from classifier import HardPromptFewShotClassifier as Classifier
    params_dict["classifier.text_weight"] = [0, 0.1, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 20.0, 30.0, 40.0, 50.0, 100.0]
    params_dict["classifier.prompt_text"] = [
        "",
        "a photo showing the task of"
    ]
elif CLASSIFIER_ARG == "nearest_neighbor":
    from classifier import NearestNeighborFewShotClassifier as Classifier
    params_dict["classifier.neighbor_count"] = [1, 2, 3, 4, 5, 10, 20]
    params_dict["classifier.neighbor_weights"] = ["uniform", "distance"]
elif CLASSIFIER_ARG == "gaussian_proto":
    from classifier import GaussianFewShotClassifier as Classifier
    params_dict["classifier.text_weight"] = [0, 0.1, 1.0, 4.0, 10.0, 100.0]
    params_dict["classifier.prior_count"] = [0, 1, 3, 10, 30, 100]
    params_dict["classifier.prior_var"] = [0, 1, 3, 10, 30, 100]
elif CLASSIFIER_ARG == "subvideo":
    from classifier import SubVideoAverageFewShotClassifier as Classifier
    params_dict["classifier.text_weight"] = [0, 0.1, 1.0, 4.0, 10.0, 100.0]
    params_dict["classifier.subvideo_segment_duration"] = [1, 2, 5]
    params_dict["classifier.subvideo_max_segments"] = [32]
    params_dict["classifier.subvideo_discard_proportion"] = [0, 0.1, 0.25, 0.5]
elif CLASSIFIER_ARG == "tip_adapter":
    from classifier import TipAdapterFewShotClassifier as Classifier
    params_dict["classifier.alpha"] = [1.0, 5.0, 10.0, 50.0, 100.0, 500.0]
    params_dict["classifier.beta"] = [1.0, 2.5, 5.0, 5.5, 6.0, 10.0, 20.0]
    params_dict["classifier.finetune_lr"] = [1e-4, 1e-3, 1e-2]
    params_dict["classifier.finetune_epochs"] = [5, 10, 20]
    params_dict["classifier.weight_decay"] = [0, 1e-3, 1e-2]
elif CLASSIFIER_ARG == "smsm_object_oracle":
    from classifier.smsm_object_oracle import SmsmObjectOracleFewShotClassifier as Classifier
elif CLASSIFIER_ARG == "coop":
    from classifier.coop import CoopFewShotClassifier as Classifier
    params_dict["classifier.lr"] = [3e-4, 1e-3, 3e-3, 1e-2, 2e-2]
    params_dict["classifier.epochs"] = [5, 20]
    params_dict["classifier.random_augment"] = [True, False]
    params_dict["classifier.batch_size"] = [2, 8]
else:
    raise ValueError("Unrecognized classifier arg")


    
TEST_FILENAME = f"{Classifier.__name__}.{VLM.__name__}.csv"
test_handler = FewShotTestHandler(TEST_FILENAME)



'''
VAL RUNS
'''

vlm = None
cur_vlm_params = None
classifier = None
cur_classifier_params = None
query_dataset = None
support_dataset = None
cur_dataset_params = None

pbar = tqdm(list(itertools.product(*params_dict.values())))
for params in pbar:
    # Associate keys to each param
    params = dict(zip(params_dict.keys(), params))
    
    pbar.set_postfix(params)
    
    # vlm params
    vlm_params = {key[4:]: val for key, val in params.items() if key.startswith("vlm.")}
    classifier_params = {key[11:]: val for key, val in params.items() if key.startswith("classifier.")}
    dataset_params = {key[8:]: val for key, val in params.items() if key.startswith("dataset.")}
    test_params = {key[5:]: val for key, val in params.items() if key.startswith("test.")}
    
    # Update dataset
    if query_dataset is None or cur_dataset_params != dataset_params:
        query_dataset = DatasetHandler(**dataset_params)
        support_dataset_params = dict(dataset_params, split="train")
        support_dataset = DatasetHandler(**support_dataset_params)
        
        cur_dataset_params = dataset_params
        new_dataset = True
    else:
        new_dataset = False
    
    # Update vlm (which forces update of classifier)
    if vlm is None or cur_vlm_params != vlm_params:
        vlm = VLM(**vlm_params)
        
        cur_vlm_params = vlm_params
        new_vlm = True
    else:
        new_vlm = False
            
    if new_vlm or classifier is None or cur_classifier_params != classifier_params:
        classifier = Classifier(vlm, **classifier_params)
        cur_classifier_params = classifier_params
        
    # Fill dataset caches
    if new_dataset or new_vlm:
        query_dataset.fill_cache(vlm)
        support_dataset.fill_cache(vlm)
    
    # Run test
    test_handler.run_few_shot_test(classifier, query_dataset, support_dataset, **test_params)






'''
TEST RUNS
'''

best_hyperparam_values = find_hyperparameters(
    test_handler.results,
    hyperparam_cols=[col for col in test_handler.results if col.startswith("classifier.") or col.startswith("vlm.")]
)

# Change params_dict to only include dataset and test info, then run tests with best hyperparameter values
test_split_params_dict = {
    "dataset.split": ["test"]
}
for key, val in params_dict.items():
    if key.startswith("classifier.") or key.startswith("vlm.") or key == "dataset.split":
        continue
    test_split_params_dict[key] = val
    
final_test_handler = FewShotTestHandler(f"test.{TEST_FILENAME}")
    
vlm = None
cur_vlm_params = None
classifier = None
cur_classifier_params = None
query_dataset = None
support_dataset = None
cur_dataset_params = None

pbar = tqdm(list(itertools.product(*test_split_params_dict.values())))
for params in pbar:
    # Associate keys to each param
    params = dict(zip(test_split_params_dict.keys(), params))
    
    # Determine dataset and test parameters
    dataset_params = {key[8:]: val for key, val in params.items() if key.startswith("dataset.")}
    test_params = {key[5:]: val for key, val in params.items() if key.startswith("test.")}
    
    # Update dataset
    if query_dataset is None or cur_dataset_params != dataset_params:
        query_dataset = DatasetHandler(**dataset_params)
        support_dataset_params = dict(dataset_params, split="train")
        support_dataset = DatasetHandler(**support_dataset_params)
        
        # Construct dummy val dataset to get id for filtering dataframe results with corresponding hyperparameters
        val_dataset = DatasetHandler(**dict(dataset_params, split="val"))
        
        cur_dataset_params = dataset_params
        new_dataset = True
    else:
        new_dataset = False
        
    # Determine vlm and classifier params from hyperparameter dataframe
    matched_hyperparam_values = np.ones(len(best_hyperparam_values)).astype(bool)
    matched_hyperparam_values &= (best_hyperparam_values["vlm_class"] == VLM.__name__) & (best_hyperparam_values["classifier_class"] == Classifier.__name__)
    matched_hyperparam_values &= (best_hyperparam_values["query_dataset"] == val_dataset.id()) & (best_hyperparam_values["support_dataset"] == support_dataset.id())
    for col, val in test_params.items():
        if pd.isna(val):
            matched_hyperparam_values &= pd.isna(best_hyperparam_values[col])        
        else:
            matched_hyperparam_values &= (best_hyperparam_values[col] == val)
    matched_hyperparam_values = best_hyperparam_values[matched_hyperparam_values].reset_index(drop=True)
    
    vlm_params = {}
    classifier_params = {}
    for col in matched_hyperparam_values.columns:
        if col.startswith("vlm."):
            val = matched_hyperparam_values.loc[0, col]
            if not pd.isna(val):
                vlm_params[col[4:]] = val
        
        if col.startswith("classifier."):
            val = matched_hyperparam_values.loc[0, col]
            if not pd.isna(val):
                classifier_params[col[11:]] = val
        
    for key, val in vlm_params.items():
        params[f"vlm.{key}"] = val
    for key, val in classifier_params.items():
        params[f"classifier.{key}"] = val
    pbar.set_postfix(params)
    
    
    
    # Update vlm (which forces update of classifier)
    if vlm is None or cur_vlm_params != vlm_params:
        vlm = VLM(**vlm_params)
        
        cur_vlm_params = vlm_params
        new_vlm = True
    else:
        new_vlm = False
            
    if new_vlm or classifier is None or cur_classifier_params != classifier_params:
        classifier = Classifier(vlm, **classifier_params)
        cur_classifier_params = classifier_params
        
    # Fill dataset caches
    if new_dataset or new_vlm:
        query_dataset.fill_cache(vlm)
        support_dataset.fill_cache(vlm)
    
    # Run test
    final_test_handler.run_few_shot_test(classifier, query_dataset, support_dataset, **test_params)
    


print(final_test_handler.results)

plot(
    final_test_handler.results,
    x_col="n_support",
    y_col="accuracy",
    plot_descriptor_cols=["dataset", "classifier_class"],
    line_descriptor_cols=["vlm_class"]
)