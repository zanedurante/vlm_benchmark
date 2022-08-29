import numpy as np
from tqdm import tqdm, trange
import json
import argparse
import itertools

from SimilarityVLM import SimilarityVLM
from dataset import DatasetHandler
from FewShotTestHandler import FewShotTestHandler
from classifier import WeightedTextFewShotClassifier # TODO: Support multiple variations of classifiers



def get_vlm(vlm_name: str) -> SimilarityVLM:
    if vlm_name == "VT-TWINS":
        from VTTWINS.wrapper import VTTWINS_SimilarityVLM
        return VTTWINS_SimilarityVLM(reset_cache=False)
        
    if vlm_name == "CLIP":
        from CLIP.CLIPVLM import ClipVLM
        return ClipVLM(reset_cache=False)
        
    if vlm_name == "UniVL":
        from UNIVL.wrapper import UniVL_SimilarityVLM
        return UniVL_SimilarityVLM(reset_cache=False)
        
    if vlm_name == "MILES":
        from MILES.wrapper import MILES_SimilarityVLM
        return MILES_SimilarityVLM(reset_cache=False)
    
    if vlm_name == "VideoCLIP":
        from video_clip.video_clip import VideoClipVLM
        return VideoClipVLM(reset_cache=False)
    
    raise ValueError(f"Unrecognized VLM name: {vlm_name}")

def get_param_iterator(param_json_file: str) -> list:
    PARAM_KEYS = ["dataset_name", "dataset_split", "n_way", "n_support", "n_query", "n_episodes", "text_weight"]
    
    with open(args.parameters, "r") as fp:
        params = json.load(fp)
    
    # Check json validity
    missing_keys = [key for key in PARAM_KEYS if key not in params.keys()]
    if len(missing_keys):
        raise ValueError(f"Param json file missing required keys: {missing_keys}")
    
    # Convert all given param values to lists (even if they only have a single value)
    for key in params.keys():
        if type(params[key]) is not list:
            params[key] = [params[key]]
            
    # Create an experiment param iterator which samples all valid combinations of params
    experiment_param_iter = [
        {PARAM_KEYS[i]: param_values_tuple[i] for i in range(len(PARAM_KEYS))}
        for param_values_tuple in itertools.product(*[params[key] for key in PARAM_KEYS])
    ]
    
    # Filter out parameter instances which have n_support = 0, but not text_weight = 1 (text_weight does nothing in the zero-shot case)
    experiment_param_iter = list(filter(lambda exp_params: not (exp_params["n_support"] == 0 and exp_params["text_weight"] != 1), experiment_param_iter))
    
    return experiment_param_iter



if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Runs repeated few-shot tests on the given VLM with all combinations of the parameters specified in the given json file.")
    argparser.add_argument("vlm_name", type=str, help="VLM name to use for experiments. Assumes the script is run in the corresponding conda environment.")
    argparser.add_argument("parameters", type=str, help="Path to a json file specifying parameter values (singular or lists) to run the experiments on.")
    args = argparser.parse_args()
    
    print(f"\n\n\n----- {args.vlm_name} -----")
    
    vlm = get_vlm(args.vlm_name)
    param_iter = get_param_iterator(args.parameters)
    
    # During testing, save most recently loaded dataset for reuse
    # Assumes dataset params are in the outermost loop of the product / first in the PARAM_KEYS list
    prev_dataset = None
    
    # Save which datasets have been explicitly cached (repeatedly running fill_cache is fast but still wasted time)
    datasets_in_cache = set()
    
    test_handler = FewShotTestHandler()
    pbar = tqdm(param_iter)
    for exp_params in pbar:
        pbar.set_postfix(exp_params)
        
        # Load dataset
        if prev_dataset is not None and prev_dataset.name == exp_params["dataset_name"] and prev_dataset.split == exp_params["dataset_split"]:
            dataset = prev_dataset
        else:
            dataset = DatasetHandler(exp_params["dataset_name"], exp_params["dataset_split"])
            
        # Fill vlm cache
        if dataset.id() not in datasets_in_cache:
            test_handler.fill_cache(vlm, dataset)
            datasets_in_cache.add(dataset.id())
            
        # Skip if few-shot task params are too large for the dataset
        if not dataset.valid_for_few_shot(exp_params["n_way"], exp_params["n_support"], exp_params["n_query"]):
            continue
        
        # Construct classifier around vlm
        classifier = WeightedTextFewShotClassifier(vlm, text_weight=exp_params["text_weight"])
        
        # Run experiment
        test_handler.run_few_shot_test(classifier, dataset, exp_params["n_way"], exp_params["n_support"],
                                       exp_params["n_query"], exp_params["n_episodes"])
        
        prev_dataset = dataset