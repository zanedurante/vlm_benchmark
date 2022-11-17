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

from FewShotTestHandler import FewShotTestHandler, optimize_hyperparameters, find_hyperparameters, test_already_stored, filter_test_results
from dataset import DatasetHandler
from similarity_metrics import Similarity
from plotting_utils import plot

argparser = argparse.ArgumentParser()
argparser.add_argument("vlm", choices=["clip", "miles", "videoclip"],
                       help="VLM to run. Requires corresponding conda environment")
argparser.add_argument("classifier", choices=["vl_proto", "hard_prompt_vl_proto", "nearest_neighbor", "gaussian_proto",
                                              "subvideo", "tip_adapter", "coop", "cona", "cona_tip"],
                       help="Classifier to run")
argparser.add_argument("-d", "--dataset", nargs="+", default=["smsm", "moma_sact", "kinetics_100", "moma_act"],
                       help="Which dataset name to run on")
argparser.add_argument("-s", "--n_shots", nargs="+", type=int, default=[1,2,4,8,16],
                       help="Number of shots to run on")
argparser.add_argument("-w", "--n_way", nargs="+", type=int, default=[None],
                       help="Number of categories to classify between. Default value None indicates choosing the max categories for each dataset.")
argparser.add_argument("--n_episodes", type=int, default=4,
                       help="Number of support set samples to repeat every test over")
argparser.add_argument("--val_tuning", type=lambda x: (x == "True"), default=True,
                       help="Whether or not the final trained classifier is reloaded from the epoch with the best val performance")
argparser.add_argument("--class-split", action="store_true",
                       help="Flag to use class-wise splitting (meta-learning paradigm) instead of video-wise splitting (finetuning paradigm)")
argparser.add_argument("-m", "--method", default="grid", choices=["gp", "forest", "random", "grid"],
                       help="Hyperparameter search method name.")
argparser.add_argument("-n", "--n_search_runs", type=int, default=32,
                       help="Sets the max number of hyperparameter search runs per test parameter value (dataset+n_shot)")
argparser.add_argument("-f", "--folder", default=None,
                       help="Optional folder path in which to save val and test results. By default creates folder for VLM and Classifier choice")
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
        if unknown_arg in ["True", "False"]:
            unknown_arg = (unknown_arg == "True")
        cur_key_vals.append(unknown_arg)
if cur_key is not None:
    unknown_args[cur_key] = cur_key_vals



'''
Test Setup
'''
# Parameters which will be iterated over.
# Each product will receive individual hyperparam optimization over vlm and classifier params
test_params_dict = {}

# Dataset Params - dataset.____ keys are passed into DatasetHandler constructor
test_params_dict["dataset.name"] = args.dataset

# Few-Shot Test Params - test.____ keys are passed into few-shot test call
test_params_dict["test.n_way"] = args.n_way # None value gets manually converted to the max size for each dataset
test_params_dict["test.n_support"] = args.n_shots
test_params_dict["test.n_query"] = [None]
test_params_dict["test.n_episodes"] = [args.n_episodes]




'''
VLM Setup
'''
fixed_vlm_kwargs = {} # VLM keyword parameters to pass to constructor
vlm_hyperparams = [] # Hyperparameter spaces in skopt format
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
classifier_hyperparams = [] # Hyperparameter spaces in skopt format
if args.classifier == "vl_proto":
    from classifier import WeightedTextFewShotClassifier as Classifier
    classifier_hyperparams.append(skopt.space.Real(
        1e-2, 1e2,
        name="text_weight", prior="log-uniform"
    ))
    classifier_hyperparams.append(skopt.space.Categorical(
        ["tip_adapter", "vid_action"],
        name="prompt_ensembling"
    ))
elif args.classifier == "hard_prompt_vl_proto":
    from classifier import HardPromptFewShotClassifier as Classifier
    classifier_hyperparams.append(skopt.space.Real(
        1e-2, 1e3,
        name="text_weight", prior="log-uniform"
    ))
    classifier_hyperparams.append(skopt.space.Categorical(
        [
            "a photo showing the activity of",
            "a photo showing a",
            "the video shows me",
            "i am"
        ],
        name="prompt_text"
    ))
elif args.classifier == "nearest_neighbor":
    from classifier import NearestNeighborFewShotClassifier as Classifier
    classifier_hyperparams.append(skopt.space.Integer(
        1, 32,
        name="neighbor_count", prior="log-uniform"
    ))
    classifier_hyperparams.append(skopt.space.Categorical(
        ["uniform", "distance"],
        name="neighbor_weights"
    ))
elif args.classifier == "gaussian_proto":
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
        1e-2, 1e2,
        name="prior_var", prior="log-uniform"
    ))
elif args.classifier == "subvideo":
    from classifier import SubVideoAverageFewShotClassifier as Classifier
elif args.classifier == "tip_adapter":
    from classifier import TipAdapterFewShotClassifier as Classifier
    fixed_classifier_kwargs["finetune_epochs"] = 20
    fixed_classifier_kwargs["random_augment"] = False
    fixed_classifier_kwargs["beta"] = 5.5
    
    classifier_hyperparams.append(skopt.space.Categorical(
        [1e0, 1e1, 1e2, 1e3],
        name="alpha"
    ))
    classifier_hyperparams.append(skopt.space.Categorical(
        [1e-4, 4e-4, 1e-3, 4e-3],
        name="finetune_lr"
    ))
    classifier_hyperparams.append(skopt.space.Categorical(
        ["tip_adapter", "vid_action"],
        name="prompt_ensembling"
    ))
elif args.classifier == "smsm_object_oracle":
    from classifier.smsm_object_oracle import SmsmObjectOracleFewShotClassifier as Classifier
elif args.classifier == "coop":
    from classifier.coop import CoopFewShotClassifier as Classifier
    fixed_classifier_kwargs["random_augment"] = False
    fixed_classifier_kwargs["batch_size"] = 8
    fixed_classifier_kwargs["optimizer"] = "sgd"
    fixed_classifier_kwargs["epochs"] = 50
    fixed_classifier_kwargs["csc"] = False
    
    classifier_hyperparams.append(skopt.space.Categorical(
        [8, 16],
        name="context_len"
    ))
    classifier_hyperparams.append(skopt.space.Categorical(
        [6.25e-5, 5e-4, 2e-3, 4e-3],
        name="lr"
    ))
    
elif args.classifier == "cona":
    from classifier.cona import CoNaFewShotClassifier as Classifier
    fixed_classifier_kwargs["random_augment"] = False
    fixed_classifier_kwargs["batch_size"] = 32
    fixed_classifier_kwargs["optimizer"] = "adamw"
    fixed_classifier_kwargs["epochs"] = 50
    fixed_classifier_kwargs["context_len"] = 2
    
    ORIG_COOP_BATCH_SIZE = 32
    ORIG_COOP_LR = 2e-3
    equiv_lr = ORIG_COOP_LR / ORIG_COOP_BATCH_SIZE * fixed_classifier_kwargs["batch_size"]
    
    fixed_classifier_kwargs["warmup_lr"] = min(1e-5, 0.1 * 0.1 * equiv_lr)
    
    classifier_hyperparams.append(skopt.space.Real(
        0.1 * equiv_lr, 10 * equiv_lr,
        name="lr", prior="log-uniform"
    ))
    classifier_hyperparams.append(skopt.space.Real(
        1e-1, 1e1,
        name="name_regularization", prior="log-uniform"
    ))
    
elif args.classifier == "cona_tip":
    from classifier.cona_tip_adapter import CoNaTipAdapterFewShotClassifier as Classifier
    fixed_classifier_kwargs["random_augment"] = False
    fixed_classifier_kwargs["batch_size"] = 8
    fixed_classifier_kwargs["optimizer"] = "adamw"
    fixed_classifier_kwargs["epochs"] = 20
    #fixed_classifier_kwargs["name_regularization"] = 20
    fixed_classifier_kwargs["adapter_regularization"] = 1e-2
    
    classifier_hyperparams.append(skopt.space.Categorical(
        [1e-3],
        name="lr"
    ))
    classifier_hyperparams.append(skopt.space.Categorical(
        [0.1],
        name="adapter_lr_multiplier"
    ))
    classifier_hyperparams.append(skopt.space.Categorical(
        [1, 2, 4],
        name="context_len"
    ))
    classifier_hyperparams.append(skopt.space.Categorical(
        [20],
        name="name_regularization"
    ))
else:
    raise ValueError("Unrecognized classifier arg")

# Update classifier/vlm args with any that have been manually specified
# If only a single value follows, fixed_<classifier/vlm>_args is updated,
# if multiple values follow, <classifier/vlm>_hyperparams are updated or added to
for key, val_list in unknown_args.items():
    if key.startswith("vlm.") and len(val_list) > 0:
        key = key[4:]
        
        if len(val_list) == 1:
            # Update fixed args
            fixed_vlm_kwargs[key] = val_list[0]
        
        else:
            # Update hyperparams    
            new_hyperparam = skopt.space.Categorical(val_list, name=key)
            
            matched_name_hyperparam_inds = [i for i, hyper in enumerate(vlm_hyperparams) if hyper.name == key]
            if len(matched_name_hyperparam_inds):
                vlm_hyperparams[matched_name_hyperparam_inds[0]] = new_hyperparam
            else:
                vlm_hyperparams.append(new_hyperparam)
    elif key.startswith("classifier.") and len(val_list) > 0:
        key = key[11:]
        
        if len(val_list) == 1:
            # Update fixed args
            fixed_classifier_kwargs[key] = val_list[0]
            
        else:
            # Update hyperparams
            new_hyperparam = skopt.space.Categorical(val_list, name=key)
            
            matched_name_hyperparam_inds = [i for i, hyper in enumerate(classifier_hyperparams) if hyper.name == key]
            if len(matched_name_hyperparam_inds):
                classifier_hyperparams[matched_name_hyperparam_inds[0]] = new_hyperparam
            else:
                classifier_hyperparams.append(new_hyperparam)
    else:
        raise ValueError(f"Unrecognized argument: --{' '.join([key] + val_list)}")

'''
Set up results folder
'''
if args.folder is None:
    RESULTS_FOLDER = os.path.join("hyperparam_search", Classifier.__name__, VLM.__name__)
else:
    RESULTS_FOLDER = args.folder
os.makedirs(RESULTS_FOLDER, exist_ok=True)

val_run_handler = FewShotTestHandler(os.path.join(RESULTS_FOLDER, "results_val.csv"))
test_run_handler = FewShotTestHandler(os.path.join(RESULTS_FOLDER, "results_test.csv"))



'''
Hyperparameter Search
'''
# Combine vlm and classifier hyperparams
for vlm_hyper in vlm_hyperparams:
    vlm_hyper.name = f"vlm.{vlm_hyper.name}"
for classifier_hyper in classifier_hyperparams:
    classifier_hyper.name = f"classifier.{classifier_hyper.name}"
hyperparam_space = vlm_hyperparams + classifier_hyperparams

support_dataset_val = None  # Source for example videos during val phase
query_dataset_val = None    # Source for query videos during val phase
support_dataset_test = None # Source for example videos during test phase
query_dataset_test = None   # Source for query videos during test phase
val_tuning_dataset = None   # Optional set of val query videos for selecting best epoch (in val and test phase)
cur_dataset_kwargs = None

vlm = None
cur_vlm_kwargs = None

pbar = tqdm(list(itertools.product(*test_params_dict.values())))
for test_params in pbar:
    test_params = dict(zip(test_params_dict.keys(), test_params))
    pbar.set_postfix(test_params)

    dataset_kwargs = {key[8:]: val for key, val in test_params.items() if key.startswith("dataset.")}
    test_kwargs = {key[5:]: val for key, val in test_params.items() if key.startswith("test.")}

    # Update datasets
    if cur_dataset_kwargs != dataset_kwargs:
        # Primary case: fine-tuning paradigm, videowise splits, support set = train split, query set = val or test split
        if not args.class_split:
            query_dataset_val = DatasetHandler(**dataset_kwargs, split="val", split_type="video")
            query_dataset_test = DatasetHandler(**dataset_kwargs, split="test", split_type="video")
            support_dataset_val = support_dataset_test = DatasetHandler(**dataset_kwargs, split="train", split_type="video")
            val_tuning_dataset = query_dataset_val if args.val_tuning else None
            
        # Alternate case: meta-learning paradigm, classwise splits, support set + query set drawn from same split
        # val-tuning is disabled with this setting
        else:
            query_dataset_val = support_dataset_val = DatasetHandler(**dataset_kwargs, split="val", split_type="class")
            query_dataset_test = support_dataset_test = DatasetHandler(**dataset_kwargs, split="test", split_type="class")
            val_tuning_dataset = None
        
        cur_dataset_kwargs = dataset_kwargs
        
    # Convert n_way = None into n_way = max-ways
    if test_kwargs["n_way"] is None:
        test_kwargs["n_way"] = min(query_dataset_val.category_count(), query_dataset_test.category_count())
        
    '''
    Define functions for skopt optimizer methods, or for generally running tests with any sampled hyperparam values
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
            vlm = VLM(**dict(fixed_vlm_kwargs, **vlm_kwargs))
            cur_vlm_kwargs = vlm_kwargs
            
        # Update classifier
        classifier = Classifier(vlm, **dict(fixed_classifier_kwargs, **classifier_kwargs))
        
        accuracy = val_run_handler.run_few_shot_test(classifier, query_dataset_val, support_dataset_val, **test_kwargs, val_tuning_dataset=val_tuning_dataset)
        return -1 * accuracy
    
    # Callback function for progress bar
    hyper_search_pbar = None
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
        hyper_search_pbar.set_postfix(postfix_dict)
        hyper_search_pbar.update(1)
        
    
        
    '''
    Potentially Skip if matching run already exists in the test split results csv
    '''
    matching_test_run_results = filter_test_results(
        test_run_handler.results,
        dict(
            test_kwargs,
            query_dataset=query_dataset_test.id(),
            support_dataset=support_dataset_test.id(),
            val_tuning_dataset=val_tuning_dataset.id() if val_tuning_dataset is not None else None,
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
    Run hyperparameter search for current test params
    '''
    if args.method in ["gp", "forest"]:
        '''
        Find any previous val runs which shall be fed into skopt hyperparam search alg
        Possible since hyperparameter spaces are named to match names in results csvs, which cover all vlm and classifier parameters
        Only used for skopt search methods
        '''
        prev_val_run_results = filter_test_results(
            val_run_handler.results,
            dict(
                test_kwargs,
                query_dataset=query_dataset_val.id(),
                support_dataset=support_dataset_val.id(),
                val_tuning_dataset=val_tuning_dataset.id() if val_tuning_dataset is not None else None,
                vlm_class=VLM.__name__,
                **{f"vlm.{key}": val for key, val in fixed_vlm_kwargs.items()},
                classifier_class=Classifier.__name__,
                **{f"classifier.{key}": val for key, val in fixed_classifier_kwargs.items()}
            )
        ).reset_index(drop=True)
        if len(prev_val_run_results):
            x0, y0 = [], []
            for i in range(len(prev_val_run_results)):
                # Discard points outside of the current hyperparam space bounds
                in_bounds = True
                for hyper in hyperparam_space:
                    val = prev_val_run_results.loc[i, hyper.name]
                    if isinstance(hyper, skopt.space.space.Categorical):
                        if val not in hyper.categories:
                            in_bounds = False
                            break
                    else:
                        if val < hyper.low or val > hyper.high:
                            in_bounds = False
                            break
                if in_bounds:
                    x0.append(tuple(prev_val_run_results.loc[i, hyper.name] for hyper in hyperparam_space))
                    y0.append(-1 * prev_val_run_results.loc[i, "accuracy"])
        else:
            x0, y0 = None, None
        
        if args.method == "gp":
            skopt_func = skopt.gp_minimize
        elif args.method == "forest":
            skopt_func = skopt.forest_minimize
        else:
            raise NotImplementedError
        
        hyper_search_pbar = tqdm(total=args.n_search_runs)
        skopt_results = skopt_func(val_neg_accuracy, hyperparam_space, n_calls=args.n_search_runs, callback=skopt_callback, x0=x0, y0=y0)
    elif args.method == "random":
        hyper_search_pbar = tqdm(total=args.n_search_runs)
        for _ in range(args.n_search_runs):
            val_neg_accuracy([hyper.rvs(1)[0] for hyper in hyperparam_space])
            hyper_search_pbar.update(1)
    elif args.method == "grid":
        '''
        If doing grid search, convert categorical and continues hyperparam spaces into an evenly spaced grid of samples
        '''
        categorical_hyperparams = [hyper for hyper in hyperparam_space if type(hyper) is skopt.space.space.Categorical]
        other_hyperparams = [hyper for hyper in hyperparam_space if type(hyper) is not skopt.space.space.Categorical]
        
        # Grid must iterate over all selected categories
        runs_per_category_choice = args.n_search_runs
        for hyper in categorical_hyperparams:
            runs_per_category_choice = runs_per_category_choice // len(hyper.categories)
        
        if runs_per_category_choice == 0:
            raise ValueError(f"Too many categorical hyperparameters to iterate over all choices without exceeding {args.n_search_runs} runs.")
        
        if len(other_hyperparams) == 0:
            discretized_hyperparam_space = [hyper.categories for hyper in hyperparam_space]
        else:
            samples_per_cont_hyper = int(np.power(runs_per_category_choice, 1 / len(other_hyperparams)))
            
            if samples_per_cont_hyper == 0:
                raise ValueError(f"Too many hyperparameters to iterate over all categories and still choose multiple values per continuous space, without exceeding {args.n_search_runs} runs.")
            
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
        hyper_search_pbar = tqdm(total=len(hyperparam_value_iter))
        for i, hyperparam_values in enumerate(hyperparam_value_iter):
            val_neg_accuracy(hyperparam_values)
            hyper_search_pbar.update(1)
    else:
        raise NotImplementedError
    
    '''
    Save skopt dependence plots if not grid search
    '''
    if args.method != "grid":
        completed_val_run_results = filter_test_results(
            val_run_handler.results,
            dict(
                test_kwargs,
                query_dataset=query_dataset_val.id(),
                support_dataset=support_dataset_val.id(),
                val_tuning_dataset=val_tuning_dataset.id() if val_tuning_dataset is not None else None,
                vlm_class=VLM.__name__,
                **{f"vlm.{key}": val for key, val in fixed_vlm_kwargs.items()},
                classifier_class=Classifier.__name__,
                **{f"classifier.{key}": val for key, val in fixed_classifier_kwargs.items()}
            )
        ).reset_index(drop=True)
        x0, y0 = [], []
        for i in range(len(completed_val_run_results)):
            # Discard points outside of the current hyperparam space bounds
            in_bounds = True
            for hyper in hyperparam_space:
                val = completed_val_run_results.loc[i, hyper.name]
                if isinstance(hyper, skopt.space.space.Categorical):
                    if val not in hyper.categories:
                        in_bounds = False
                        break
                else:
                    if val < hyper.low or val > hyper.high:
                        in_bounds = False
                        break
            if in_bounds:
                x0.append(tuple(completed_val_run_results.loc[i, hyper.name] for hyper in hyperparam_space))
                y0.append(-1 * completed_val_run_results.loc[i, "accuracy"])
        dummy_results = skopt.gp_minimize(val_neg_accuracy, hyperparam_space, n_calls=0, n_initial_points=0, x0=x0, y0=y0)
        ax = skopt.plots.plot_objective(dummy_results, levels=100)
        if isinstance(ax, np.ndarray):
            fig = ax[0,0].get_figure()
        else:
            fig = ax.get_figure()
        fig.savefig(
            os.path.join(RESULTS_FOLDER, ".".join([query_dataset_val.id()] + [f"{key}_{val}" for key, val in test_kwargs.items()] + ["jpg"])),
            facecolor="white", bbox_inches="tight"
        )
    
    
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
            query_dataset=query_dataset_val.id(),
            support_dataset=support_dataset_val.id(),
            val_tuning_dataset=val_tuning_dataset.id() if val_tuning_dataset is not None else None,
            vlm_class=VLM.__name__,
            **{f"vlm.{key}": val for key, val in fixed_vlm_kwargs.items()},
            classifier_class=Classifier.__name__,
            **{f"classifier.{key}": val for key, val in fixed_classifier_kwargs.items()}
        )
    ).reset_index(drop=True)
    
    vlm_kwargs = {}
    classifier_kwargs = {}
    for col in matching_hyperparam_values.columns:
        # Skip vlm/classifier args that aren't in hyperparam space
        if col not in [hyper.name for hyper in hyperparam_space]:
            continue
        
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
                if col == "classifier.metric":
                    classifier_kwargs[col[11:]] = Similarity[val]
                else:
                    classifier_kwargs[col[11:]] = val
                
    # Update vlm if necessary (allow reuse if unchanging)
    if vlm is None or cur_vlm_kwargs != vlm_kwargs:
        vlm = VLM(**fixed_vlm_kwargs, **vlm_kwargs)
        cur_vlm_kwargs = vlm_kwargs
        
    # Update classifier
    classifier = Classifier(vlm, **fixed_classifier_kwargs, **classifier_kwargs)
    
    test_acc = test_run_handler.run_few_shot_test(classifier, query_dataset_test, support_dataset_test, **test_kwargs, val_tuning_dataset=val_tuning_dataset)
    print(f"Test Run Complete!")
    print(f"Accuracy: {test_acc}")
    print(f"Dataset: {query_dataset_test.id()}")
    print(f"Test: {json.dumps(test_kwargs, indent=2)}")
    print(f"VLM: {json.dumps(vlm.params(), indent=2)}")
    print(f"Classifier: {json.dumps(classifier.params(), indent=2)}")
    
    