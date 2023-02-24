import os, sys
import math
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from similarity_metrics import Similarity

N_COLS = 6
N_SUPPORT = 16



VLM_ARG = sys.argv[1]
DATASET_NAME = sys.argv[2]

'''
VLM Setup
'''
if VLM_ARG == "clip":
    from CLIP.CLIPVLM import ClipVLM as VLM
    vlm = VLM(num_frames=10)
    
elif VLM_ARG == "miles":
    from MILES.wrapper import MILES_SimilarityVLM as VLM
    vlm = VLM()
    
elif VLM_ARG == "videoclip":
    from video_clip.video_clip import VideoClipVLM as VLM
    vlm = VLM(num_seconds=4, sample_strat="spread", use_cuda=True)
    
else:
    raise ValueError


'''
Classifier setup
'''
N_EPOCHS = 50
BATCH_SIZE = 8
CONTEXT_LEN = 4

from classifier.coop import CoopFewShotClassifier
coop = CoopFewShotClassifier(
    vlm,
    lr=1e-3,
    epochs=N_EPOCHS,
    context_len=CONTEXT_LEN,
    batch_size=BATCH_SIZE,
    random_augment=False
)

from classifier.cona import CoNaFewShotClassifier
cona = CoNaFewShotClassifier(
    vlm,
    lr=1e-3,
    epochs=N_EPOCHS,
    optimizer="adamw",
    name_regularization=1,
    context_len=CONTEXT_LEN,
    batch_size=BATCH_SIZE,
    random_augment=False
)

from classifier.cona_adapter import CoNaAdapterFewShotClassifier
cona_adapter = CoNaAdapterFewShotClassifier(
    vlm,
    lr=1e-3,
    epochs=N_EPOCHS,
    optimizer="adamw",
    adapter_lr_multiplier=0.1,
    name_regularization=1,
    adapter_regularization=1e-2,
    context_len=CONTEXT_LEN,
    batch_size=BATCH_SIZE,
    random_augment=False
)


'''
Dataset Setup
'''
from dataset import DatasetHandler
dataset = DatasetHandler(DATASET_NAME, split="all")
support_dataset = DatasetHandler(DATASET_NAME, split="train")
val_tuning_dataset = query_dataset = DatasetHandler(DATASET_NAME, split="val")

dataset.fill_cache(vlm)



# folder path for specific vlm/dataset combo
RESULTS_FOLDER = os.path.join("class_embed_shift_visualizations", dataset.id(), VLM.__name__)
os.makedirs(RESULTS_FOLDER, exist_ok=True)



'''
Train then save class embeds
'''
n_way = dataset.category_count()
n_support = 16

from FewShotTestHandler import FewShotTestHandler
test_handler = FewShotTestHandler(None)

test_handler.run_few_shot_test(coop, query_dataset, support_dataset, n_way, n_support, n_query=None, n_episodes=1, val_tuning_dataset=val_tuning_dataset)
test_handler.run_few_shot_test(cona, query_dataset, support_dataset, n_way, n_support, n_query=None, n_episodes=1, val_tuning_dataset=val_tuning_dataset)
test_handler.run_few_shot_test(cona_adapter, query_dataset, support_dataset, n_way, n_support, n_query=None, n_episodes=1, val_tuning_dataset=val_tuning_dataset)



# Set fixed order for category names, paths and associated embeddings
category_names = sorted(list(dataset.data_dict.keys()))
category_paths = [dataset.data_dict[name] for name in category_names]

# Record saved as {class name -> [orig text embed, tuned text embed epoch 0, epoch 1, ...]}
coop_text_embeds_per_category = [coop.text_embed_training_record[name] for name in category_names]
cona_text_embeds_per_category = [cona.text_embed_training_record[name] for name in category_names]
cona_adapter_text_embeds_per_category = [cona_adapter.text_embed_training_record[name] for name in category_names]

vid_embeds_per_category = [
    [
        vlm.get_video_embeds(path)
        for path in paths
    ]
    for paths in category_paths
]



for use_vids in [False, True]:
    '''
    T-SNE Embedding Compute
    '''
    # Stack embeddings to perform T-SNE over all together
    stacked_embeddings = []
    coop_text_stacked_indices = []
    cona_text_stacked_indices = []
    cona_adapter_text_stacked_indices = []
    vid_stacked_indices = []

    next_index = 0
    for coop_text_embeds, cona_text_embeds, cona_adapter_text_embeds, vid_embeds in zip(coop_text_embeds_per_category, cona_text_embeds_per_category, cona_adapter_text_embeds_per_category, vid_embeds_per_category):
        stacked_embeddings += coop_text_embeds
        coop_text_stacked_indices.append([next_index + i for i in range(len(coop_text_embeds))])
        next_index += len(coop_text_embeds)
        
        stacked_embeddings += cona_text_embeds
        cona_text_stacked_indices.append([next_index + i for i in range(len(cona_text_embeds))])
        next_index += len(cona_text_embeds)
        
        stacked_embeddings += cona_adapter_text_embeds
        cona_adapter_text_stacked_indices.append([next_index + i for i in range(len(cona_adapter_text_embeds))])
        next_index += len(cona_adapter_text_embeds)
        
        if use_vids:
            stacked_embeddings += vid_embeds
            vid_stacked_indices.append([next_index + i for i in range(len(vid_embeds))])
            next_index += len(vid_embeds)
            
    stacked_embeddings = np.array(stacked_embeddings)



    '''
    if vlm.default_similarity_metric() is Similarity.COSINE:
        sklearn_metric = "cosine"
    elif vlm.default_similarity_metric() is Similarity.DOT:
        # NOTE: This is imperfect. No distance metric can match dot-product ordering without violating triangle inequality
        # (For any 2 vectors which aren't directly opposite each other, a third vector exists with arbitrarily-high similarity to both)
        sklearn_metric = lambda a, b: math.exp(-Similarity.DOT(a[None, :], b[None, :]))
    else:
        raise ValueError("Unknown equivalent sklearn metric name")
    '''
    sklearn_metric = "cosine"

    sne_embeddings = TSNE(n_components=2, metric=sklearn_metric).fit_transform(stacked_embeddings)



    # Unstack SNE embeddings into original fixed order of embeddings
    coop_text_sne_embeds_per_category = [
        [
            sne_embeddings[stack_ind]
            for stack_ind in single_category_text_stacked_indices
        ]
        for single_category_text_stacked_indices in coop_text_stacked_indices
    ]

    cona_text_sne_embeds_per_category = [
        [
            sne_embeddings[stack_ind]
            for stack_ind in single_category_text_stacked_indices
        ]
        for single_category_text_stacked_indices in cona_text_stacked_indices
    ]

    cona_adapter_text_sne_embeds_per_category = [
        [
            sne_embeddings[stack_ind]
            for stack_ind in single_category_text_stacked_indices
        ]
        for single_category_text_stacked_indices in cona_adapter_text_stacked_indices
    ]

    if use_vids:
        vid_sne_embeds_per_category = [
            np.array([
                sne_embeddings[stack_ind]
                for stack_ind in single_category_vid_stacked_indices
            ])
            for single_category_vid_stacked_indices in vid_stacked_indices
        ]
        
        
        
    '''
    Plot text embeds alone then alongside videos 
    '''
    method_names = ["CoOp", "CoNa (ours)", "CoN-Adapter (ours)"]
    method_sne_embeds_per_category = [coop_text_sne_embeds_per_category, cona_text_sne_embeds_per_category, cona_adapter_text_sne_embeds_per_category]

    fig, axs = plt.subplots(3, N_COLS, figsize=(5 * N_COLS, 15), sharex=True, sharey=True)
    for row_ind, classifier_name, text_sne_embeds_per_category in zip(range(3), method_names, method_sne_embeds_per_category):
        for col_ind, record_ind in enumerate(np.round(np.linspace(0, N_EPOCHS, num=N_COLS, endpoint=True)).astype(int)):
            ax = axs[row_ind, col_ind]
            ax.tick_params(left = False, bottom = False, labelleft = False, labelbottom = False)
            if row_ind == 2:
                ax.set_xlabel(f"{record_ind} Epochs", fontsize=30)
            if col_ind == 0:
                ax.set_ylabel(classifier_name, fontsize=30)
            
            #ax.set_title(f"{classifier_name}: {record_ind} Epochs", fontdict={"fontsize": 20})
            for cat_ind in range(len(category_names)):
                text_embed = text_sne_embeds_per_category[cat_ind][record_ind]
                
                color = next(ax._get_lines.prop_cycler)["color"]
                ax.scatter([text_embed[0]], [text_embed[1]], marker="o", color=color, s=100)
            
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_FOLDER, "text_only.pdf" if use_vids else "text_sne.text_only.pdf"))
    plt.show()

    if use_vids:
        fig, axs = plt.subplots(3, N_COLS, figsize=(5 * N_COLS, 15), sharex=True, sharey=True)
        for row_ind, classifier_name, text_sne_embeds_per_category in zip(range(3), method_names, method_sne_embeds_per_category):
            for col_ind, record_ind in enumerate(np.round(np.linspace(0, N_EPOCHS, num=N_COLS, endpoint=True)).astype(int)):
                ax = axs[row_ind, col_ind]
                ax.tick_params(left = False, bottom = False, labelleft = False, labelbottom = False)
                if row_ind == 2:
                    ax.set_xlabel(f"{record_ind} Epochs", fontsize=30)
                if col_ind == 0:
                    ax.set_ylabel(classifier_name, fontsize=30)
                
                #ax.set_title(f"{classifier_name}: {record_ind} Epochs", fontdict={"fontsize": 20})
                for cat_ind in range(len(category_names)):
                    text_embed = text_sne_embeds_per_category[cat_ind][record_ind]
                    vid_embeds = vid_sne_embeds_per_category[cat_ind]
                    
                    color = next(ax._get_lines.prop_cycler)["color"]
                    ax.scatter(vid_embeds[:, 0], vid_embeds[:, 1], marker="x", color=color, alpha=0.1, s=10)
                    ax.scatter([text_embed[0]], [text_embed[1]], marker="o", color=color, s=100)
                
        plt.tight_layout()
        fig.savefig(os.path.join(RESULTS_FOLDER, f"text_plus_vids.pdf"))
        plt.show()