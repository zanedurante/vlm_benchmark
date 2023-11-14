import numpy as np
from copy import deepcopy
from typing import Optional
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

"""
A simple example script for running the vl_prompt classifier over multiple GPUs.
"""



def distributed_run(rank, world_size):
    if world_size > 1:
        torch.cuda.set_device(rank)
        torch.distributed.init_process_group(backend="nccl", world_size=world_size, rank=rank)
        
    from FewShotTestHandler import FewShotTestHandler
    from dataset import DatasetHandler
    from VIFI_CLIP import ViFiCLIP_SimilarityVLM as VLM
    from classifier import VLPromptFewShotClassifier as Classifier
    
    test_handler = FewShotTestHandler("temp.csv")
    
    name = sys.argv[1]
    support_dataset = DatasetHandler(name, split="train")
    query_dataset = DatasetHandler(name, split="test")
    
    vlm = VLM(
        num_frames=32,
        load_vifi=True
    )
    
    classifier = Classifier(
        vlm,
        text_context_len=10,
        text_context_depth=12,
        vision_context_len=10,
        vision_context_depth=12,
        lr=8e-3,
        batch_size=4,
        accumulation_steps=64 // (4 * world_size), # Set accumulation steps to be equivalent to ViFiCLIP's batch-size of 64
        epochs=50,
        optimizer="adamw"
    )
    
    test_handler.run_few_shot_test(
        classifier,
        query_dataset,
        support_dataset,
        n_way = support_dataset.category_count(),
        n_support = int(sys.argv[2]),
        n_query = None,
        n_episodes = 1,
        val_tuning_dataset = None
    )
    
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    
    
    
    
if __name__ == "__main__":    
    world_size = 1#torch.cuda.device_count()
    
    if world_size > 1:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "6768"
        torch.multiprocessing.spawn(
            distributed_run,
            args=(world_size,),
            nprocs=world_size,
            join=True
        )
    else:
        distributed_run(0, 1)