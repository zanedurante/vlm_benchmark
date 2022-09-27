# TODO: Cleanup imports

import numpy as np
import random
import os
import argparse
from torchvision.transforms import Compose, Lambda, CenterCrop
import torch
import math
import decord
import pdb
import argparse
from importlib import import_module
import pandas as pd
import transformers
from sacred import Experiment
from tqdm import tqdm
import glob
import json
import sys


# Need to import like this, since directory structure is frozen-in-time
module_data = import_module(".frozen-in-time.data_loader.data_loader", package="frozen")
module_metric = import_module(".frozen-in-time.model.metric", package="frozen")
model_arch = import_module(".frozen-in-time.model.model", package="frozen")
parse_config = import_module(".frozen-in-time.parse_config", package="frozen")
trainer = import_module(".frozen-in-time.trainer.trainer", package="frozen")
utils = import_module(".frozen-in-time.utils", package="frozen")
transforms = import_module(".frozen-in-time.data_loader.transforms", package="frozen")

# Add files to sys.modules to allow for torch model to be loaded despite
# the unusual import scheme above
sys.modules["parse_config"] = parse_config

from SimilarityVLM import SimilarityVLM
from similarity_metrics import Similarity

# Default cache locations
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_CUDA = DEVICE == "cuda"
FILE_DIR = os.path.dirname(os.path.realpath(__file__))
CACHE_NAME = "cache"


class FrozenVLM(SimilarityVLM):
    """
    Similarity-based VLM that uses Frozen-in-Time for frame and text encoders.
    TODO: Use different sampling schemes for frames in the video (currently use first 4 by default)
    TODO: Investigate using > 4 frames from video
    """
    def __init__(self, config_path: str = "frozen/config.json",
                 use_cuda: bool = False, reset_cache: bool = False):
        """
        :param config_path: Path to the frozen in time config file
        :param use_cuda: Whether to use cuda for GPU, if false uses CPU
        :param reset_cache: Whether to reset the embedding cache
        """
        super().__init__(cache_file=os.path.join(FILE_DIR, CACHE_NAME), reset_cache=reset_cache)

        self.cuda = use_cuda
        self.config_path = config_path  # Config for pre-trained Frozen VLM
        self.config = None
        self.set_config_vals()

        self.model = None
        self.tokenizer = None
        self.load_model()

        self.tfms = self.get_transforms()
        decord.bridge.set_bridge("torch")  # Video loader



    def params(self) -> dict:
        """
        Specify the value of all VLM-specific parameters which may affect prediction accuracy.
        This is used to differentiate test results which use different versions of the same VLM.
        :return:
        :rtype: dict
        """
        return {
            "config_path": self.config_path,
        }

    def set_config_vals(self):
        args = argparse.ArgumentParser(description='PyTorch Template')

        args.add_argument('-r', '--resume', default=None, type=str,
                          help='path to latest checkpoint (default: None)')
        args.add_argument('-d', '--device', default=None, type=str,
                          help='indices of GPUs to enable (default: all)')
        args.add_argument('-c', '--config', default=self.config_path, type=str,
                          help='config file path (default: None)')
        args.add_argument('-s', '--sliding_window_stride', default=-1, type=int,
                          help='test time temporal augmentation, repeat samples with different start times.')
        #args.add_argument('-g', '--gpu', default=False, type=bool,
        #                  help='whether or not to use GPUs while training and testing, if false the model runs on CPU.')
        args.add_argument('--save_feats', default=None,
                          help='path to store text & video feats, this is for saving embeddings if you want to do offline retrieval.')
        args.add_argument('--save_type', default='both', choices=['both', 'text', 'video'],
                          help='Whether to save video, text or both feats. If running on inference videos, text is just a placeholder')
        args.add_argument('--vis_token_similarity', action='store_true')
        args.add_argument('--split', default='test', choices=['train', 'val', 'test'],
                          help='split to evaluate on.')
        args.add_argument('--batch_size', default=16, type=int,
                          help='size of batch')
        config = parse_config.ConfigParser(args, test=True)

        # Set config values to be correct, even though these aren't used...
        config._config['data_loader'] = config._config['data_loader'][0]
        config._config['data_loader']['args']['tsfm_split'] = 'test'  # set transform to test split to remove augmentations
        self.config = config
        return

    def load_model(self):
        """
        Loads the model from the weights specified in `self.config_path`
        :return:
        """
        text_model_name = self.config['arch']['args']['text_params']['model']
        if "openai/clip" in text_model_name:
                tokenizer_builder = transformers.CLIPTokenizer
        else:
            tokenizer_builder = transformers.AutoTokenizer
        self.tokenizer = tokenizer_builder.from_pretrained(
            text_model_name,
            model_max_length=self.config['arch']['args']['text_params'].get('max_length', 1e6),
            TOKENIZERS_PARALLELISM=False)

        self.model = self.config.initialize('arch', model_arch, use_cuda=USE_CUDA)

        if self.config['n_gpu'] > 1 and USE_CUDA:
            self.model = torch.nn.DataParallel(self.model).module # Add .module allows for support on GPU

        device = torch.device(DEVICE)
        self.model = self.model.to(device)
        self.model.eval()

        return

    def tokenize(self, text):
        """
        Tokenizes text via tokenizer (likely variant of huggingface BertTokenizer)
        :param text:, list of text to tokenize
        :return: Tokenized text
        """
        tokenized_text = []
        for t in text:
            tokenized_text.append(self.tokenizer(t, return_tensors='pt', padding=True, truncation=False))
        return tokenized_text

    def text_encoder(self, text):
        """
        Encodes tokenized text into joint text/video embedding space
        :param text:
        :return:
        """

        tokens = self.tokenize([text])[0]
        with torch.no_grad():
            text_embed = self.model.compute_text(tokens).squeeze().numpy()

        return text_embed

    def open_video(self, path):
        """
        Opens video and returns basic, non-transformed video tensor
        :param path:
        :return:
        """
        video_reader = decord.VideoReader(path, num_threads=1)
        fps = video_reader.get_avg_fps()
        video_length = len(video_reader)

        # Frozen takes tensors in input shape b, t, c, h, w
        video_tensor = video_reader.get_batch(self.get_indices())
        video_tensor = video_tensor.float() / 255.0

        # t, c, h, w
        video_tensor = video_tensor.permute(0, 3, 1, 2)

        return video_tensor

    def transform(self, video):
        """
        Transforms video using model-specific transforms
        :param video:
        :return:
        """
        inputs = self.tfms(video)
        inputs = inputs.unsqueeze(0) # Add batch dim

        return inputs

    def video_encoder(self, path):
        """
        Encodes transformed video into joint text/video embedding space
        :param path:
        :return:
        """
        video = self.open_video(path)
        video = self.transform(video)

        with torch.no_grad():
            video_features = self.model.compute_video(video)
            video_features = video_features.cpu().numpy().squeeze()
        return video_features

    def default_similarity_metric(self) -> Similarity:
        """
        Returns a reference to the default similarity metric used by this VLM
        :return:
        """
        return Similarity.DOT

    def get_transforms(self):
        # Input is T, H, W, C
        tfms = transforms.init_transform_dict()["test"]
        return tfms

    # TODO: Implement more complex sampling strategies, similar to VideoCLIP
    def get_indices(self):
        return [1, 2, 3, 4]


