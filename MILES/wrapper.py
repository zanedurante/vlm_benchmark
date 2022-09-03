import os, sys
from transformers import AutoTokenizer

import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from SimilarityVLM import SimilarityVLM
from similarity_metrics import Similarity

FILE_DIR = os.path.dirname(os.path.realpath(__file__))

REPO_DIR = os.path.join(FILE_DIR, "MCQ")
sys.path.append(REPO_DIR)
from .MCQ.MILES.model.model_MILES import MILES
from .MCQ.utils.util import state_dict_data_parallel_fix
from .MCQ.base.base_dataset import read_frames_cv2
from .MCQ.MILES.data_loader.transforms import init_transform_dict

# Model Args, adapted from MCQ/MILES/configs/zero_msrvtt_4f_i21k_MILES.json
MODEL_ARGS = {
    "video_params": {
        "model": "SpaceTimeTransformer",
        "arch_config": "base_patch16_224_temporal",
        "num_frames": 4,
        "pretrained": True
    },
    "text_params": {
        "model": "distilbert-base-uncased",
        "pretrained": True,
        "input": "text"
    },
    "projection": "minimal",
    "load_checkpoint" : None # Built-in checkpoint loading doesn't work on cpu
}
INPUT_RES = 224
NUM_FRAMES = 4

# Given MILES eval config just uses the default transforms
VIDEO_TRANSFORM = init_transform_dict()["test"]

# Pretrained State
PRETRAINED_CHECKPOINT_PATH = os.path.join(FILE_DIR, "pretrained/MILES.pth")

# Cache file location
CACHE_NAME = "cache"

class MILES_SimilarityVLM(SimilarityVLM):
    def __init__(self, reset_cache: bool = False):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ARGS['text_params']['model'])
        self.model = MILES(**MODEL_ARGS)
        
        # Load pretrained model checkpoint
        checkpoint = torch.load(PRETRAINED_CHECKPOINT_PATH, map_location="cpu")
        state_dict = checkpoint['state_dict']
        new_state_dict = state_dict_data_parallel_fix(state_dict, self.model.state_dict())
        self.model.load_state_dict(new_state_dict, strict=False)
        
        self.model.to(DEVICE)
        
        # Video transform
        
        super().__init__(cache_file=os.path.join(FILE_DIR, CACHE_NAME), reset_cache=reset_cache)
        
    def text_encoder(self, text):
        """
        Tokenize and encode text into a joint text/video embedding space
        :param tokens:
        :return:
        """
        # Tokenize
        inputs = self.tokenizer([text], return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        # Process
        with torch.no_grad():
            text_embed = self.model.compute_text(inputs)[0].cpu().numpy()
            
        return text_embed

    def video_encoder(self, video_path):
        """
        Load, transform and encode a video file into a joint text/video embedding space
        :param video:
        :return:
        """
        # Load frames
        frames, frame_indices = read_frames_cv2(video_path, NUM_FRAMES, "uniform", fix_start=None)

        # Transform
        frames = VIDEO_TRANSFORM(frames)
        video_input = torch.zeros(NUM_FRAMES, 3, INPUT_RES, INPUT_RES)
        video_input[:frames.shape[0]] = frames # Zero-pad frames to desired length
        video_input = video_input.unsqueeze(0)
        video_input = video_input.to(DEVICE)
        
        # Process
        with torch.no_grad():
            vid_embed = self.model.compute_video(video_input)[0].cpu().numpy()
            
        return vid_embed

    def default_similarity_metric(self) -> Similarity:
        """
        Returns a reference to the default similarity metric used by this VLM
        :return:
        """
        return Similarity.COSINE