import os, sys
from typing import Optional
import numpy as np
import decord
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
EMBED_DIM = 256
VIDEO_TOKEN_DIM = 768
VIDEO_NUM_FRAME_PATCHES = (224 // 16)**2
VIDEO_NUM_FRAMES = 4

# Given MILES eval config just uses the default transforms
# 'test' key gives constant transforms, 'train' key gives randomized tranforms for augmentation
VIDEO_TRANSFORM_DICT = init_transform_dict()

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
        
        super().__init__(cache_file=os.path.join(FILE_DIR, CACHE_NAME), reset_cache=reset_cache)
        
    def text_encoder(self, text: str) -> np.ndarray:
        """
        Tokenize and encode text into a joint text/video embedding space
        :param text:
        :return:
        """
        # Tokenize
        inputs = self.tokenizer([text], return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        # Process
        with torch.no_grad():
            text_embed = self.model.compute_text(inputs)[0].cpu().numpy()
            
        return text_embed

    def video_encoder(self, video_path: str, subvideo_start_frame: Optional[int] = None, subvideo_end_frame: Optional[int] = None) -> np.ndarray:
        """
        Load, transform and encode a video file into a joint text/video embedding space
        :param video:
        :param subvideo_start_frame:
        :param subvideo_end_frame:
        :return:
        """
        # Load frames
        video_reader = decord.VideoReader(video_path, num_threads=1)
        video_len = len(video_reader)
        frame_indices = self.sample_frame_indices(video_len, subvideo_start_frame, subvideo_end_frame)
        frames = video_reader.get_batch(frame_indices).asnumpy()

        # Transform
        frames = torch.from_numpy(frames).float() / 255
        frames = frames.permute(0, 3, 1, 2)
        frames = VIDEO_TRANSFORM_DICT["test"](frames)
        video_input = torch.zeros(VIDEO_NUM_FRAMES, 3, INPUT_RES, INPUT_RES)
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
    
    def sample_frame_indices(self, video_len: int, subvideo_start_frame: Optional[int] = None, subvideo_end_frame: Optional[int] = None) -> np.ndarray:
        start_frame = subvideo_start_frame or 0
        end_frame = subvideo_end_frame or video_len
            
        frame_indices = np.minimum(
            np.round(
                np.linspace(start_frame, end_frame, VIDEO_NUM_FRAMES, endpoint=False) + (end_frame - start_frame) / (2 * VIDEO_NUM_FRAMES)
            ),
            end_frame - 1
        )
        return frame_indices