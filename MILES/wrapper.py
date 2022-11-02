import os, sys
from typing import Optional
import numpy as np
import decord
from transformers import AutoTokenizer

import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from SimilarityVLM import SimilarityVLM
from similarity_metrics import Similarity
from .token_exposed_implementation import MILES_ExposedTokens

FILE_DIR = os.path.dirname(os.path.realpath(__file__))

REPO_DIR = os.path.join(FILE_DIR, "MCQ")
sys.path.append(REPO_DIR)
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
VIDEO_FRAME_PATCH_SIZE = 16
VIDEO_NUM_FRAME_PATCHES = (224 // VIDEO_FRAME_PATCH_SIZE)**2
VIDEO_TRAIN_NUM_FRAMES = 4 # Number of frames used for pretraining. Video encoder learned this many temporal embeddings

# Given MILES eval config just uses the default transforms
# 'test' transforms have no randomization, 'train' transforms have random resize crop, random horizontal flip, and random color jitter
VIDEO_TRANSFORM_DICT = init_transform_dict()

# Pretrained State
PRETRAINED_CHECKPOINT_PATH = os.path.join(FILE_DIR, "pretrained/MILES.pth")

# Cache file location
CACHE_NAME = "cache"

class MILES_SimilarityVLM(SimilarityVLM):
    def __init__(self, num_frames: int = 4, reset_cache: bool = False):
        self.num_frames = int(num_frames)
        
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ARGS['text_params']['model'])
        self.model = MILES_ExposedTokens(**MODEL_ARGS)
        
        # Load pretrained model checkpoint
        checkpoint = torch.load(PRETRAINED_CHECKPOINT_PATH, map_location="cpu")
        state_dict = checkpoint['state_dict']
        new_state_dict = state_dict_data_parallel_fix(state_dict, self.model.state_dict())
        self.model.load_state_dict(new_state_dict, strict=False)
        
        self.model.to(DEVICE)
        
        super().__init__(cache_file=os.path.join(FILE_DIR, CACHE_NAME), reset_cache=reset_cache)
        
    def params(self) -> dict:
        return {
            "num_frames": self.num_frames
        }
        
    def embed_dim(self):
        return EMBED_DIM
    
    def video_token_dim(self):
        return VIDEO_TOKEN_DIM
    
    def video_frame_patch_size(self):
        return VIDEO_FRAME_PATCH_SIZE
    
    def video_num_frame_patches(self):
        return VIDEO_NUM_FRAME_PATCHES
    
    def video_num_frames(self):
        return self.num_frames
    
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

    def video_encoder_to_tokens(self, video_path: str, subvideo_start_frame: Optional[int] = None, subvideo_end_frame: Optional[int] = None, random_augment: bool = False) -> torch.Tensor:
        """Converts a single video into a 1-batch tensor of tokens in the internal model format.

        Args:
            video_path (str): _description_
            subvideo_start_frame (Optional[int], optional): _description_. Defaults to None.
            subvideo_end_frame (Optional[int], optional): _description_. Defaults to None.

        Returns:
            torch.Tensor: video patch tokens in format (batch = 1, frames, patches, token_dim)
        """
        # Load frames
        video_reader = decord.VideoReader(video_path, num_threads=1)
        video_len = len(video_reader)
        frame_indices = self.sample_frame_indices(video_len, subvideo_start_frame, subvideo_end_frame, random_offset=random_augment)
        frames = video_reader.get_batch(frame_indices).asnumpy()

        # Transform
        frames = torch.from_numpy(frames).float() / 255
        frames = frames.permute(0, 3, 1, 2)
        if random_augment:
            frames = VIDEO_TRANSFORM_DICT["train"](frames)
        else:
            frames = VIDEO_TRANSFORM_DICT["test"](frames)
        video_input = torch.zeros(self.num_frames, 3, INPUT_RES, INPUT_RES)
        video_input[:frames.shape[0]] = frames # Zero-pad frames to desired length
        video_input = video_input.unsqueeze(0)
        video_input = video_input.to(DEVICE)
        
        # Partial model converts frames to space/time patch tokens
        video_tokens = self.model.compute_video_to_tokens(video_input)
        return video_tokens
    
    def video_encoder_from_tokens(self, video_tokens: torch.Tensor, prompt_tokens: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Converts a batch of video token tensors in internal model format into a batch of video embeddings

        Args:
            video_tokens (torch.Tensor):                        Video patch tokens in format (batch, frames, patches, token_dim)
            prompt_tokens (Optional[torch.Tensor], optional):   Optional token embeddings which will be inserted between [CLS] token
                                                                and video tokens. Shape = (prompt_tokens, token_dim). Defaults to None.

        Returns:
            torch.Tensor: video embeddings in format (batch, embed_dim)
        """
        video_embeds = self.model.compute_video_from_tokens(video_tokens, prompt_tokens)
        return video_embeds

    def video_encoder(self, video_path: str, subvideo_start_frame: Optional[int] = None, subvideo_end_frame: Optional[int] = None, random_augment: bool = False) -> np.ndarray:
        """
        Load, transform and encode a single video file into a joint text/video embedding space.
        When doing this in one stage, gradients are disabled. As opposed to manually calling to_token and from_token stages.
        :param video:
        :param subvideo_start_frame:
        :param subvideo_end_frame:
        :return:
        """
        
        # Process
        with torch.no_grad():
            video_tokens = self.video_encoder_to_tokens(video_path, subvideo_start_frame, subvideo_end_frame, random_augment)
            video_embed = self.video_encoder_from_tokens(video_tokens)
            
        # Unbatch and move to cpu
        video_embed = video_embed[0].cpu().numpy()
            
        return video_embed

    def default_similarity_metric(self) -> Similarity:
        """
        Returns a reference to the default similarity metric used by this VLM
        :return:
        """
        return Similarity.COSINE
    
    def sample_frame_indices(self, video_len: int, subvideo_start_frame: Optional[int] = None, subvideo_end_frame: Optional[int] = None, random_offset: bool = False) -> np.ndarray:
        start_frame = subvideo_start_frame or 0
        end_frame = subvideo_end_frame or video_len
            
        # Number of frame indices between consecutive samples (if no random offset augmentation)
        sample_width = (end_frame - start_frame) / self.num_frames
        
        if not random_offset:        
            frame_index_offset = sample_width / 2
        else:
            frame_index_offset = np.random.uniform(0, sample_width, self.num_frames)
            
        frame_indices = np.minimum(
            np.round(
                np.linspace(start_frame, end_frame, self.num_frames, endpoint=False) + frame_index_offset
            ),
            end_frame - 1
        )
        return frame_indices