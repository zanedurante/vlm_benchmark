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
VIDEO_NUM_FRAME_PATCHES = (224 // 16)**2
VIDEO_NUM_FRAMES = 4

# Given MILES eval config just uses the default transforms
VIDEO_TRANSFORM = init_transform_dict()["test"]

# Pretrained State
PRETRAINED_CHECKPOINT_PATH = os.path.join(FILE_DIR, "pretrained/MILES.pth")

# Cache file location
CACHE_NAME = "cache"

class MILES_SimilarityVLM(SimilarityVLM):
    def __init__(self, reset_cache: bool = False):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ARGS['text_params']['model'])
        self.model = MILES_ExposedTokens(**MODEL_ARGS)
        
        # Load pretrained model checkpoint
        checkpoint = torch.load(PRETRAINED_CHECKPOINT_PATH, map_location="cpu")
        state_dict = checkpoint['state_dict']
        new_state_dict = state_dict_data_parallel_fix(state_dict, self.model.state_dict())
        self.model.load_state_dict(new_state_dict, strict=False)
        
        self.model.to(DEVICE)
        
        super().__init__(cache_file=os.path.join(FILE_DIR, CACHE_NAME), reset_cache=reset_cache)
        
    def embed_dim(self):
        return EMBED_DIM
    
    def video_token_dim(self):
        return VIDEO_TOKEN_DIM
    
    def video_num_frame_patches(self):
        return VIDEO_NUM_FRAME_PATCHES
    
    def video_num_frames(self):
        return VIDEO_NUM_FRAMES
    
    def text_encoder_to_tokens(self, text: str) -> torch.Tensor:
        """Converts a single text string into a 1-batch tensor of token embeddings and a corresponding attention mask.

        Args:
            text (str): _description_

        Returns:
            torch.Tensor: input token embeddings for the text encoder. Shape (batch = 1, sequence_len, token_dim)
            torch.Tensor: input sequence attention mask for the text encoder. Shape (batch = 1, sequence_len)
        """
        inputs = self.tokenizer([text], return_tensors="pt", padding=True, truncation=True)
        token_ids = inputs["input_ids"].to(DEVICE)
        attn_mask = inputs["attention_mask"].to(DEVICE)
        
        # Convert token ids to token embeddings
        token_embeds = self.model.text_model.get_input_embeddings()(token_ids)
        
        return token_embeds, attn_mask
    
    def text_encoder_from_tokens(self, token_embeds: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        """Converts a batch of token embeddings and corresponding attention masks into a batch of text embeddings.

        Args:
            token_embeds (torch.Tensor): Shape (batch, sequence_len, token_dim)
            attn_mask (torch.Tensor): Shape (batch, sequence_len)

        Returns:
            torch.Tensor: Shape (batch, embed_dim)
        """
        return self.model.compute_text_from_token_embeds(token_embeds=token_embeds, attn_mask=attn_mask)
        
        
    def text_encoder(self, text: str) -> np.ndarray:
        """
        Tokenize and encode text into a joint text/video embedding space.
        When doing this in one stage, gradients are disabled. As opposed to manually calling to_token and from_token stages.
        :param text:
        :return:
        """
        # Process
        with torch.no_grad():
            token_embeds, attn_mask = self.text_encoder_to_tokens(text)
            text_embed = self.text_encoder_from_tokens(token_embeds, attn_mask)
            
        # Unbatch and move to cpu
        text_embed = text_embed[0].cpu().numpy()
            
        return text_embed

    def video_encoder_to_tokens(self, video_path: str, subvideo_start_frame: Optional[int] = None, subvideo_end_frame: Optional[int] = None) -> torch.Tensor:
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
        frame_indices = self.sample_frame_indices(video_len, subvideo_start_frame, subvideo_end_frame)
        frames = video_reader.get_batch(frame_indices).asnumpy()

        # Transform
        frames = torch.from_numpy(frames).float() / 255
        frames = frames.permute(0, 3, 1, 2)
        frames = VIDEO_TRANSFORM(frames)
        video_input = torch.zeros(VIDEO_NUM_FRAMES, 3, INPUT_RES, INPUT_RES)
        video_input[:frames.shape[0]] = frames # Zero-pad frames to desired length
        video_input = video_input.unsqueeze(0)
        video_input = video_input.to(DEVICE)
        
        # Partial model converts frames to space/time patch tokens
        video_tokens = self.model.compute_video_to_tokens(video_input)
        return video_tokens
    
    def video_encoder_from_tokens(self, video_tokens: torch.Tensor) -> torch.Tensor:
        """Converts a batch of video token tensors in internal model format into a batch of video embeddings

        Args:
            video_tokens (torch.Tensor): video patch tokens in format (batch, frames, patches, token_dim)

        Returns:
            torch.Tensor: video embeddings in format (batch, embed_dim)
        """
        video_embeds = self.model.compute_video_from_tokens(video_tokens)
        return video_embeds

    def video_encoder(self, video_path: str, subvideo_start_frame: Optional[int] = None, subvideo_end_frame: Optional[int] = None) -> np.ndarray:
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
            video_tokens = self.video_encoder_to_tokens(video_path, subvideo_start_frame, subvideo_end_frame)
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