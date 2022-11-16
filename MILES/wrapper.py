import os, sys
from typing import Optional, List
import numpy as np
from transformers import AutoTokenizer

import torch
import decord
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
INPUT_WORD_EMBED_DIM = 768
VIDEO_NUM_FRAME_PATCHES = (224 // 16)**2
VIDEO_NUM_FRAMES = 4
LOGIT_SCALE = 20

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
        
        self.model.text_model.eval()
        self.model.video_model.eval()
        
        super().__init__(cache_file=os.path.join(FILE_DIR, CACHE_NAME), reset_cache=reset_cache)
        
    def logit_scale(self) -> float:
        return LOGIT_SCALE
        
    def input_word_embed_dim(self) -> int:
        return INPUT_WORD_EMBED_DIM
    
    def text_start_special_token_count(self) -> int:
        return 1
    
    def text_end_special_token_count(self) -> int:
        return 1
        
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
    
    def get_input_word_embeddings(self, text_list: List[str]) -> torch.Tensor:
        """Converts a list of text string into a batched tensor of input word embeddings and a corresponding attention mask,
        including special tokens.

        Args:
            text_list (str): _description_

        Returns:
            torch.Tensor: input token embeddings for the text encoder. Shape (batch, sequence_len, token_dim)
            torch.Tensor: input sequence attention mask for the text encoder. Shape (batch, sequence_len)
        """
        inputs = self.tokenizer(text_list, return_tensors="pt", padding=True, truncation=True)
        token_ids = inputs["input_ids"].to(DEVICE)
        attn_mask = inputs["attention_mask"].to(DEVICE)
        
        #print(" ".join([self.tokenizer.decode(id) for id in token_ids[0]]))
        
        # Convert token ids to token embeddings
        input_word_embeds = self.model.text_model.embeddings.word_embeddings(token_ids)
        
        return input_word_embeds, attn_mask
    
    def text_encoder_from_word_embeddings(self, input_word_embeds: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        """Converts a batch of token embeddings and corresponding attention masks into a batch of text embeddings.

        Args:
            token_embeds (torch.Tensor): Shape (batch, sequence_len, token_dim)
            attn_mask (torch.Tensor): Shape (batch, sequence_len)

        Returns:
            torch.Tensor: Shape (batch, embed_dim)
        """
        batch_size, seq_len, embed_dim = input_word_embeds.shape
        
        # Add positional embeddings and layer norm
        pos_ids = torch.arange(seq_len, dtype=torch.long, device=DEVICE).expand(batch_size, seq_len)
        pos_embeds = self.model.text_model.embeddings.position_embeddings(pos_ids)
        input_embeds = input_word_embeds + pos_embeds
        input_embeds = self.model.text_model.embeddings.LayerNorm(input_embeds)
        
        text_embeds = self.model.text_model(inputs_embeds=input_embeds, attention_mask=attn_mask).last_hidden_state[:, 0, :]
        text_embeds = self.model.text_proj(text_embeds)
        return text_embeds
    
    def text_encoder_over_embeds(self, text: str) -> torch.Tensor:
        with torch.no_grad():
            input_word_embeds, attn_mask = self.get_input_word_embeddings([text])
            return self.text_encoder_from_word_embeddings(input_word_embeds, attn_mask)[0].cpu().numpy()

    def video_encoder(self, video_path: str, subvideo_start_frame: Optional[int] = None, subvideo_end_frame: Optional[int] = None, random_augment: bool = False) -> np.ndarray:
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
        frame_indices = self.sample_frame_indices(video_len, subvideo_start_frame, subvideo_end_frame, random_augment)
        frames = video_reader.get_batch(frame_indices).asnumpy()

        # Transform
        frames = torch.from_numpy(frames).float() / 255
        frames = frames.permute(0, 3, 1, 2)
        if random_augment:
            frames = VIDEO_TRANSFORM_DICT["train"](frames)
        else:
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
    
    def sample_frame_indices(self, video_len: int, subvideo_start_frame: Optional[int] = None, subvideo_end_frame: Optional[int] = None, random_augment: bool = False) -> np.ndarray:
        start_frame = subvideo_start_frame or 0
        end_frame = subvideo_end_frame or video_len
            
        if random_augment:
            frame_offset = np.random.uniform(0, (end_frame - start_frame) / VIDEO_NUM_FRAMES, size=VIDEO_NUM_FRAMES)
        else:
            frame_offset = (end_frame - start_frame) / (2 * VIDEO_NUM_FRAMES)
            
        frame_indices = np.minimum(
            np.round(
                np.linspace(start_frame, end_frame, VIDEO_NUM_FRAMES, endpoint=False) + frame_offset
            ),
            end_frame - 1
        )
        return frame_indices