import numpy as np
import random
import os
from typing import Optional, List

import torch
from .MMPT_updated.mmpt.models import MMPTClassifier

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from SimilarityVLM import SimilarityVLM
from similarity_metrics import Similarity

from torchvision.io import read_video
from pytorchvideo.transforms import *
from torchvision.transforms import Compose, Lambda, CenterCrop, RandomHorizontalFlip

import math
import decord
import pdb

# Default cache locations
FILE_DIR = os.path.dirname(os.path.realpath(__file__))
CACHE_NAME = "cache"


class VideoClipVLM(SimilarityVLM):
    """
    Similarity-based VLM that uses VideoCLIP for frame and text encoders.  This uses our own modification of the FAIR
    repository MMPT (original repo link is here: https://github.com/facebookresearch/fairseq/tree/main/examples/MMPT).
    """
    def __init__(self, path: str = "video_clip/MMPT_updated/projects/retri/videoclip/how2.yaml",
                 num_seconds: int = 2, sample_strat: str = "center", 
                 use_cuda: bool = False, reset_cache: bool = False):

        """
        :param path: Path to the videoclip config file (see setup.txt)
        :param num_seconds: Number of seconds to use in the video during inference, converts to 30fps
        :param sample_strat: Method for sampling frames from the video. Options: "center", "start", "spread"
        :param use_cuda: Whether to use cuda for GPU (if available), if false uses CPU
        :param reset_cache: Whether to reset the embedding cache
        """

        self.path = str(path)  # Pretrained video clip identifier
        self.num_seconds = int(num_seconds)
        self.sample_strat = str(sample_strat)
        self.use_cuda = bool(use_cuda)

        self.model = None        
        self.cuda = use_cuda and DEVICE == "cuda"
        self.transforms = self.get_transforms()
        self.train_transforms = self.get_train_transforms()
        decord.bridge.set_bridge("torch")  # Video loader
                
        # Do not load model, this is just dummy model to access methods
        if path is None:
            print("Dummy model loaded, no backbone or weights!")
            return
        
        assert type(self.path) is str
        assert type(self.num_seconds) is int
        assert self.sample_strat in ["center", "start", "spread"]

        
        # Load model
        self.load_model(path=self.path)

        super().__init__(cache_file=os.path.join(FILE_DIR, CACHE_NAME), reset_cache=reset_cache)


    def params(self) -> dict:
        """
        Specify the value of all VLM-specific parameters which may affect prediction accuracy.
        This is used to differentiate test results which use different versions of the same VLM.
        :return:
        :rtype: dict
        """
        return {
            "path": self.path,
            "num_seconds": self.num_seconds,
            "sample_strat": self.sample_strat,
            "use_cuda": self.use_cuda
        }
        
    def logit_scale(self) -> float:
        return 1
        
    def input_word_embed_dim(self) -> int:
        return 768
    
    def text_start_special_token_count(self) -> int:
        return 2
    
    def text_end_special_token_count(self) -> int:
        return 1

    def load_model(self, path="video_clip/MMPT_updated/projects/retri/videoclip/how2.yaml"):
        """
        Loads the model from the weights specified in `path`
        :param path:
        :return:
        """
        print("PATH IS:", path) # /home/zaned/code/vlm_benchmark/video_clip/MMPT_updated/projects/retri/videoclip/how2.yaml
        ckpt_save_dir=path[:path.rfind("/")] # Files stored in retri/videoclip repo
        ckpt_save_dir = ckpt_save_dir.replace("projects", "runs")
        print("CKPT SAVE DIR:", ckpt_save_dir) # /home/zaned/code/vlm_benchmark/video_clip/MMPT_updated/projects/retri/videoclip
        # Target: /home/zaned/code/vlm_benchmark/video_clip/MMPT_updated/runs/retri/videoclip/checkpoint_best.pt
        self.model = MMPTClassifier.from_pretrained(path, embed_extractor=True, ckpt_save_dir=ckpt_save_dir,
                                                    use_cuda=self.cuda)

        # Load random caps/cmasks for VideoCLIP so that video embeddings can be run without
        # needing to extract text embeddings first.  VideoCLIP requires both text and video inputs
        # at inference time, but uses attention mechanisms to prevent cross-modal leakage. We abstract
        # this away here.
        random_text = "random text"
        caps, cmasks = self.model.aligner._build_text_seq(
            self.model.tokenizer(random_text, add_special_tokens=False)["input_ids"])
        caps, cmasks = caps[None, :], cmasks[None, :]
        
        if self.cuda:
            self.model.caps = caps.to(DEVICE)
            self.model.cmasks = cmasks.to(DEVICE)
            self.model.to(DEVICE)
        else:
            self.model.caps = caps
            self.model.cmasks = cmasks

        return

    def tokenize(self, text):
        """
        Tokenizes text via tokenizer (likely variant of huggingface BertTokenizer)
        :param text: list of text to tokenize
        :return:
        """
        token_ids, attn_masks = [], []
        for t in text:
            caps, cmasks = self.model.aligner._build_text_seq(self.model.tokenizer(t, add_special_tokens=False)["input_ids"])
            caps, cmasks = caps[None, :], cmasks[None, :]
            token_ids.append(caps)
            attn_masks.append(cmasks)
        token_ids = torch.cat(token_ids, dim=0)
        attn_masks = torch.cat(attn_masks, dim=0)
            
        return token_ids, attn_masks

    def text_encoder(self, text):
        """
        Encodes tokenized text into joint text/video embedding space
        :param text:
        :return:
        """
        return self.text_encoder_over_embeds(text)
    
    def get_input_word_embeddings(self, text_list: List[str]) -> torch.Tensor:
        """Converts a list of text string into a batched tensor of input word embeddings and a corresponding attention mask,
        including special tokens.

        Args:
            text_list (str): _description_

        Returns:
            torch.Tensor: input token embeddings for the text encoder. Shape (batch, sequence_len, token_dim)
            torch.Tensor: input sequence attention mask for the text encoder. Shape (batch, sequence_len)
        """
        text_tokens, text_mask = self.tokenize(text_list)
        
        if self.cuda:
            text_tokens = text_tokens.to(DEVICE)
            text_mask = text_mask.to(DEVICE)
            
        text_input_embeds = self.model.mmpt_model.model.text_encoder.embeddings.word_embeddings(text_tokens)
        
        max_text_len = torch.max(torch.sum(text_mask, dim=1))
        text_input_embeds = text_input_embeds[:, :max_text_len, :]
        text_mask = text_mask[:, :max_text_len]
        return text_input_embeds, text_mask
            
    
    def text_encoder_from_word_embeddings(self, input_word_embeds: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        """Converts a batch of token embeddings and corresponding attention masks into a batch of text embeddings.

        Args:
            token_embeds (torch.Tensor): Shape (batch, sequence_len, token_dim)
            attn_mask (torch.Tensor): Shape (batch, sequence_len)

        Returns:
            torch.Tensor: Shape (batch, embed_dim)
        """
        
        # Remove second special token from start of input embeds, since it is not used in text encoder
        input_word_embeds = torch.cat([
            input_word_embeds[:, :1],
            input_word_embeds[:, 2:]
        ], dim=1)
        attn_mask = torch.cat([
            attn_mask[:, :1],
            attn_mask[:, 2:]
        ], dim=1)
        
        # Pass through text encoder
        text_outputs = self.model.mmpt_model.model.text_encoder(
            inputs_embeds=input_word_embeds,
            attention_mask=attn_mask,
            token_type_ids=torch.zeros_like(attn_mask, dtype=torch.long, device=DEVICE if self.cuda else "cpu"),
            output_hidden_states=True
        )[0]
        
        # Text output embedding is average over final hidden states for all text tokens and final [SEP]
        text_output_attn_mask = attn_mask.clone()
        text_output_attn_mask[:, 0] = False # Do not average over hidden state for first [CLS] token
        text_output_attn_mask = text_output_attn_mask.type(text_outputs.dtype) / text_output_attn_mask.sum(1, keepdim=True)
        
        pooled_text = torch.bmm(
            text_outputs.transpose(2, 1),
            text_output_attn_mask.unsqueeze(2)
        ).squeeze(-1)
        return pooled_text
        
    def text_encoder_over_embeds(self, text):
        with torch.no_grad():
            input_word_embeds, attn_mask = self.get_input_word_embeddings([text])
            return self.text_encoder_from_word_embeddings(input_word_embeds, attn_mask)[0].cpu().numpy()

    def open_video(self, video_path: str, subvideo_start_frame: Optional[int] = None, subvideo_end_frame: Optional[int] = None, random_augment: bool = False) -> np.ndarray:
        """
        Opens video and returns basic, non-transformed video tensor
        Video model requires blocks of 1 second, 30 frame videos
        :param video:
        :param subvideo_start_frame:
        :param subvideo_end_frame:
        :return:
        """
        video_reader = decord.VideoReader(video_path, num_threads=1)
        video_len = len(video_reader)
        video_fps = video_reader.get_avg_fps()
        return video_reader.get_batch(self.sample_frame_indices(video_len, video_fps, subvideo_start_frame, subvideo_end_frame, random_augment))

    def transform(self, video, random_augment: bool = False):
        """
        Transforms video using model-specific transforms
        :param video:
        :return:
        """
        if random_augment:
            inputs = self.train_transforms(video)
        else:
            inputs = self.transforms(video)
        # B, T, FPS, H, W, C (VideoCLIP is trained on 30 fps of s3d)
        _, h, w, c = inputs.size()
        inputs = inputs.view(1, -1, 30, h, w, c)  # Add singleton batch dimension
        return inputs

    def video_encoder(self, video_path: str, subvideo_start_frame: Optional[int] = None, subvideo_end_frame: Optional[int] = None, random_augment: bool = False) -> np.ndarray:
        """
        Load, transform and encode a video file into a joint text/video embedding space
        :param video:
        :param subvideo_start_frame:
        :param subvideo_end_frame:
        :return:
        """
        # Correct for any subvideo start/end frame information included in video_path ("{path}:{start}:{end}")
        video_path_split = video_path.split(":")
        if len(video_path_split) == 3:
            video_path = video_path_split[0]
            subvideo_start_frame = int(video_path_split[1])
            subvideo_end_frame = int(video_path_split[2])
            
        video = self.open_video(video_path, subvideo_start_frame, subvideo_end_frame, random_augment)
        video = self.transform(video, random_augment)
        
        if self.cuda:
            video = video.to(DEVICE)

        with torch.no_grad():
            video_features = self.model.forward(video)
            video_features = video_features.cpu().numpy()[0]
        return video_features

    def default_similarity_metric(self) -> Similarity:
        """
        Returns a reference to the default similarity metric used by this VLM
        :return:
        """
        return Similarity.DOT

    def get_transforms(self):
        # Input is T, H, W, C
        transforms = Compose([
            # Change to C, T, H, W for UniformTemporalSubsampling
            Permute((3, 0, 1, 2)),
            #UniformTemporalSubsample(30*self.num_seconds, ),
            Lambda(lambda x: x/255.0), # Only normalization for VideoCLIP is / 255.0
            ShortSideScale(size=256),
            CenterCrop(224),
            # C, T, H, W -->, T, H, W, C
            Permute((1, 2, 3, 0)),
        ])
        return transforms
    
    def get_train_transforms(self):
        # Input is T, H, W, C
        # Change to (T, C, H, W) for RandAugment 
        transforms = Compose([
            Permute((0, 3, 1, 2)),
            RandAugment(magnitude=7, num_layers=4),
            Lambda(lambda x: x/255.0),
            RandomResizedCrop(target_height=224, target_width=224, scale=(0.08, 1.0), aspect_ratio=(0.75, 1.3333)),
            RandomHorizontalFlip(p=0.5),
            # Change back to T, H, W, C
            Permute(dims=(0, 2, 3, 1)),

        ])
        
        return transforms
    
    def sample_frame_indices(self, video_len: int, video_fps: float, subvideo_start_frame: Optional[int] = None, subvideo_end_frame: Optional[int] = None, random_augment: bool = False) -> np.ndarray:
        subvideo_start_frame = subvideo_start_frame or 0
        subvideo_end_frame = subvideo_end_frame or video_len
        
        native_fps = video_fps
        total_framecount_native = (subvideo_end_frame - subvideo_start_frame)
        
        # Determine the length of the video window to focus on (in seconds/blocks)
        # NOTE: Videos with duration < 1sec will be stretched as though they cover 1sec
        focused_seconds = np.clip(int(total_framecount_native / native_fps), 1, self.num_seconds)
        
        # Extract self.num_seconds 1sec/30frame video blocks from the center of the total video duration
        # TODO: Add support for random_augment
        if self.sample_strat == "center":
            # Calculate size of focus window in number of frames for both native fps and desired 30 fps
            focused_framecount_native = math.ceil(native_fps * focused_seconds)
            focused_framecount_desired = 30 * focused_seconds # Ensure input has multiple of 30 frames
            
            # Calculate start/end frame indices to sample in native fps
            focus_start_native = max(subvideo_start_frame + total_framecount_native // 2 - focused_framecount_native // 2, subvideo_start_frame)
            focus_end_native = min(focus_start_native + focused_framecount_native, subvideo_end_frame)
            
            # Convert native frame indices to desired framerate
            focus_frame_indices_desired = np.minimum(np.round(np.linspace(focus_start_native, focus_end_native, focused_framecount_desired, endpoint=False)), subvideo_end_frame - 1)
            
            return focus_frame_indices_desired
        
        # Extract self.num_seconds 1sec/30frame video blocks from the start of the total video duration
        # TODO: Add support for random_augment
        if self.sample_strat == "start":
            # Calculate size of focus window in number of frames for both native fps and desired 30 fps
            focused_framecount_native = math.ceil(native_fps * focused_seconds)
            focused_framecount_desired = 30 * focused_seconds # Ensure input has multiple of 30 frames
            
            # Calculate start/end frame indices to sample in native fps
            focus_start_native = subvideo_start_frame
            focus_end_native = min(subvideo_start_frame + focused_framecount_native, subvideo_end_frame)
            
            # Convert native frame indices to desired framerate
            focus_frame_indices_desired = np.minimum(np.round(np.linspace(focus_start_native, focus_end_native, focused_framecount_desired, endpoint=False)), subvideo_end_frame - 1)
            
            return focus_frame_indices_desired
        
        # Collect self.num_seconds 1s/30frame blocks evenly spread throughout the video duration
        if self.sample_strat == "spread":
            block_frame_starts_native = np.round(np.linspace(subvideo_start_frame, subvideo_end_frame, focused_seconds, endpoint=False))
            if random_augment:
                block_frame_starts_native += np.random.choice(int((subvideo_end_frame - subvideo_start_frame) / (2 * focused_seconds)), size=focused_seconds) # Randomly start blocks up to half way towards the start of the subsequent block
            focus_frame_indices = []
            for block_frame_start_ind in block_frame_starts_native:
                block_frame_end_ind = min(block_frame_start_ind + native_fps, subvideo_end_frame)
                block_frame_indices = np.minimum(
                    np.round(np.linspace(block_frame_start_ind, block_frame_end_ind, 30, endpoint=False)),
                    block_frame_end_ind - 1
                )
                focus_frame_indices += block_frame_indices.tolist()
                
            return np.array(focus_frame_indices)
        
        raise ValueError(f"Unrecognized sample strat: {self.sample_strat}")