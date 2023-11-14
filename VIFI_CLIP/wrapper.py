import os, sys
from typing import Optional, List
import numpy as np

import torch
import decord
from torchvision import transforms
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from SimilarityVLM import SimilarityVLM
from similarity_metrics import Similarity
from . import clip
from vifi_utils.pipeline import Compose

# Pretrained State
FILE_DIR = os.path.dirname(os.path.realpath(__file__))
PRETRAINED_CHECKPOINT_DIR = os.path.join(FILE_DIR, "pretrained")
VIFI_CHECKPOINT_PATH = os.path.join(PRETRAINED_CHECKPOINT_DIR, "vifi_clip_10_epochs_k400_full_finetuned.pth")

# Cache file location
CACHE_NAME = "cache"

# Default augmentation settings
LABEL_SMOOTH = 0.1
COLOR_JITTER = 0.8
GRAY_SCALE = 0.2
#MIXUP = 0.8
#CUTMIX = 1.0
#MIXUP_SWITCH_PROB = 0.5
INPUT_SIZE = 224



class ViFiCLIP_SimilarityVLM(SimilarityVLM):
    def __init__(self,
                 num_frames: int = 4,
                 load_vifi: bool = True,
                 reset_cache: bool = False):
        
        # Hyperparameters
        self.num_frames = num_frames
        self.load_vifi = load_vifi
        
        # Build CLIP model from online checkpoint (ViT-16)
        original_model_path = clip._download(clip._MODELS["ViT-B/16"], root=PRETRAINED_CHECKPOINT_DIR)
        try:
            # loading JIT archive
            state_dict = torch.jit.load(original_model_path, map_location="cpu").eval().state_dict()
        except RuntimeError:
            state_dict = torch.load(original_model_path, map_location="cpu")
        self.model = clip.build_model(state_dict)
        
        
        # Update weights based on ViFiCLIP checkpoint (requires renaming since ViFi trained with a custom module)
        if load_vifi:
            state_dict = torch.load(VIFI_CHECKPOINT_PATH, map_location="cpu")
            state_dict = state_dict["model"]
            for key in list(state_dict.keys()):
                if key.startswith("module.image_encoder."):
                    state_dict[key.replace("module.image_encoder.", "visual.")] = state_dict.pop(key)
                elif key.startswith("module.text_encoder."):
                    state_dict[key.replace("module.text_encoder.", "")] = state_dict.pop(key)
                elif key.startswith("module.prompt_learner"):
                    state_dict.pop(key)
                elif key.startswith("module."):
                    state_dict[key.replace("module.", "")] = state_dict.pop(key)
            self.model.load_state_dict(state_dict, strict=False)

        # TEMP: Convert all clip weights to fp32 for torch amp autocasting
        self.model.float()
            
        # Finalize clip model
        self.model.eval()
        self.model.to(DEVICE)
        self.model.requires_grad_(False)
        
        # Initialize preprocess transform
        """
        self.image_preprocessor = transforms.Compose([
            transforms.Resize(self.model.visual.input_resolution, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(self.model.visual.input_resolution),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])
        """
        scale_resize = 256 / 224 * INPUT_SIZE
        self.frame_loader = Compose([
            dict(type='DecordInit'),
            dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=self.num_frames, test_mode=True),
            dict(type='DecordDecode'),
            dict(type='Resize', scale=(-1, scale_resize)),
            dict(type='CenterCrop', crop_size=INPUT_SIZE),
            dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False),
            dict(type='FormatShape', input_format='NCHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs'])
        ])
        self.frame_loader_augmented = Compose([
            dict(type='DecordInit'),
            dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=self.num_frames),
            dict(type='DecordDecode'),
            dict(type='Resize', scale=(-1, scale_resize)),
            dict(
                type='MultiScaleCrop',
                input_size=INPUT_SIZE,
                scales=(1, 0.875, 0.75, 0.66),
                random_crop=False,
                max_wh_scale_gap=1),
            dict(type='Resize', scale=(INPUT_SIZE, INPUT_SIZE), keep_ratio=False),
            dict(type='Flip', flip_ratio=0.5),
            dict(type='ColorJitter', p=COLOR_JITTER),
            dict(type='GrayScale', p=GRAY_SCALE),
            dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False),
            dict(type='FormatShape', input_format='NCHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs', 'label']),
        ])
        
        super().__init__(cache_file=os.path.join(FILE_DIR, CACHE_NAME), reset_cache=reset_cache)
        
    def params(self) -> dict:
        """
        Specify the value of all VLM-specific parameters which may affect prediction accuracy.
        This is used to differentiate test results which use different versions of the same VLM.
        :return:
        :rtype: dict
        """
        return {
            "num_frames": self.num_frames,
            "load_vifi": self.load_vifi
        }
        
    def logit_scale(self) -> float:
        return float(self.model.logit_scale.item())
    
    def output_embed_dim(self) -> int:
        return self.model.text_projection.size(1)
    
    def input_word_embed_dim(self) -> int:
        return self.model.token_embedding.embedding_dim
    
    def text_start_special_token_count(self) -> int:
        return 1
    
    def text_end_special_token_count(self) -> int:
        return 1
    
    def text_encoder(self, text) -> np.ndarray:
        """
        Tokenize and encode text into a joint text/video embedding space
        :param tokens:
        :return:
        """
        tokens = clip.tokenize([text]).to(DEVICE)
        with torch.no_grad():
            text_features = self.model.encode_text(tokens).cpu().float().numpy()[0]
        return text_features
    
    def get_input_word_embeddings(self, text_list: List[str]) -> torch.Tensor:
        """Converts a list of text string into a batched tensor of input word embeddings and a corresponding attention mask,
        including special tokens.

        Args:
            text_list (str): _description_

        Returns:
            torch.Tensor: input token embeddings for the text encoder. Shape (batch, sequence_len, token_dim)
            torch.Tensor: input sequence attention mask for the text encoder. Shape (batch, sequence_len)
        """
        tokens = clip.tokenize(text_list).to(DEVICE)
        
        # Remove extra tokens (clip.tokenize returns the max possible sequence length regardless of input)
        max_length = tokens.argmax(dim=1).max() + 1
        tokens = tokens[:, :max_length]
        
        input_word_embeds = self.model.token_embedding(tokens).type(self.model.dtype)
        attn_mask = torch.arange(tokens.size(1), device=DEVICE)[None, :].expand(tokens.size(0), -1) <= tokens.argmax(dim=1, keepdim=True)
        return input_word_embeds, attn_mask
    
    def text_encoder_from_word_embeddings(self, input_word_embeds: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        """Converts a batch of token embeddings and corresponding attention masks into a batch of text embeddings.

        Args:
            token_embeds (torch.Tensor): Shape (batch, sequence_len, token_dim)
            attn_mask (torch.Tensor): Shape (batch, sequence_len)

        Returns:
            torch.Tensor: Shape (batch, embed_dim)
        """
        bsz, seq_len, input_embed_dim = input_word_embeds.shape
        
        # Expand input word embeddings to max sequence length
        max_seq_len = self.model.positional_embedding.size(0)
        input_word_embeds = torch.cat([
            input_word_embeds,
            torch.zeros(bsz, max_seq_len - seq_len, input_embed_dim, device=DEVICE)
        ], dim=1)
        attn_mask = torch.cat([
            attn_mask,
            torch.zeros(bsz, max_seq_len - seq_len, device=DEVICE)
        ], dim=1)
        
        x = input_word_embeds.type(self.model.dtype)

        x = x + self.model.positional_embedding[None, :, :].type(self.model.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.ln_final(x).type(self.model.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding
        eot_token_inds = (attn_mask * torch.arange(attn_mask.size(1), device=DEVICE)[None, :].expand(bsz, -1)).argmax(dim=-1)
        x = x[torch.arange(x.shape[0]), eot_token_inds] @ self.model.text_projection
        x = x.float()
    
        return x
    
    def text_encoder_over_embeds(self, text):
        with torch.no_grad():
            input_word_embeds, attn_mask = self.get_input_word_embeddings([text])
            return self.text_encoder_from_word_embeddings(input_word_embeds, attn_mask)[0].cpu().numpy()
        
    def load_video_frames(self, video_path: str, subvideo_start_frame: Optional[int] = None, subvideo_end_frame: Optional[int] = None, random_augment: bool = False) -> torch.Tensor:        
        item_info = dict(
            filename = video_path,
            label = 0,
            tar = False,
            start_index = 0,
            modality = "RGB"
        )
        if subvideo_start_frame is not None:
            item_info["start_frame"] = subvideo_start_frame
        if subvideo_end_frame is not None:
            item_info["end_frame"] = subvideo_end_frame
        
        if not random_augment:
            item_info = self.frame_loader(item_info)
        else:
            item_info = self.frame_loader_augmented(item_info)
        frames = item_info["imgs"]
        return frames
        
    
    def video_encoder(self, video_path: str, subvideo_start_frame: Optional[int] = None, subvideo_end_frame: Optional[int] = None, random_augment: bool = False) -> np.ndarray:
        # Correct for any subvideo start/end frame information included in video_path ("{path}:{start}:{end}")
        video_path_split = video_path.split(":")
        if len(video_path_split) == 3:
            video_path = video_path_split[0]
            subvideo_start_frame = int(video_path_split[1])
            subvideo_end_frame = int(video_path_split[2])
        
        frames = self.load_video_frames(video_path, subvideo_start_frame, subvideo_end_frame, random_augment).to(DEVICE)
        
        # Encode
        with torch.no_grad():
            frame_features = self.model.encode_image(frames)
            video_features = frame_features.mean(dim=0).cpu().float().numpy()
        return video_features
    
    
    
    
    
    def default_similarity_metric(self) -> Similarity:
        """
        Returns a reference to the default similarity metric used by this VLM
        :return:
        """
        return Similarity.COSINE
    
    def sample_frame_indices(self, video_len: int, subvideo_start_frame: Optional[int] = None, subvideo_end_frame: Optional[int] = None, random_augment: bool = False) -> np.ndarray:
        start_frame = subvideo_start_frame or 0
        end_frame = subvideo_end_frame or video_len
        
        frame_indices = np.linspace(start_frame, end_frame, self.num_frames, endpoint=False)
        
        if random_augment:
            frame_indices += np.random.choice((end_frame - start_frame) // self.num_frames)
        else:
            frame_indices += (end_frame - start_frame) / self.num_frames / 2

        frame_indices = np.minimum(
            np.round(frame_indices),
            end_frame - 1
        )
        
        return frame_indices