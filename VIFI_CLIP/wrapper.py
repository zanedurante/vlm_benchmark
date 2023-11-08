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

# Pretrained State
FILE_DIR = os.path.dirname(os.path.realpath(__file__))
PRETRAINED_CHECKPOINT_PATH = os.path.join(FILE_DIR, "pretrained/vifi_clip_10_epochs_k400_full_finetuned.pth")

# Cache file location
CACHE_NAME = "cache"



class ViFiCLIP_SimilarityVLM(SimilarityVLM):
    def __init__(self,
                 num_frames: int = 4,
                 load_vifi: bool = True,
                 reset_cache: bool = False):
        
        # Hyperparameters
        self.num_frames = num_frames
        self.load_vifi = load_vifi
        
        # Build CLIP model from online checkpoint (ViT-16)
        original_model_path = clip._download(clip._MODELS["ViT-B/16"])
        try:
            # loading JIT archive
            state_dict = torch.jit.load(original_model_path, map_location="cpu").eval().state_dict()
        except RuntimeError:
            state_dict = torch.load(original_model_path, map_location="cpu")
        self.model = clip.build_model(state_dict)
        
        
        # Update weights based on ViFiCLIP checkpoint (requires renaming since ViFi trained with a custom module)
        if load_vifi:
            state_dict = torch.load(PRETRAINED_CHECKPOINT_PATH, map_location="cpu")
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

        # Finalize clip model
        self.model.eval()
        self.model.to(DEVICE)
        self.model.requires_grad_(False)
        
        # Initialize preprocess transform
        self.image_preprocessor = transforms.Compose([
            transforms.Resize(self.model.visual.input_resolution, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(self.model.visual.input_resolution),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
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
        decord.bridge.set_bridge("torch")
        video_reader = decord.VideoReader(video_path, num_threads=1)
        video_len = len(video_reader)
        frame_indices = self.sample_frame_indices(video_len, subvideo_start_frame, subvideo_end_frame, random_augment)
        frames = video_reader.get_batch(frame_indices)
        frames = frames.float() / 255
        frames = frames.permute(0, 3, 1, 2)
        
        # Preprocess
        frames = torch.stack([
            self.image_preprocessor(frame)
            for frame in frames
        ])
        
        decord.bridge.set_bridge("native")
        return frames
    
    def video_encoder(self, video_path: str, subvideo_start_frame: Optional[int] = None, subvideo_end_frame: Optional[int] = None, random_augment: bool = False) -> np.ndarray:
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