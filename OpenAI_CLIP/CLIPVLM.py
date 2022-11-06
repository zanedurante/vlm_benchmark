import numpy as np
import random
import os
from typing import Optional, List
import torch # 1  
import decord # 2 DO NOT CHANGE ORDER--will cause error due to weird decord bug

DEVICE = "cuda" if torch.cuda.is_available() else "cpu" #2

from SimilarityVLM import SimilarityVLM
from similarity_metrics import Similarity

from . import clip_utils #1

from torchvision.transforms import (
    Compose, Normalize, CenterCrop, ToTensor,
    RandomResizedCrop, RandomHorizontalFlip
) #1





import tracemalloc #1

tracemalloc.start() #1

# Default cache locations
FILE_DIR = os.path.dirname(os.path.realpath(__file__))
CACHE_NAME = "cache"


class ClipVLM(SimilarityVLM):
    """
    Similarity-based VLM that uses CLIP for frame and text encoders.  Currently, we use the hugging face implementation
    for CLIP since it is easier to set up.
    TODO: Implement the larger version of CLIP since this should get better performance.
    """

    def __init__(self, model_name="ViT-B/32", num_frames=1, sample_strat='uniform',
                 reset_cache=False):
        self.model = None
        self.tokenizer = None
        self.processor = None
        
        self.model_name = model_name
        self.num_frames = num_frames  # The number of frames CLIP will use to classify videos
        self.sample_strat = sample_strat  # 'rand' or 'uniform'

        decord.bridge.set_bridge("torch")  # Video loader
        
        self.train_transforms = self.get_train_transforms()
        self.test_transforms = self.get_test_transforms()
        
        self.model = clip_utils.load_clip_to_cpu(model_name).float()
        self.model.to(DEVICE)
        
        
        super().__init__(cache_file=os.path.join(FILE_DIR, CACHE_NAME), reset_cache=reset_cache)
        
    def params(self) -> dict:
        """
        Specify the value of all VLM-specific parameters which may affect prediction accuracy.
        This is used to differentiate test results which use different versions of the same VLM.
        :return:
        :rtype: dict
        """
        return {
            "model_name": self.model_name,
            "num_frames": self.num_frames,
            "sample_strat": self.sample_strat
        }
        
    def logit_scale(self) -> float:
        return 100
        
    def input_word_embed_dim(self) -> int:
        return 512
    
    def text_start_special_token_count(self) -> int:
        return 1
    
    def text_end_special_token_count(self) -> int:
        return 1
    
    def text_encoder(self, text):
        """
        Tokenize and encode text into a joint text/video embedding space
        :param tokens:
        :return:
        """
        with torch.no_grad():
            input_word_embeds, attn_mask = self.get_input_word_embeddings([text])
            return self.text_encoder_from_word_embeddings(input_word_embeds, attn_mask)[0].cpu().numpy()
        
    def get_input_word_embeddings(self, text_list: List[str]) -> torch.Tensor:
        """Converts a list of text string into a batched tensor of input word embeddings and a corresponding attention mask,
        including special tokens.

        Args:
            text_list (str): _description_

        Returns:
            torch.Tensor: input token embeddings for the text encoder. Shape (batch, sequence_len, token_dim)
            torch.Tensor: input sequence attention mask for the text encoder. Shape (batch, sequence_len)
        """
        tokenized_text = clip_utils.tokenize(text_list).to(DEVICE)
        token_embeds = self.model.token_embedding(tokenized_text)
        
        attn_mask = torch.any(token_embeds != 0, dim=-1)
        
        return token_embeds, attn_mask
        
    def text_encoder_from_word_embeddings(self, input_word_embeds: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        """Converts a batch of token embeddings and corresponding attention masks into a batch of text embeddings.

        Args:
            token_embeds (torch.Tensor): Shape (batch, sequence_len, token_dim)
            attn_mask (torch.Tensor): Shape (batch, sequence_len)

        Returns:
            torch.Tensor: Shape (batch, embed_dim)
        """
        x = input_word_embeds + self.model.positional_embedding.type(self.model.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.ln_final(x).type(self.model.dtype)
        
        # Instead of finding eot token position as max token id,
        # identify it as the last nonzero element of attention mask in each row
        bsz, seq_len = attn_mask.shape
        attn_mask_positions = attn_mask * torch.arange(seq_len, device=DEVICE).unsqueeze(0).expand(bsz, seq_len)
        final_nonzero_indices = attn_mask_positions.argmax(dim=-1)
        
        # Take features from the eot embedding (final token)
        x = x[torch.arange(x.shape[0], device=DEVICE), final_nonzero_indices]
        return x
        
    def text_encoder_over_embeds(self, text):
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
        # Load
        # TODO: Figure out best way to subsample the video with CLIP (in the paper they just use one single frame)
        video_reader = decord.VideoReader(video_path, num_threads=1)
        video_len = len(video_reader)
        frame_indices = self.sample_frame_indices(video_len, subvideo_start_frame, subvideo_end_frame, random_augment)
        frames = video_reader.get_batch(frame_indices)
        frames = frames.float() / 255
        frames = frames.permute(0, 3, 1, 2)
        # Convert frame batch axis into list
        frames = [frame for frame in frames]
        
        frames = torch.stack(frames)
        # Preprocess
        inputs = self.transform(images=frames, random_augment=random_augment).to(DEVICE)

        # Encode
        with torch.no_grad():
            video_features = self.model.visual(inputs)  # Frame-level video features
            video_features = video_features.mean(dim=0) # Average over sampled frames from this video
            video_features = video_features.cpu().numpy()

        return video_features
    

    def transform(self, images: list, random_augment: bool = False):
        """
        Test and train-time normalization a list of frames to be input into CLIP. Set self.use_train_transforms to toggle between train and val transforms.
        :param images: A list of input frames to transform individually
        :return: A tensor of input frames that are normalized correctly
        """
        if random_augment:
            return self.train_transforms(images)
        else:
            return self.test_transforms(images)
        
    
    def get_train_transforms(self):
        return Compose([
            RandomResizedCrop((224, 224), scale=(0.8, 1.0), interpolation="bicubic"), # Changed from scale=(0.08, 1.0) since this is too small for fine-grained detection
            RandomHorizontalFlip(p=0.5),
            Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
        ])
    
    def get_test_transforms(self):
        return Compose([
            CenterCrop((224, 224)),
            RandomHorizontalFlip(p=0.5),
            Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
        ])
        
        
    def default_similarity_metric(self) -> Similarity:
        """
        Returns a reference to the default similarity metric used by this VLM
        :return:
        """
        return Similarity.COSINE

    def sample_frame_indices(self, video_len: int, subvideo_start_frame: Optional[int] = None, subvideo_end_frame: Optional[int] = None) -> np.ndarray:
        start_frame = subvideo_start_frame or 0
        end_frame = subvideo_end_frame or video_len
        
        frame_indices = np.linspace(start_frame, end_frame, self.num_frames, endpoint=False)
        
        if self.sample_strat == "rand":
            frame_indices += np.random.choice((end_frame - start_frame) // self.num_frames)
        elif self.sample_strat == "uniform":
            frame_indices += (end_frame - start_frame) / self.num_frames / 2
        else:
            raise ValueError
        
        frame_indices = np.minimum(
            np.round(frame_indices),
            end_frame - 1
        )
        
        return frame_indices