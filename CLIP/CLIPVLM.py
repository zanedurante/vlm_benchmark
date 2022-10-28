import numpy as np
import decord
import random
import os
from typing import Optional

import torch
from transformers import CLIPModel, CLIPTokenizer, CLIPProcessor
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from SimilarityVLM import SimilarityVLM
from similarity_metrics import Similarity



# Default cache locations
FILE_DIR = os.path.dirname(os.path.realpath(__file__))
CACHE_NAME = "cache"



class ClipVLM(SimilarityVLM):
    """
    Similarity-based VLM that uses CLIP for frame and text encoders.  Currently, we use the hugging face implementation
    for CLIP since it is easier to set up.
    TODO: Implement the larger version of CLIP since this should get better performance.
    """

    def __init__(self, path="openai/clip-vit-base-patch32", num_frames=1, sample_strat='uniform',
                 reset_cache=False):
        self.model = None
        self.tokenizer = None
        self.processor = None
        
        self.path = path  # Pretrained CLIP identifier
        self.num_frames = num_frames  # The number of frames CLIP will use to classify videos
        self.sample_strat = sample_strat  # 'rand' or 'uniform'

        decord.bridge.set_bridge("torch")  # Video loader
        
        # Load model
        self.model = CLIPModel.from_pretrained(path)
        self.tokenizer = CLIPTokenizer.from_pretrained(path)
        self.processor = CLIPProcessor.from_pretrained(path)
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
            "path": self.path,
            "num_frames": self.num_frames,
            "sample_strat": self.sample_strat
        }
    
    def text_encoder(self, text):
        """
        Tokenize and encode text into a joint text/video embedding space
        :param tokens:
        :return:
        """
        tokens = self.tokenizer(text, padding=True, return_tensors="pt", max_length=77, truncation=True)
        with torch.no_grad():
            text_features = self.model.get_text_features(**tokens).cpu().numpy()[0]
        return text_features

    def video_encoder(self, video_path: str, subvideo_start_frame: Optional[int] = None, subvideo_end_frame: Optional[int] = None) -> np.ndarray:
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
        frame_indices = self.sample_frame_indices(video_len, subvideo_start_frame, subvideo_end_frame)
        frames = video_reader.get_batch(frame_indices)
        frames = frames.float() / 255
        frames = frames.permute(0, 3, 1, 2)
        # Convert frame batch axis into list
        frames = [frame for frame in frames]

        # Preprocess
        inputs = self.processor(images=frames, return_tensors="pt")
        
        # Encode
        with torch.no_grad():
            video_features = self.model.get_image_features(**inputs)  # Frame-level video features
            video_features = video_features.mean(dim=0) # Average over sampled frames from this video
            video_features = video_features.cpu().numpy()

        return video_features

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