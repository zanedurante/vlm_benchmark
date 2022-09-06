import numpy as np
import decord
import random
import os

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
        tokens = self.tokenizer(text, padding=True, return_tensors="pt")
        with torch.no_grad():
            text_features = self.model.get_text_features(**tokens).cpu().numpy()[0]
        return text_features

    def video_encoder(self, video_path):
        """
        Load, transform and encode a video file into a joint text/video embedding space
        :param video:
        :return:
        """
        # Load
        # TODO: Figure out best way to subsample the video with CLIP (in the paper they just use one single frame)
        video_reader = decord.VideoReader(video_path, num_threads=1)
        vlen = len(video_reader)
        frame_idxs = self.get_frame_idxs(vlen)
        frames = video_reader.get_batch(frame_idxs)
        frames = frames.float() / 255
        frames = frames.permute(0, 3, 1, 2)
        # Convert frame batch axis into list
        frames = [frame for frame in frames]
        
        # Preprocess
        inputs = self.processor(images=frames, return_tensors="pt")
        
        # Encode
        with torch.no_grad():
            video_features = self.model.get_image_features(**inputs)  # Frame-level video features
            video_features = video_features.cpu().numpy()[0]

        return video_features

    def default_similarity_metric(self) -> Similarity:
        """
        Returns a reference to the default similarity metric used by this VLM
        :return:
        """
        return Similarity.COSINE

    def get_frame_idxs(self, vlen):
        # Determine number of samples
        num_samples = min(self.num_frames, vlen)

        # Determine intervals of indices to sample from (e.g. [0, 3, 6, 10] for num_frames=3, vlen=10)
        intervals = np.linspace(start=0, stop=vlen, num=num_samples + 1).astype(int)
        ranges = []
        for idx, interval in enumerate(intervals[:-1]):
            ranges.append((interval, intervals[idx + 1] - 1))
        if self.sample_strat == 'rand':
            frame_idxs = [random.choice(range(x[0], x[1])) for x in ranges]
        elif self.sample_strat == 'uniform':
            frame_idxs = [(x[0] + x[1]) // 2 for x in ranges]
        else:
            raise NotImplementedError

        return frame_idxs

