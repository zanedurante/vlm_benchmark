import importlib
import numpy as np
import os, sys

import torch
import decord
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from SimilarityVLM import *

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
REPO_PATH = os.path.join(FILE_DIR, "VT-TWINS")

# Pretrained files for loading various parts of the VLM
WORD2VEC_PATH = "data/word2vec.pth"
TOKEN_DICT_PATH = "data/dict.npy"
PRETRAINED_PATH = "checkpoints/pretrained.pth.tar"

# Import from VT-TWINS repo
sys.path.append(REPO_PATH)
from s3dg import S3D
from loader.msrvtt_loader import MSRVTT_DataLoader

# Default VLM parameters
DEFAULT_EMBED_DIM = 512

# Cache file location
CACHE_NAME = "cache"

class SpoofDataLoader(MSRVTT_DataLoader):
    def __init__(self):
        self.size = 224
        self.num_frames = 32
        self.fps = 10
        self.num_clip = 10
        self.num_sec = self.num_frames / float(self.fps)
        self.crop_only = False
        self.center_crop = True
        self.max_words = 30
        self.word_to_token = {w: t + 1 for t, w in enumerate(np.load(os.path.join(REPO_PATH, TOKEN_DICT_PATH)))}

class VTTWINS_SimilarityVLM(SimilarityVLM):
    def __init__(self, reset_cache: bool = False):
        
        # Create model
        self.model = S3D(word2vec_path=WORD2VEC_PATH, token_to_word_path=TOKEN_DICT_PATH)
        self.model.to(DEVICE)
        self.model.eval()
        
        # Create spoof object to run dataloader video loading methods against
        self.spoof_dataloader = SpoofDataLoader()
        
        # Load model
        pretrained_path = os.path.join(REPO_PATH, PRETRAINED_PATH)
        checkpoint = torch.load(pretrained_path, map_location="cpu")
        checkpoint_module = {k[7:]: v for k, v in checkpoint.items()}
        self.model.load_state_dict(checkpoint_module)
        
        super().__init__(cache_file=os.path.join(FILE_DIR, CACHE_NAME), reset_cache=reset_cache)
    
    def text_encoder(self, text):
        """
        Tokenize and encode text into a joint text/video embedding space
        :param tokens:
        :return:
        """
        # Tokenize
        tokens = self.spoof_dataloader.words_to_ids(text)
        
        # Encode
        with torch.no_grad():
            tokens = tokens.unsqueeze(0).to(DEVICE)
            embedding = self.model.text_module(tokens).cpu().numpy()[0]

        return embedding
    
    def video_encoder(self, video_path: str, subvideo_start_frame: Optional[int] = None, subvideo_end_frame: Optional[int] = None) -> np.ndarray:
        """
        Load, transform and encode a video file into a joint text/video embedding space
        :param video:
        :param subvideo_start_frame:
        :param subvideo_end_frame:
        :return:
        """
        # Convert start/end bounds to seconds
        video_reader = decord.VideoReader(video_path)
        video_len = len(video_reader)
        video_fps = video_reader.get_avg_fps()
        
        subvideo_start_frame = subvideo_start_frame or 0
        subvideo_end_frame = subvideo_end_frame or video_len
        
        subvideo_start_time = subvideo_start_frame / video_fps
        subvideo_end_time = subvideo_end_frame / video_fps
        
        # Load and Preprocess
        # Loads a single video into multiple clips which will be individually encoded and averaged
        video = self.spoof_dataloader._get_video(video_path, subvideo_start_time, subvideo_end_time, self.spoof_dataloader.num_clip)
        video /= 255
        
        # Encode
        with torch.no_grad():
            video = video.float().to(DEVICE)
            embedding = self.model.forward_video(video).mean(dim=0).cpu().numpy()
        
        return embedding
    
    def default_similarity_metric(self) -> Similarity:
        """
        Returns a reference to the default similarity metric used by this VLM
        :return:
        """
        return Similarity.DOT