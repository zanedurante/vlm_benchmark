import os
import sys
from types import SimpleNamespace
import pandas as pd
import numpy as np
import decord
from typing import Optional
from torchvision.transforms import Compose, CenterCrop, Resize, Lambda

import torch
import torch.nn.functional as F
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from SimilarityVLM import SimilarityVLM
from similarity_metrics import Similarity

FILE_DIR = os.path.dirname(os.path.realpath(__file__))



# S3D Feature Extractor Setup
FEATURE_EXTRACTOR_REPO_PATH = os.path.join(FILE_DIR, "VideoFeatureExtractor")
sys.path.append(FEATURE_EXTRACTOR_REPO_PATH)
from extract import FRAMERATE_DICT, SIZE_DICT, CENTERCROP_DICT, FEATURE_LENGTH
from preprocessing import Preprocessing
from model import init_weight
from videocnn.models import s3dg

FEATURE_EXTRACTOR_TYPE = "s3dg"
FEATURE_EXTRACTOR_BATCH_SIZE = 64
FEATURE_EXTRACTOR_PRETRAINED_PATH = os.path.join(FEATURE_EXTRACTOR_REPO_PATH, "model/s3d_howto100m.pth")



# UniVL Setup
UNIVL_REPO_PATH = os.path.join(FILE_DIR, "UniVL")
sys.path.append(UNIVL_REPO_PATH)
from modules.modeling import UniVL
from modules.tokenization import BertTokenizer

UNIVL_PRETRAINED_PATH = os.path.join(UNIVL_REPO_PATH, "weight/univl.pretrained.bin")
UNIVL_TASK_CONFIG = SimpleNamespace(
    # Sets model parameters - Do not change
    video_dim=1024,
    visual_num_hidden_layers=6,
    
    # Only affects dataloading/mem-usage (pad to this length, truncate beyond it)
    max_words=48,
    max_frames=64,
    
    # Required for vestigial loss function init (value will not affect model behavior)
    batch_size=256,
    margin=0.1,
    hard_negative_rate=0.5,
    negative_weighting=1,
    n_pair=1,
    n_gpu=1,
    use_mil=True # This also determined whether embeddings were normalized before similarity measurement (Generally it was True for MILNCE-trained unimodal encoder training)
)



# Cache file location
CACHE_NAME = "cache"

class UniVL_SimilarityVLM(SimilarityVLM):
    def __init__(self, reset_cache: bool = False) -> None:
        
        # S3D Video Feature Extractor
        self.pre_preprocessor_transform = Compose([
            Lambda(lambda x: torch.permute(x, (0, 3, 1, 2))),
            Resize(SIZE_DICT[FEATURE_EXTRACTOR_TYPE]),
            CenterCrop(SIZE_DICT[FEATURE_EXTRACTOR_TYPE])
        ])
        self.video_preprocessor = Preprocessing(FEATURE_EXTRACTOR_TYPE, FRAMERATE_DICT)
        self.s3d_model = s3dg.S3D(last_fc=False)
        init_weight(self.s3d_model, torch.load(FEATURE_EXTRACTOR_PRETRAINED_PATH))
        self.s3d_model.to(DEVICE)
        self.s3d_model.eval()
        
        # UniVL Tokenizer and Model
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
        
        model_state_dict = torch.load(UNIVL_PRETRAINED_PATH, map_location="cpu")
        self.model = UniVL.from_pretrained("bert-base-uncased", "visual-base", "cross-base", "decoder-base",
                                           state_dict=model_state_dict, cache_dir=None, task_config=UNIVL_TASK_CONFIG)
        self.model.to(DEVICE)
        self.model.eval()
        
        super().__init__(cache_file=os.path.join(FILE_DIR, CACHE_NAME), reset_cache=reset_cache)
    
    def text_encoder(self, text):
        """
        Tokenize and encode text into a joint text/video embedding space
        :param tokens:
        :return:
        """
        # Tokenize
        tokens = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        
        # Encode
        if len(tokens) > UNIVL_TASK_CONFIG.max_words - 2:
            raise ValueError(f"Token count ({len(tokens)}) is larger than model can input")
        
        cls_token_id, sep_token_id = self.tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]"])
        input_ids = [cls_token_id] + tokens + [sep_token_id]
        
        token_len = len(input_ids)
        pad_len = UNIVL_TASK_CONFIG.max_words - token_len
        input_ids = torch.tensor(input_ids + [0] * pad_len, device=DEVICE)
        input_mask = torch.tensor([1] * token_len + [0] * pad_len, device=DEVICE)
        segment_ids = torch.tensor([0] * UNIVL_TASK_CONFIG.max_words, device=DEVICE)
        
        with torch.no_grad():
            encoded_layers, _ = self.model.bert(input_ids.unsqueeze(0), segment_ids.unsqueeze(0), input_mask.unsqueeze(0), output_all_encoded_layers=True)
            encoded_tokens_batch = encoded_layers[-1]
            encoded_tokens = encoded_tokens_batch[0]
            mean_pooled_text_embed = torch.sum(encoded_tokens * input_mask.unsqueeze(1), dim=0) / torch.sum(input_mask, dim=0)
            if not UNIVL_TASK_CONFIG.use_mil:
                mean_pooled_text_embed = F.normalize(mean_pooled_text_embed, dim=-1)
            
        return mean_pooled_text_embed.cpu().numpy()
    
    def video_encoder(self, video_path: str, subvideo_start_frame: Optional[int] = None, subvideo_end_frame: Optional[int] = None) -> np.ndarray:
        """
        Load, transform and encode a video file into a joint text/video embedding space
        :param video:
        :param subvideo_start_frame:
        :param subvideo_end_frame:
        :return:
        """
        
        # Load
        video_reader = decord.VideoReader(video_path)
        video_len = len(video_reader)
        video_fps = video_reader.get_avg_fps()
        raw_frames = video_reader.get_batch(self.sample_frame_indices(video_len, video_fps, subvideo_start_frame, subvideo_end_frame))

        # Rescale and center-crop before video preprocessor and feature extractor
        frames = torch.from_numpy(raw_frames.asnumpy())
        frames = self.pre_preprocessor_transform(frames)

        # Preprocess
        frames = self.video_preprocessor(frames)
        
        # S3D Feature Extraction
        features = torch.zeros(len(frames), FEATURE_LENGTH[FEATURE_EXTRACTOR_TYPE], device=DEVICE)
        for i in range(0, len(frames), FEATURE_EXTRACTOR_BATCH_SIZE):
            video_batch = frames[i : i + FEATURE_EXTRACTOR_BATCH_SIZE].to(DEVICE)
            feature_batch = self.s3d_model(video_batch)
            feature_batch = F.normalize(feature_batch, dim=1) # If args.l2_normalize (default is 1)
            features[i : i + FEATURE_EXTRACTOR_BATCH_SIZE] = feature_batch
        
        # UniVL visual branch
        if len(features) > UNIVL_TASK_CONFIG.max_frames:
            raise ValueError(f"Feature frame count ({len(features)}) is larger than model can input")
        
        frame_len = len(features)
        pad_len = UNIVL_TASK_CONFIG.max_frames - frame_len
        input_vid_features = torch.concat([features, torch.zeros(pad_len, features.shape[1], device=DEVICE)], dim=0)
        input_mask = torch.tensor([1] * frame_len + [0] * pad_len, device=DEVICE)
        
        with torch.no_grad():
            encoded_layers, _ = self.model.visual(input_vid_features.unsqueeze(0), input_mask.unsqueeze(0), output_all_encoded_layers=True)
            encoded_frames_batch = encoded_layers[-1]
            encoded_frames = encoded_frames_batch[0]
            mean_pooled_vid_embed = torch.sum(encoded_frames * input_mask.unsqueeze(1), dim=0) / torch.sum(input_mask, dim=0)
            if not UNIVL_TASK_CONFIG.use_mil:
                mean_pooled_vid_embed = F.normalize(mean_pooled_vid_embed, dim=-1)
        
        return mean_pooled_vid_embed.cpu().numpy()
    
    def default_similarity_metric(self) -> Similarity:
        """
        Returns a reference to the default similarity metric used by this VLM
        :return:
        """
        return Similarity.DOT
    
    def sample_frame_indices(self, video_len: int, video_fps: float, subvideo_start_frame: Optional[int] = None, subvideo_end_frame: Optional[int] = None) -> np.ndarray:
        # Sample up to UNIVL_TASK_CONFIG.max_frames=64 frames at FRAMERATE_DICT["s3dg"]=16 fps to pass into the feature extractor model
        
        start_frame = subvideo_start_frame or 0
        end_frame = subvideo_end_frame or video_len
        
        duration = (end_frame - start_frame) / video_fps
        if duration <= UNIVL_TASK_CONFIG.max_frames / FRAMERATE_DICT[FEATURE_EXTRACTOR_TYPE]:
            frame_indices = np.linspace(start_frame, end_frame, int(duration * FRAMERATE_DICT[FEATURE_EXTRACTOR_TYPE]), endpoint=False)
            frame_indices = np.minimum(
                np.round(frame_indices),
                end_frame - 1
            )
        else:
            unused_duration = duration - UNIVL_TASK_CONFIG.max_frames / FRAMERATE_DICT[FEATURE_EXTRACTOR_TYPE]
            frame_indices = np.linspace(start_frame + int(unused_duration * video_fps / 2), end_frame - int(unused_duration * video_fps / 2), UNIVL_TASK_CONFIG.max_frames, endpoint=False)
            frame_indices = np.minimum(
                np.round(frame_indices),
                end_frame - 1
            )
            
        return frame_indices