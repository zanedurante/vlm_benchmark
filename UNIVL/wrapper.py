import os
import sys
from types import SimpleNamespace
import pandas as pd
import numpy as np

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
from video_loader import VideoLoader
from preprocessing import Preprocessing
from model import init_weight
from videocnn.models import s3dg

FEATURE_EXTRACTOR_TYPE = "s3dg"
FEATURE_EXTRACTOR_BATCH_SIZE = 64
FEATURE_EXTRACTOR_PRETRAINED_PATH = os.path.join(FEATURE_EXTRACTOR_REPO_PATH, "model/s3d_howto100m.pth")

'''
Dummy version of VideoFeatureExtractor VideoLoader, which loads videos directly rather than
from a specifically formatted csv file
'''
class SpoofVideoLoader(VideoLoader):
    def __init__(self) -> None:
        self.csv = pd.DataFrame([[None, None]], columns=["video_path", "feature_path"])
        self.centercrop = CENTERCROP_DICT[FEATURE_EXTRACTOR_TYPE]
        self.size = SIZE_DICT[FEATURE_EXTRACTOR_TYPE]
        self.framerate = FRAMERATE_DICT[FEATURE_EXTRACTOR_TYPE]

    def load_video(self, video_path: str) -> torch.Tensor:
        self.csv = pd.DataFrame([[video_path, ""]], columns=["video_path", "feature_path"])
        loader_object = self[0]
        return self[0]["video"]



# UniVL Setup
UNIVL_REPO_PATH = os.path.join(FILE_DIR, "UniVL")
sys.path.append(UNIVL_REPO_PATH)
from modules.modeling import UniVL
from modules.tokenization import BertTokenizer

UNIVL_PRETRAINED_PATH = os.path.join(UNIVL_REPO_PATH, "weight/univl.pretrained.bin")
UNIVL_TASK_CONFIG = SimpleNamespace(
    num_thread_reader=1,
    lr=0.0001,
    epochs=20,
    batch_size=256,
    batch_size_val=3500,
    lr_decay=0.9,
    n_display=100,
    video_dim=1024,
    seed=42,
    max_words=20,
    max_frames=100,
    min_words=0,
    feature_framerate=1,
    min_time=5.0,
    margin=0.1,
    hard_negative_rate=0.5,
    negative_weighting=1,
    n_pair=1,
    
    #do_lower_case=True,
    n_gpu=1,
    
    use_mil=False
)



# Cache file location
CACHE_INDEX_NAME = "cache_index.pickle"
CACHE_DIR_NAME = "cache_dir"

class UniVL_SimilarityVLM(SimilarityVLM):
    def __init__(self, reset_cache: bool = False) -> None:
        
        # S3D Video Feature Extractor
        self.video_loader = SpoofVideoLoader()
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
        
        # Cache file locations
        cache_index_path = os.path.join(FILE_DIR, CACHE_INDEX_NAME)
        cache_dir_path = os.path.join(FILE_DIR, CACHE_DIR_NAME)
        
        super().__init__(UNIVL_PRETRAINED_PATH, cache_file=cache_index_path, cache_dir=cache_dir_path, reset_cache=reset_cache)
        
    def load_model(self, path):
        pass
    
    def tokenize(self, text):
        return self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
    
    def text_encoder(self, tokens):
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
    
    def open_video(self, path):
        return self.video_preprocessor(self.video_loader.load_video(path))
        
    def transform(self, video):
        return video
    
    def video_encoder(self, video):
        # S3D Feature Extraction
        features = torch.zeros(len(video), FEATURE_LENGTH[FEATURE_EXTRACTOR_TYPE], device=DEVICE)
        for i in range(0, len(video), FEATURE_EXTRACTOR_BATCH_SIZE):
            video_batch = video[i : i + FEATURE_EXTRACTOR_BATCH_SIZE].to(DEVICE)
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
        return Similarity.DOT