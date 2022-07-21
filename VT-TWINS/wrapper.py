import numpy as np
import os, sys

import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from SimilarityVLM import *

sys.path.append(os.path.join(os.path.dirname(__file__), "VT-TWINS"))
from s3dg import S3D
from loader.msrvtt_loader import MSRVTT_DataLoader

REPO_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "VT-TWINS")
WORD2VEC_PATH = "data/word2vec.pth"
TOKEN_DICT_PATH = "data/dict.npy"
PRETRAINED_PATH = "checkpoints/pretrained.pth.tar"

DEFAULT_EMBED_DIM = 512

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
    def __init__(self):
        # Create model
        self.model = S3D(word2vec_path=WORD2VEC_PATH, token_to_word_path=TOKEN_DICT_PATH)
        self.model.to(DEVICE)
        self.model.eval()
        
        # Create spoof object to run dataloader video loading methods against
        self.spoof_dataloader = SpoofDataLoader()
        
        pretrained_path = os.path.join(REPO_PATH, PRETRAINED_PATH)
        super().__init__(pretrained_path)
        
    def load_model(self, path):
        checkpoint = torch.load(path, map_location="cpu")
        checkpoint_module = {k[7:]: v for k, v in checkpoint.items()}
        self.model.load_state_dict(checkpoint_module)
        
    def tokenize(self, text):
        return self.spoof_dataloader.words_to_ids(text)
    
    def text_encoder(self, tokens):
        with torch.no_grad():
            tokens = tokens.unsqueeze(0).to(DEVICE)
            return self.model.text_module(tokens).cpu().numpy()[0]
    
    # Loads a single video into multiple clips which will be individually encoded and averaged
    def open_video(self, path):
        duration = self.spoof_dataloader._get_duration(path)
        video = self.spoof_dataloader._get_video(path, 0, float(duration), self.spoof_dataloader.num_clip)
        return video
    
    def transform(self, video):
        return video
    
    def video_encoder(self, video):
        with torch.no_grad():
            video = video.float().to(DEVICE)
            return self.model.forward_video(video).mean(dim=0).cpu().numpy()
    
    def default_similarity_metric(self) -> Similarity:
        return Similarity.DOT
    
    
    
    
    
    
    
if __name__ == "__main__":
    test = VTTWINS_SimilarityVLM()
    
    TEXT_LIST = [
        "eating pasta",
        "murder",
        "setting a table",
        "pouring a drink"
    ]
    VID_PATH = "D:/datasets/PAC/few_shot_act_reg/smsm_cmn/001.Pouring_something_into_something/3411.webm"
    
    test.open_video(VID_PATH)
    vid_embed = test.video_encoder(test.open_video(VID_PATH))
    
    for text in TEXT_LIST:
        test.tokenize(text)
        text_embed = test.text_encoder(test.tokenize(text))
        print(f"{text}: {test.get_similarity(text_embed, vid_embed)}")