import numpy as np
import random
import os

import torch
from .MMPT_updated.mmpt.models import MMPTClassifier

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from SimilarityVLM import SimilarityVLM
from similarity_metrics import Similarity

from torchvision.io import read_video
from pytorchvideo.transforms import *
from torchvision.transforms import Compose, Lambda, CenterCrop

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

        self.model = None
        self.cuda = use_cuda and DEVICE == "cuda"
        self.path = path  # Pretrained video clip identifier
        self.num_seconds = num_seconds
        self.sample_strat = sample_strat
        self.transforms = self.get_transforms()
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
            "sample_strat": self.sample_strat
        }

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
        self.model.caps = caps.to(DEVICE)
        self.model.cmasks = cmasks.to(DEVICE)
        self.model.to(DEVICE)

        return

    def tokenize(self, text):
        """
        Tokenizes text via tokenizer (likely variant of huggingface BertTokenizer)
        :param text:, list of text to tokenize
        :return: Tokenized text
        """
        tokenized_text = []
        for t in text:
            caps, cmasks = self.model.aligner._build_text_seq(self.model.tokenizer(t, add_special_tokens=False)["input_ids"])
            caps, cmasks = caps[None, :], cmasks[None, :]
            tokenized_text.append((caps, cmasks))
        return tokenized_text

    def text_encoder(self, text):
        """
        Encodes tokenized text into joint text/video embedding space
        :param text:
        :return:
        """

        text_tokens, text_mask = self.tokenize([text])[0]


        # Note: We have to generate random video frames since VideoCLIP requires video and text input
        # (it uses attention masking to ensure leakage). Can verify by changing video embedding and
        # see there is no difference in the output text embeddings.

        random_video = np.zeros((1, 1, 30, 224, 224, 3))
        video_frames = torch.from_numpy(random_video).float()
        if self.cuda:
            video_frames = video_frames.to('cuda')
            text_tokens = text_tokens.to('cuda')
            text_mask = text_mask.to('cuda')

        with torch.no_grad():
            output = self.model.mmpt_model(video_frames, text_tokens, text_mask, return_score=False)
            text_features = output["pooled_text"].cpu().numpy().squeeze()
        return text_features

    def open_video(self, video_path: str, subvideo_start_frame: Optional[int] = None, subvideo_end_frame: Optional[int] = None) -> np.ndarray:
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
        return video_reader.get_batch(self.sample_frame_indices(video_len, video_fps, subvideo_start_frame, subvideo_end_frame))

    def transform(self, video):
        """
        Transforms video using model-specific transforms
        :param video:
        :return:
        """
        inputs = self.transforms(video)
        # B, T, FPS, H, W, C (VideoCLIP is trained on 30 fps of s3d)
        _, h, w, c = inputs.size()
        inputs = inputs.view(1, -1, 30, h, w, c)  # Add singleton batch dimension
        return inputs

    def video_encoder(self, video_path: str, subvideo_start_frame: Optional[int] = None, subvideo_end_frame: Optional[int] = None) -> np.ndarray:
        """
        Load, transform and encode a video file into a joint text/video embedding space
        :param video:
        :param subvideo_start_frame:
        :param subvideo_end_frame:
        :return:
        """
        video = self.open_video(video_path, subvideo_start_frame, subvideo_end_frame)
        video = self.transform(video)

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
            # Change back to T, H, W, C
            Permute(dims=(0, 2, 3, 1)),
            Lambda(lambda x: x/255.0),
            RandomResizedCrop(target_height=224, target_width=224, scale=(0.08, 1.0), aspect_ratio=(0.75, 1.3333)),
            RandomHorizontalFlip(p=0.5),
        ])
        
        return transforms
    
    def sample_frame_indices(self, video_len: int, video_fps: float, subvideo_start_frame: Optional[int] = None, subvideo_end_frame: Optional[int] = None, use_strat: Optional[str] = None) -> np.ndarray:
        subvideo_start_frame = subvideo_start_frame or 0
        subvideo_end_frame = subvideo_end_frame or video_len
        
        native_fps = video_fps
        total_framecount_native = (subvideo_end_frame - subvideo_start_frame)
        
        # Determine the length of the video window to focus on (in seconds/blocks)
        # NOTE: Videos with duration < 1sec will be stretched as though they cover 1sec
        focused_seconds = np.clip(int(total_framecount_native / native_fps), 1, self.num_seconds)
        
        if not use_strat:
            use_strat = self.sample_strat
        
        # Extract self.num_seconds 1sec/30frame video blocks from the center of the total video duration
        if use_strat == "center":
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
        if use_strat == "start":
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
        if use_strat == "spread":
            block_frame_starts_native = np.round(np.linspace(subvideo_start_frame, subvideo_end_frame, focused_seconds, endpoint=False))
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