import torch
import json
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
    UniformCropVideo
)
from typing import Dict
from torchvision.models import video

from test import get_mvit_model

# Device on which to run the model
# Set to cuda to load on GPU
device = "cuda"

# TODO: Change pretrained model
# Pick a pretrained model and load the pretrained weights

# SCRIPT PARAMETERS SHOWN BELOW:
model_name="mvit_base"
split = "test"

#model_name = "slowfast_r101"
if "slowfast" in model_name:
    model = torch.hub.load("facebookresearch/pytorchvideo", model=model_name, pretrained=True)
elif "mvit" in model_name:
    model = get_mvit_model()
    
#import pdb; pdb.set_trace()
#model = video.mvit_v1_b()

# Set to eval mode and move to desired device
model = model.to(device)
model = model.eval()

with open("kinetics_classnames.json", "r") as f:
    kinetics_classnames = json.load(f)

# Create an id to label name mapping
kinetics_id_to_classname = {}
for k, v in kinetics_classnames.items():
    kinetics_id_to_classname[v] = str(k).replace('"', "")

# TODO: Change to match new models
####################
# SlowFast transform
####################

side_size = 256

crop_size = 224 
frames_per_second = 30
alpha = 4
mean = [0.45, 0.45, 0.45]
std = [0.225, 0.225, 0.225]

if "slowfast" in model_name:
    num_frames = 32
    sampling_rate = 2

elif "mvit" in model_name:
    num_frames = 16
    sampling_rate = 4


class PackPathway(torch.nn.Module):
    """
    Transform for converting video frames as a list of tensors.
    """
    def __init__(self):
        super().__init__()

    def forward(self, frames: torch.Tensor):
        fast_pathway = frames
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // alpha
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
        return frame_list

transform = None

if "slowfast" in model_name:
    transform =  ApplyTransformToKey(
        key="video",
        transform=Compose(
            [
                UniformTemporalSubsample(num_frames),
                Lambda(lambda x: x/255.0),
                NormalizeVideo(mean, std),
                ShortSideScale(
                    size=side_size
                ),
                CenterCropVideo(crop_size),
                PackPathway()
            ]
        ),
    )

if "mvit" in model_name:
    transform = ApplyTransformToKey(
        key='video',
        transform=Compose(
          transforms=[
            UniformTemporalSubsample(num_samples=num_frames),
            Lambda(lambda x: x/255.0),
            NormalizeVideo(mean, std),
            ShortSideScale(side_size),
            CenterCropVideo(crop_size)
          ]
        )
    )
    
# The duration of the input clip is also specific to the model.
clip_duration = (num_frames * sampling_rate)/frames_per_second

# Function to extract embeddings from a given model
def get_embeds(model, x) -> torch.Tensor:
    with torch.no_grad():
        if "slowfast" in model_name:
            num_blocks = len(model.blocks)
            for idx, block in enumerate(model.blocks):
                if idx != num_blocks - 1: # Skip last block to get embeddings instead of preds
                    x = block(x)
                    x = block.output_pool(x)
            return x.flatten()
        if "mvit" in model_name:
            x = torch.stack(x)
            # c,b,t,h,w --> b,c,t,w,h
            x = x.permute(1,0,2,3,4)
            x = model.patch_embed(x)
            x = model.cls_positional_encoding(x)
            #x = self.model.pos_drop(x)

            thw = model.cls_positional_encoding.patch_embed_shape
            for i, blk in enumerate(model.blocks):
                x, thw = blk(x, thw)
            x = model.norm_embed(x)
            x = model.head.sequence_pool(x)
            return x.flatten()


def get_vid_embeds(video_path):
    # Select the duration of the clip to load by specifying the start and end duration
    # The start_sec should correspond to where the action occurs in the video
    start_sec = 0
    end_sec = start_sec + clip_duration

    # Initialize an EncodedVideo helper class
    video = EncodedVideo.from_path(video_path)

    # Load the desired clip
    video_data = video.get_clip(start_sec=start_sec, end_sec=end_sec)

    # Apply a transform to normalize the video input
    video_data = transform(video_data)

    # Move the inputs to the desired device
    inputs = video_data["video"]
    inputs = [i.to(device)[None, ...] for i in inputs]
    return get_embeds(model, inputs)

# Load the example video
import numpy as np
from tqdm import tqdm
train_paths = np.load("full_vlm_embeddings/clip_embeddings/smsm.v."+ split +"/video_paths.npy")
embeds = []
for path in tqdm(train_paths):
    embeds.append(get_vid_embeds(path).cpu().numpy())
embeds = np.array(embeds)
np.save(model_name + '_' + split + "_embeddings.npy", embeds)