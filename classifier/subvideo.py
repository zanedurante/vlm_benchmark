from functools import lru_cache
from typing import Optional, Tuple
import numpy as np
import math
import decord

from SimilarityVLM import SimilarityVLM
from similarity_metrics import Similarity
from .base import FewShotClassifier

MEM_CACHE_SIZE = 2**14

'''
Collects video embeddings multiple times along its duration, averaging all
of those subvideo embeddings with the text embedding to construct a class prototype.
'''

class SubVideoAverageFewShotClassifier(FewShotClassifier):
    '''
    Args:
        vlm (SimilarityVLM):                    The vlm to use to embed video and text
        metric (Similarity | None):             The similarity metric to use, if None uses vlm default
        text_weight (float):                    Effective weight of text embeddings compared to individual subvideo embeddings
                                                when averaging for class prototype. Defaults to 1.
        subvideo_segment_duration (float):      Number of seconds assigned to each subvideo. Defaults to 1.
        subvideo_stride_duration (float):       Number of seconds between the starts of consecutive subvideo segments. Defaults to 1.
        subvideo_max_segments (int):            Maximum number of subvideo segments to extract from the video. Defaults to 32
        subvideo_discard_proportion (float):    Proportion of support subvideo embeddings which will not be used for prototype computation,
                                                in order of their similarity with the text embedding. (outlier subvideo embeddings are discarded first).
                                                Must be in range [0, 1). Defaults to 0.
    '''
    def __init__(self, vlm: SimilarityVLM, metric: Optional[Similarity] = None, text_weight: float = 1,
                 subvideo_segment_duration: float = 1, subvideo_stride_duration: float = 1, 
                 subvideo_max_segments: int = 32, subvideo_discard_proportion: float = 0) -> None:
        self.vlm = vlm
        self.metric = metric or vlm.default_similarity_metric()
        self.text_weight = float(text_weight)
        self.subvideo_segment_duration = float(subvideo_segment_duration)
        self.subvideo_stride_duration = float(subvideo_stride_duration)
        self.subvideo_max_segments = int(subvideo_max_segments)
        self.subvideo_discard_proportion = float(subvideo_discard_proportion)
        
        if text_weight < 0:
            raise ValueError(f"Text weight must be at least 0. Got {text_weight}.")
        
        if subvideo_stride_duration <= 0:
            raise ValueError(f"Subvideo stride duration must be greater than 0. Got {subvideo_stride_duration}.")
        
        if subvideo_discard_proportion < 0 or subvideo_discard_proportion >= 1:
            raise ValueError(f"Subvideo discard proportion must be at least 0 and less than 1. Got {subvideo_discard_proportion}.")
        
    '''
    Returns a dict with the value of all classifier-specific parameters which may affect prediction
    accuracy (apart from the underlying VLM object used).
    This is used to differentiate test results which use different classifier parameters.
    '''
    def params(self) -> dict:
        return {
            "metric": self.metric.name,
            "text_weight": self.text_weight,
            "subvideo_segment_duration": self.subvideo_segment_duration,
            "subvideo_stride_duration": self.subvideo_stride_duration,
            "subvideo_max_segments": self.subvideo_max_segments,
            "subvideo_discard_proportion": self.subvideo_discard_proportion
        }
        
    '''
    Predicts categories for a set of query videos in a few-shot task (formatted like FewShotTaskDataset)
    Args:
        category_names (np.array):          Array of names for each given few-shot category.
                                            Shape = (n_way,).
        support_video_paths (np.array):     Array of support video paths for each given few-shot category.
                                            Shape = (n_way, n_support).
                                            Can be None if n_support == 0.
        query_video_paths (np.array):       Array of query video paths to be predicted, associated with each
                                            given category.
                                            Shape = (n_way, n_query).
    Returns:
        (np.array):                         Predicted category index (with respect to the first index of the given
                                            category names and support videos) for each query video path.
                                            Shape = (n_way, n_query).
    '''
    def predict(self, category_names: np.ndarray, support_video_paths: Optional[np.ndarray], query_video_paths: np.ndarray) -> np.ndarray:
        n_way = category_names.shape[0]
        n_query = query_video_paths.shape[1]
        if support_video_paths is not None:
            n_support = support_video_paths.shape[1]
        else:
            n_support = 0
        
        # Create Category Prototypes (n_way, embed_dim)
        prototype_embeds = []
        for cat_ind in range(n_way):
            support_embeds = [] # Each element is a single support embedding
            support_embed_weights = []
            
            # Text embed
            text_embed = self.vlm.get_text_embeds(category_names[cat_ind])
            support_embeds.append(text_embed)
            support_embed_weights.append(self.text_weight)
            
            # Video embeds
            if n_support > 0:
                for support_vid_path in support_video_paths[cat_ind]:
                    subvid_embeds = self.get_subvideo_embeds(support_vid_path)
                    
                    # Discard proportion of subvideo embeddings which are furthest from the text embedding
                    subvideo_discard_count = int(len(subvid_embeds) * self.subvideo_discard_proportion)
                    if subvideo_discard_count > 0:
                        subvid_embeds = np.array(subvid_embeds)
                        text_subvid_similarities = self.vlm.default_similarity_metric()(text_embed[None, :], subvid_embeds)[0, :]
                        sorted_subvid_indices = np.argsort(text_subvid_similarities) # Sorted from low similarity to high
                        subvid_embeds = subvid_embeds[sorted_subvid_indices[subvideo_discard_count:]]
                        subvid_embeds = list(subvid_embeds)
                    
                    support_embeds += subvid_embeds
                    support_embed_weights += [1] * len(subvid_embeds)
                        
            # Construct prototype from average
            support_embeds = np.array(support_embeds)
            prototype_embeds.append(np.average(support_embeds, axis=0, weights=support_embed_weights))
            
        prototype_embeds = np.array(prototype_embeds)
        
        # Collect subvideo embeds for each query video path, check similarity to prototypes and vote
        flat_query_predictions = []
        for query_video_path in query_video_paths.flatten():
            subvid_embeds = np.array(self.get_subvideo_embeds(query_video_path))
            subvid_to_proto_similarities = self.metric(subvid_embeds, prototype_embeds)
            
            # Sort subvids, so those closest to any prototype are first, and will be more important for votes if there is a tie
            subvid_to_best_proto_similarities = np.max(subvid_to_proto_similarities, axis=1)
            subvid_predictions = np.argmax(subvid_to_proto_similarities, axis=1)
            subvid_predictions = subvid_predictions[np.argsort(subvid_to_best_proto_similarities)]
            
            prediction_values, prediction_counts = np.unique(subvid_predictions, return_counts=True)
            prediction = prediction_values[np.argmax(prediction_counts)]
            
            flat_query_predictions.append(prediction)
            
        return np.array(flat_query_predictions).reshape(n_way, n_query)
    
    @lru_cache(maxsize=MEM_CACHE_SIZE)
    def get_video_metadata(self, video_path: str) -> Tuple[int, float]:
        video_reader = decord.VideoReader(video_path)
        return len(video_reader), video_reader.get_avg_fps()
    
    @lru_cache(maxsize=MEM_CACHE_SIZE)
    def get_subvideo_embeds(self, video_path: str) -> list:
        """Computes subvideo embeddings for the given video path

        Args:
            video_path (str): _description_

        Returns:
            np.ndarray: _description_
        """
        video_len, video_fps = self.get_video_metadata(video_path)
        subvideo_frame_bounds = self.get_subvideo_frame_bounds(video_len, video_fps)
        subvideo_embeds = [
            self.vlm.get_video_embeds(video_path, subvideo_start_frame=subvideo_start_frame, subvideo_end_frame=subvideo_end_frame)
            for subvideo_start_frame, subvideo_end_frame in subvideo_frame_bounds
        ]
        
        return subvideo_embeds
    
    def get_subvideo_frame_bounds(self, video_len: int, video_fps: float) -> list:
        """Computes the frame-index bounds for subvideos for the given video duration info.
        Returns a list of 2-int tuples, each containing the start and end frame indices
        for a subvideo index.

        Args:
            video_len (int): Video length in frames.
            video_fps (float): Average video fps.

        Returns:
            list: _description_
        """
        subvideo_segment_frames = int(self.subvideo_segment_duration * video_fps)
        subvideo_stride_frames = max(int(self.subvideo_stride_duration * video_fps), 1)
        
        # Return the full duration if there isn't enough room for at least one subvideo segment
        if video_len < subvideo_segment_frames:
            return [(0, video_len)]
        
        subvideo_segment_count = min(
            1 + (video_len - subvideo_segment_frames) // subvideo_stride_frames,
            self.subvideo_max_segments
        )
        leftover_frames = video_len - subvideo_segment_frames - (subvideo_segment_count - 1) * subvideo_stride_frames
        
        subvideo_starts = subvideo_stride_frames * np.arange(subvideo_segment_count) + \
            np.round(np.linspace(0, leftover_frames, subvideo_segment_count + 1, endpoint=False))[1:] # Add spacing between for leftover frames

        subvideo_ends = subvideo_starts + subvideo_segment_frames
        
        subvideo_bounds = [(start, end) for start, end in zip(subvideo_starts, subvideo_ends)]

        return subvideo_bounds