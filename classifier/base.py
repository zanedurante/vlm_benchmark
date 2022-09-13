from typing import Optional
import numpy as np

from SimilarityVLM import SimilarityVLM
from similarity_metrics import Similarity


'''
Class for adapting a SimilarityVLM as a few-shot classifier,
using a ProtoNet-style algorithm.
'''

class FewShotClassifier:
    '''
    Args:
        vlm (SimilarityVLM):            The vlm to use to embed video and text
        metric (Similarity | None):     The similarity metric to use, if None uses vlm default
    '''
    def __init__(self, vlm: SimilarityVLM, metric: Optional[Similarity] = None) -> None:
        self.vlm = vlm
        self.metric = metric or vlm.default_similarity_metric()
        
    '''
    Returns a dict with the value of all classifier-specific parameters which may affect prediction
    accuracy (apart from the underlying VLM object used).
    This is used to differentiate test results which use different classifier parameters.
    '''
    def params(self) -> dict:
        return {
            "metric": self.metric.name
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
        
        flat_query_embeds = np.vstack([self.vlm.get_video_embeds(vid) for vid in query_video_paths.flatten()])
        query_embeds = flat_query_embeds.reshape(n_way, n_query, -1)
        
        # Create Category Prototypes (n_way, embed_dim)
        support_embeds = [] # Each element should have shape (n_way, n_supporting_embeds, embed_dim)
        
        text_embeds = np.vstack([self.vlm.get_text_embeds(name) for name in category_names])
        support_embeds.append(text_embeds[:, None, :])
        
        if n_support > 0:
            flat_support_embeds = np.vstack([self.vlm.get_video_embeds(vid) for vid in support_video_paths.flatten()])
            support_vid_embeds = flat_support_embeds.reshape(n_way, n_support, -1)
            support_embeds.append(support_vid_embeds)
        
        support_embeds = np.concatenate(support_embeds, axis=1)
        prototype_embeds = np.average(support_embeds, axis=1)
        
        # Compare query similarity to prototypes
        flat_query_to_proto_similarities = self.metric(flat_query_embeds, prototype_embeds)
        query_to_proto_similarities = flat_query_to_proto_similarities.reshape(n_way, n_query, n_way)
        
        # Choose category index with max similarity for each query
        query_category_index_predictions = np.argmax(query_to_proto_similarities, axis=2)
        
        return query_category_index_predictions