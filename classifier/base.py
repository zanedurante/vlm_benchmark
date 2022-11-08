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
        query_video_paths (np.array):       Array of query video paths to be predicted.
                                            Shape = (n_predict,).
        val_tuning_video_paths (Optional[np.array]):  Optional set of video paths from val split which the classifier can use to select the best-performing model/epoch.
        val_tuning_video_labels (Optional[np.array]): Labels for val_tuning_video_paths.
    Returns:
        (np.array):                         Predicted category index (with respect to the first index of the given
                                            category names and support videos) for each query video path.
                                            Shape = (n_predict,).
    '''
    def predict(self, category_names: np.ndarray, support_video_paths: Optional[np.ndarray], query_video_paths: np.ndarray,
                val_tuning_video_paths: Optional[np.array] = None, val_tuning_video_labels: Optional[np.array] = None) -> np.ndarray:
        raise NotImplementedError