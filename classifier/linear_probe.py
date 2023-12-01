from typing import Optional
import numpy as np
from sklearn.linear_model import LogisticRegression

from SimilarityVLM import SimilarityVLM
from .base import FewShotClassifier

'''
Simplest Linear Probe Classifier
'''

class LinearProbe(FewShotClassifier):
    def __init__(self, vlm: SimilarityVLM, regularization: float):
        self.vlm = vlm
        self.regularization = float(regularization)
        
    '''
    Returns a dict with the value of all classifier-specific parameters which may affect prediction
    accuracy (apart from the underlying VLM object used).
    This is used to differentiate test results which use different classifier parameters.
    '''
    def params(self) -> dict:
        return {
            "regularization": self.regularization
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
    def predict(self, category_names: np.ndarray, support_video_paths: Optional[np.ndarray], query_video_paths: np.ndarray, query_video_labels: np.ndarray,
                val_tuning_video_paths: Optional[np.array] = None, val_tuning_video_labels: Optional[np.array] = None) -> np.ndarray:
        n_way = category_names.shape[0]
        n_predict = query_video_paths.shape[0]
        if support_video_paths is not None:
            n_support = support_video_paths.shape[1]
        else:
            n_support = 0
            
            
        # Use default similarity to text embeds if zero-shot
        if n_support == 0:
            query_embeds = np.array([self.vlm.get_video_embeds(vid) for vid in query_video_paths])
            text_embeds = np.array([self.vlm.get_text_embeds(name) for name in category_names])
            query_to_text_similarities = self.vlm.default_similarity_metric()(query_embeds, text_embeds)
            query_predictions = np.argmax(query_to_text_similarities, axis=1)
            accuracy = (query_predictions == query_video_labels).mean()
            return accuracy
        
        # Linear probe ignoring text embeds
        query_embeds = np.array([self.vlm.get_video_embeds(vid) for vid in query_video_paths])
        support_embeds = np.array([self.vlm.get_video_embeds(vid) for vid in support_video_paths.flatten()])
        support_labels = np.repeat(np.arange(n_way), n_support)
        
        classifier = LogisticRegression(C = 1/self.regularization, max_iter=1000)
        classifier.fit(support_embeds, support_labels)
        
        query_predictions = classifier.predict(query_embeds)
        accuracy = (query_predictions == query_video_labels).mean()
        return accuracy