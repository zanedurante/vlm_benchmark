from typing import Optional
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from SimilarityVLM import SimilarityVLM
from similarity_metrics import Similarity
from .base import FewShotClassifier


'''
Class for adapting a SimilarityVLM as a few-shot classifier,
naively chooses the class which has the closest embedding to the query
(whether video or text). 
'''

class NearestNeighborFewShotClassifier(FewShotClassifier):
    '''
    Args:
        vlm (SimilarityVLM):            The vlm to use to embed video and text
        metric (Similarity | None):     The similarity metric to use, if None uses vlm default
        neighbor_count (int):           Number of neighbors which vote for the resulting prediction (K in K-nearest-neighbors)
        neighbor_weights (str):         Method for weighing votes of nearest neighbors. Must be in ["uniform", "distance"]
    '''
    def __init__(self, vlm: SimilarityVLM, metric: Optional[Similarity] = None, neighbor_count: int = 1, neighbor_weights: str = "uniform") -> None:
        super().__init__(vlm, metric)
        
        assert type(neighbor_count) is int and neighbor_count > 0
        assert neighbor_weights in ["uniform", "distance"]
        
        self.neighbor_count = neighbor_count
        self.neighbor_weights = neighbor_weights
        
        if self.metric is Similarity.COSINE:
            self.sklearn_metric = "cosine"
        elif self.metric is Similarity.DOT:
            self.sklearn_metric = lambda a, b: np.exp(-Similarity.DOT(a[None, :], b[None, :])[0, 0])
        else:
            raise NotImplementedError
        
        
    '''
    Returns a dict with the value of all classifier-specific parameters which may affect prediction
    accuracy (apart from the underlying VLM object used).
    This is used to differentiate test results which use different classifier parameters.
    '''
    def params(self) -> dict:
        return {
            "metric": self.metric.name,
            "neighbor_count": self.neighbor_count,
            "neighbor_weights": self.neighbor_weights
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
        
        # Collect all support example embeddings (text embeds followed by n_support video embeds)
        support_embeds = [] # Each element should have shape (n_way, n_supporting_embeds, embed_dim)
        
        text_embeds = np.vstack([self.vlm.get_text_embeds(name) for name in category_names])
        support_embeds.append(text_embeds[:, None, :])
        
        if n_support > 0:
            flat_support_embeds = np.vstack([self.vlm.get_video_embeds(vid) for vid in support_video_paths.flatten()])
            support_vid_embeds = flat_support_embeds.reshape(n_way, n_support, -1)
            support_embeds.append(support_vid_embeds)
        
        support_embeds = np.concatenate(support_embeds, axis=1)
        flat_support_embeds = support_embeds.reshape(n_way * (n_support + 1), -1)
        support_category_inds = (np.arange(n_way)[:, None] * np.ones((n_way, n_support + 1))).astype(int)
        flat_support_category_inds = support_category_inds.flatten()
        
        # Fit KNN classifier
        knn = KNeighborsClassifier(
            n_neighbors=min(self.neighbor_count, n_way * (n_support + 1)),
            weights=self.neighbor_weights,
            metric=self.sklearn_metric
        )
        knn.fit(flat_support_embeds, flat_support_category_inds)
        
        # Predict for query embeds
        flat_query_predictions = knn.predict(flat_query_embeds)
        query_predictions = flat_query_predictions.reshape(n_way, n_query)
        
        return query_predictions