from typing import Optional
import numpy as np

from SimilarityVLM import SimilarityVLM
from similarity_metrics import Similarity
from .base import FewShotClassifier







'''
Each class prototype is a Gaussian distribution with a diagonal covar matrix. (Variance can differ between embedding dimensions)
Allows setting an initial variance prior.
'''
class GaussianFewShotClassifier(FewShotClassifier):
    '''
    Args:
        vlm (SimilarityVLM):                The vlm to use to embed video and text
        text_weight (float):                Relative weight of text embeddings compared to video embeddings when computing class prototypes
        prior_count (int):                  Prior over possible Gaussian variances (assuming fixed mean), as if one had seen
                                            prior_count examples with prior_var variance before seeing examples from task itself.
        prior_var (float):                  Prior over possible Gaussian variances (assuming fixed mean), as if one had seen
                                            prior_count examples with prior_var variance before seeing examples from task itself.
        normalize (bool):                   Whether VLM embeddings should be normalized before predictions. Can help when VLM embeddings are
                                            trained using cosine or dot-product similarity.
    '''
    def __init__(self, vlm: SimilarityVLM, text_weight: float = 1.0,
                 prior_count: int = 0, prior_var: float = 0, normalize: bool = True):
        
        self.vlm = vlm
        self.text_weight = text_weight
        self.prior_count = int(prior_count)
        self.prior_var = float(prior_var)
        self.normalize = normalize
        
    '''
    Returns a dict with the value of all classifier-specific parameters which may affect prediction
    accuracy (apart from the underlying VLM object used).
    This is used to differentiate test results which use different classifier parameters.
    '''
    def params(self) -> dict:
        return {
            "text_weight": self.text_weight,
            "prior_count": self.prior_count,
            "prior_var": self.prior_var,
            "normalize": self.normalize
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
        if self.normalize:
            flat_query_embeds /= np.linalg.norm(flat_query_embeds, axis=1, keepdims=True)
        query_embeds = flat_query_embeds.reshape(n_way, n_query, -1)
        
        # Create Category Prototypes (n_way, embed_dim)
        support_embeds = [] # Each element should have shape (n_way, n_supporting_embeds, embed_dim)
        support_embed_weights = []
        
        text_embeds = np.vstack([self.vlm.get_text_embeds(name) for name in category_names])
        if self.normalize:
            text_embeds /= np.linalg.norm(text_embeds, axis=1, keepdims=True)
        support_embeds.append(text_embeds[:, None, :])
        support_embed_weights += [self.text_weight]
        
        if n_support > 0:
            flat_support_embeds = np.vstack([self.vlm.get_video_embeds(vid) for vid in support_video_paths.flatten()])
            if self.normalize:
                flat_support_embeds /= np.linalg.norm(flat_support_embeds, axis=1, keepdims=True)
            support_vid_embeds = flat_support_embeds.reshape(n_way, n_support, -1)
            support_embeds.append(support_vid_embeds)
            support_embed_weights += [1] * n_support
        
        support_embeds = np.concatenate(support_embeds, axis=1)                     # Shape = (n_way, n_support + 1, embed_dim)
        
        # Compute weighted mean for each class
        means = np.average(support_embeds, axis=1, weights=support_embed_weights)   # Shape = (n_way, embed_dim)
        
        # Compute weighted variance for each class and each embedding dimension (including specified prior)
        # Simulates seeing <prior_count> examples with <prior_var> variance
        prior_vars = np.repeat(
            np.ones_like(means)[:, None, :] * self.prior_var / (self.prior_count if self.prior_count > 0 else 1),
            self.prior_count, axis=1
        )   # Shape = (n_way, prior_count, embed_dim)
        prior_var_weights = [1] * self.prior_count
        
        support_vars = np.square(support_embeds - means[:, None, :])    # Shape = (n_way, n_support + 1, embed_dim)
        support_var_weights = support_embed_weights
        
        all_support_vars = np.concatenate([prior_vars, support_vars], axis=1)
        all_support_var_weights = prior_var_weights + support_var_weights
        vars = np.average(all_support_vars, axis=1, weights=all_support_var_weights) # Shape = (n_way, embed_dim)
        
        # If no prior and only one example, variance will be 0. In this case, set all variances to 1
        if np.any(vars == 0):
            vars = np.ones_like(vars)
        
        # Predict relative query likelihood under each gaussian class distribution
        flat_query_log_likelihoods = np.sum(
            -0.5 * np.log(vars[None, :, :]) - 0.5 * np.square(flat_query_embeds[:, None, :] - means[None, :, :]) / vars[None, :, :],
            axis=2
        )   # Shape = (n_way * n_query, n_way)
        
        flat_query_predictions = np.argmax(flat_query_log_likelihoods, axis=1)
        query_predictions = flat_query_predictions.reshape(n_way, n_query)
        return query_predictions