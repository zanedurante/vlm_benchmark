from typing import Optional
import numpy as np

from SimilarityVLM import SimilarityVLM
from similarity_metrics import Similarity
from .weighted_average import WeightedTextFewShotClassifier

'''
Prototype-based classifier (WeightedTextFewShotClassifier) which additionally
adds a fixed text prompt to the each text input.
'''

class HardPromptFewShotClassifier(WeightedTextFewShotClassifier):
    '''
    Args:
        vlm (SimilarityVLM):            The vlm to use to embed video and text
        metric (Similarity | None):     The similarity metric to use, if None uses vlm default
        text_weight (float):            Relative weight of text embeddings compared to video embeddings when computing class prototypes
        prompt_text (str):              Text prompt to be added to the beginning/end of each text input before encoding
        prompt_location (str):          Strategy to insert text prompt. Must be in ["start", "end"]
    '''
    def __init__(self, vlm: SimilarityVLM, metric: Optional[Similarity] = None, text_weight: float = 1.0,
                 prompt_text: str = "", prompt_location: str = "start") -> None:
        super().__init__(vlm=vlm, metric=metric, text_weight=text_weight)
        
        assert prompt_location in ["start", "end"]
        
        self.prompt_text = prompt_text.strip().lower()
        self.prompt_location = prompt_location
        
    '''
    Returns a dict with the value of all classifier-specific parameters which may affect prediction
    accuracy (apart from the underlying VLM object used).
    This is used to differentiate test results which use different classifier parameters.
    '''
    def params(self) -> dict:
        return {
            "metric": self.metric.name,
            "text_weight": self.text_weight,
            "prompt_text": self.prompt_text,
            "prompt_location": self.prompt_location
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
    Returns:
        (np.array):                         Predicted category index (with respect to the first index of the given
                                            category names and support videos) for each query video path.
                                            Shape = (n_predict,).
    '''
    def predict(self, category_names: np.ndarray, support_video_paths: Optional[np.ndarray], query_video_paths: np.ndarray) -> np.ndarray:
        if self.prompt_text != "":
            if self.prompt_location == "start":
                category_names = np.array([f"{self.prompt_text} {name.strip()}" for name in category_names])

            elif self.prompt_location == "end":
                category_names = np.array([f"{name.strip()} {self.prompt_text}" for name in category_names])
                
        return super().predict(category_names, support_video_paths, query_video_paths)