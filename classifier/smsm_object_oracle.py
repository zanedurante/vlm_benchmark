from typing import Optional
import numpy as np
import json

from SimilarityVLM import SimilarityVLM
from similarity_metrics import Similarity
from .base import FewShotClassifier


'''
Class for testing the potential benefits of object detection with regards to class text embeddings.
Only valid for smsm, since we have metadata indicating class text templates and object nouns to insert for each video.
'''

SMSM_METADATA_PATHS = [
    "/home/datasets/something-something-labels/train.json",
    "/home/datasets/something-something-labels/validation.json"
]

class SmsmObjectOracleFewShotClassifier(FewShotClassifier):
    '''
    Args:
        vlm (SimilarityVLM):            The vlm to use to embed video and text
        metric (Similarity | None):     The similarity metric to use, if None uses vlm default
    '''
    def __init__(self, vlm: SimilarityVLM, metric: Optional[Similarity] = None) -> None:
        self.vlm = vlm
        self.metric = metric or vlm.default_similarity_metric()
        
        # Load smsm metadata
        self.smsm_video_objects = {} # {video id string -> list of placeholder nouns}
        for metadata_path in SMSM_METADATA_PATHS:
            with open(metadata_path, "r") as fp:
                metadata = json.load(fp)
                
            for video_info in metadata:
                self.smsm_video_objects[video_info["id"]] = video_info["placeholders"]
        
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
        n_way = category_names.shape[0]
        n_predict = query_video_paths.shape[0]
        if support_video_paths is not None:
            n_support = support_video_paths.shape[1]
        else:
            n_support = 0
        
        # Initial test is just for 0-shot
        if n_support != 0:
            raise NotImplementedError
        
        # For each query video, find the query video's placeholder nouns, then insert them into every category name, then embed all and predict via similarity
        predictions = []
        for query_vid_path in query_video_paths:
            query_vid_id = query_vid_path.split("/")[-1].split(".")[0]
            if query_vid_id not in self.smsm_video_objects:
                raise ValueError(f"{query_vid_id} not in smsm metadata. Path = {query_vid_path}")
            
            query_vid_objects = self.smsm_video_objects[query_vid_id]
            
            new_category_names = []
            for name in category_names:
                name_between_placeholders = name.split("something")
                
                # Interleave name segments with query video objects
                # If there are more query video objects than "something"s in the category name, use the early ones first
                # If there are less query video objects than "something"s in the category name, repeat the last query video object
                name_with_objects = [name_between_placeholders[0]]
                for i in range(0, len(name_between_placeholders) - 1):
                    if i < len(query_vid_objects):
                        name_with_objects.append(query_vid_objects[i])
                    else:
                        name_with_objects.append(query_vid_objects[-1])
                    name_with_objects.append(name_between_placeholders[i + 1])
                    
                name_with_objects = "".join(name_with_objects)
                new_category_names.append(name_with_objects)
                
            # Create Text Embeddings
            text_embeds = np.array([
                self.vlm.get_text_embeds(name)
                for name in new_category_names
            ])
            
            # Create Query Vid Embedding
            query_embed = self.vlm.get_video_embeds(query_vid_path)
            
            # Similarity between query video and altered text embeddings
            query_to_text_similarities = self.metric(query_embed[None, :], text_embeds)[0, :]
            
            # Use best-similarity index as prediction
            predictions.append(np.argmax(query_to_text_similarities))
            
        return np.array(predictions)