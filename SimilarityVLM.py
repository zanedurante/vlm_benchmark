from abc import ABC, abstractmethod
from typing import Optional
import torch
import pickle
import os
import json
import shelve
from functools import lru_cache

from similarity_metrics import Similarity

# Reduce I/O load from repeatedly reading the disk-cached embeddings on the same vlm instance
# NOTE: Using @lru_cache as a decorator has the side effect that the SimilarityVLM instance won't be garbage collected
# (If this becomes a problem, you can use self.func = lru_cache(self._func, maxsize=...) instead)
MEM_CACHE_SIZE = 2**14

class SimilarityVLM(ABC):
    """
    Abstract Base Class (ABC) for similarity-based VLMs.  In general, these models can take video
    and language separately and embed each modality into a joint text/video embedding space (like CLIP).
    """

    def __init__(self, cache_file=None, reset_cache=False):
        """
        Sets up embedding cache, leaves model-specific setup and loading to subclass __init__().
        :param cache_file: File to a cache file for precomputing video/text embeddings and enabling faster computation.
        :param cache_dir: Directory to store cached embeddings.
        :param reset_cache: Whether to delete (reset) the existing cache.  This should=True when changes to the
                model or data loading have been made and the video embeddings need to be recomputed.
        """

        # Load cache and set cache flags
        self.cache_file = cache_file
        self.reset_cache = reset_cache
        
        self.disk_cache = None
        if self.cache_file is not None:
            self.disk_cache = shelve.open(self.cache_file)
            
            if self.reset_cache:
                self.disk_cache.clear()
                #self.disk_cache.close()
                #self.disk_cache = shelve.open(self.cache_file)
        
    def params(self) -> dict:
        """
        Specify the value of all VLM-specific parameters which may affect prediction accuracy.
        This is used to differentiate test results which use different versions of the same VLM.
        :return:
        :rtype: dict
        """
        return {}
                
    def cache_item_key(self, video_path: Optional[str] = None, text: Optional[str] = None) -> str:
        """Generates the cache item key for a given video path or text input. This key should uniquely
        identify a possible embedding this vlm could produce.

        Args:
            video_path (Optional[str], optional): _description_. Defaults to None.
            text (Optional[str], optional): _description_. Defaults to None.

        Returns:
            str: _description_
        """
        assert sum([video_path is not None, text is not None]) == 1, "Exactly one arg video_path or text must be set"
        
        key_dict = self.params()
        if video_path is not None:
            key_dict["video_path"] = video_path
        elif text is not None:
            key_dict["text"] = text
        else:
            raise ValueError
        
        return json.dumps(key_dict)

    @lru_cache(maxsize=MEM_CACHE_SIZE)
    def get_text_embeds(self, text):
        """
        Embeds text one string at a time
        :param text: String to embed
        :return: Pytorch embedding tensor for the text
        """
        cache_item_key = self.cache_item_key(text=text)
        if self.disk_cache is not None and cache_item_key in self.disk_cache:
            return self.disk_cache[cache_item_key]
        
        text_embed = self.text_encoder(text)
        
        if self.disk_cache is not None:
            self.disk_cache[cache_item_key] = text_embed
            #self.disk_cache.close()
            #self.disk_cache = shelve.open(self.cache_file)
            
        return text_embed

    @lru_cache(maxsize=MEM_CACHE_SIZE)
    def get_video_embeds(self, video_path):
        """
        Embeds video one video tensor at a time
        TODO: See if we should change to encode batches of videos
        :param path: Path to the video
        :return:
        """
        cache_item_key = self.cache_item_key(video_path=video_path)
        if self.disk_cache is not None and cache_item_key in self.disk_cache:
            return self.disk_cache[cache_item_key]
        
        vid_embed = self.video_encoder(video_path)
        
        if self.disk_cache is not None:
            self.disk_cache[cache_item_key] = vid_embed
            #self.disk_cache.close()
            #self.disk_cache = shelve.open(self.cache_file)
            
        return vid_embed

    @abstractmethod
    def text_encoder(self, text):
        """
        Tokenize and encode text into a joint text/video embedding space
        :param tokens:
        :return:
        """
        pass

    @abstractmethod
    def video_encoder(self, video_path):
        """
        Load, transform and encode a video file into a joint text/video embedding space
        :param video:
        :return:
        """
        pass

    @abstractmethod
    def default_similarity_metric(self) -> Similarity:
        """
        Returns a reference to the default similarity metric used by this VLM
        :return:
        """
        pass
