from abc import ABC, abstractmethod
import torch
import pickle
import os

from similarity_metrics import Similarity


class SimilarityVLM(ABC):
    """
    Abstract Base Class (ABC) for similarity-based VLMs.  In general, these models can take video
    and language separately and embed each modality into a joint text/video embedding space (like CLIP).
    """

    def __init__(self, cache_file=None, cache_dir=None, reset_cache=False):
        """
        Sets up embedding cache, leaves model-specific setup and loading to subclass __init__().
        :param cache_file: File to a cache file for precomputing video/text embeddings and enabling faster computation.
        :param cache_dir: Directory to store cached embeddings.
        :param reset_cache: Whether to delete (reset) the existing cache.  This should=True when changes to the
                model or data loading have been made and the video embeddings need to be recomputed.
        """

        # Load cache and set cache flags
        self.use_cache = False  # Set to true in load_cache if cache_file is not None
        self.cache_file = cache_file
        self.cache_dir = cache_dir
        self.reset_cache = reset_cache
        self.embed_cache = {}  # Initialize self.embed_cache to empty dictionary, maps video-path or text --> tensor path
        self.load_cache()  # Initialize self.embed_cache
        
    def params(self) -> dict:
        """
        Specify the value of all VLM-specific parameters which may affect prediction accuracy.
        This is used to differentiate test results which use different versions of the same VLM.
        :return:
        :rtype: dict
        """
        return {}

    def load_cache(self):
        """
        Loads cache of precomputed video embeddings for each video path string
        :return:
        """
        if self.cache_file is not None and self.cache_dir is not None:
            self.use_cache = True
            
        # Optionally delete cache before init
        if self.use_cache and self.reset_cache:
            if os.path.exists(self.cache_file):
                os.remove(self.cache_file)
            if os.path.exists(self.cache_dir):
                for embed_file in os.listdir(self.cache_dir):
                    os.remove(os.path.join(self.cache_dir, embed_file))
                os.rmdir(self.cache_dir)
                
        # Init cache
        if self.use_cache:
            if os.path.exists(self.cache_file) and os.path.exists(self.cache_dir):
                with open(self.cache_file, "rb") as cf:
                    self.embed_cache = pickle.load(cf)
            else:
                os.makedirs(self.cache_dir, exist_ok=True)
                self.embed_cache = {}

    def save_cache(self):
        """
        Saves the cached video embeddings.  Note: this needs to be called in the script using the SimilarityVLM
        :return:
        """
        if self.use_cache:
            with open(self.cache_file, "wb") as cf:
                pickle.dump(self.embed_cache, cf)

    def cache_video(self, path, video_embed):
        """
        Caches video embedding
        :param path: Path to video file
        :param video_embed: Embedding created by Similarity VLM
        :return:
        """
        if self.use_cache:
            embed_filename = path.strip("/").replace("/", ".")
            torch.save(video_embed, os.path.join(self.cache_dir, embed_filename))
            self.embed_cache[path] = embed_filename
            
    def cache_text(self, text, text_embed):
        """
        Caches text embedding
        :param text: Input text (lower case)
        :param text_embed: Embedding created by Similarity VLM
        :return:
        """
        if self.use_cache:
            embed_filename = text.replace(" ", "_")
            torch.save(text_embed, os.path.join(self.cache_dir, embed_filename))
            self.embed_cache[text] = embed_filename

    def get_text_embeds(self, text):
        """
        Embeds text one string at a time
        :param text: List of strings to embed
        :return: Pytorch embedding tensor for the text
        """
        if text in self.embed_cache:
            return torch.load(os.path.join(self.cache_dir, self.embed_cache[text]))
        
        text_embed = self.text_encoder(text)
        self.cache_text(text, text_embed)
        
        return text_embed

    def get_video_embeds(self, video_path):
        """
        Embeds video one video tensor at a time
        TODO: See if we should change to encode batches of videos
        :param path: Path to the video
        :return:
        """
        if video_path in self.embed_cache:
            return torch.load(os.path.join(self.cache_dir, self.embed_cache[video_path]))  # Note: May need to add .cuda() or change dtype

        video_embed = self.video_encoder(video_path)
        self.cache_video(video_path, video_embed)

        return video_embed

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
