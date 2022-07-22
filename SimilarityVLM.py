from abc import ABC, abstractmethod
import torch
import pickle
import os


class SimilarityVLM(ABC):
    """
    Abstract Base Class (ABC) for similarity-based VLMs.  In general, these models can take video
    and language separately as input and embed them in a joint embedding space (like CLIP).  Since they can embed video
    and language separately.
    """

    def __init__(self, path, cache_file=None, cache_dir=None, reset_cache=False):
        """
        :param cache_file: File to a cache file for precomputing video embeddings and enabling faster computation.
        :param cache_dir: Directory to store cached embeddings.
        :param reset_cache: Whether to delete (reset) the existing cache.  This should=True when changes to the
                model or data loading have been made and the video embeddings need to be recomputed.
        """

        # Load cache and set cache flags
        self.use_cache = False  # Set to true in load_cache if cache_file is not None
        self.cache_file = cache_file
        self.cache_dir = cache_dir
        self.reset_cache = reset_cache
        self.embed_cache = {}  # Initialize self.embed_cache to empty dictionary, maps video path --> tensor path
        self.load_cache(cache_file)  # Initialize self.embed_cache

        # Load video and text encoders
        self.video_encoder = None
        self.text_encoder = None
        self.load_model(path)

    def load_cache(self):
        """
        Loads cache of precomputed video embeddings for each video path string
        :return:
        """
        if self.cache_file is not None:
            self.use_cache = True
        with open(self.cache_file, "rb") as cf:
            self.embed_cache = pickle.load(cf)

    def save_cache(self):
        """
        Saves the cached video embeddings.  Note: this needs to be called in the script using the SimilarityVLM
        :return:
        """
        with open(self.cache_file, "wb") as cf:
            pickle.dump(self.embed_cache, cf)

    def cache(self, path, video_embed):
        """
        Caches video embedding
        :param path: Path to video file
        :param video_embed: Embedding created by Similarity VLM
        :return:
        """
        path_to_cached = os.path.join(self.cache_dir, path)
        torch.save(video_embed, path_to_cached)
        self.embed_cache[path] = path_to_cached
        return

    def get_text_embeds(self, text):
        """
        Embeds text one string at a time
        :param text: List of strings to embed
        :return: Pytorch embedding tensor for the text
        TODO: Cache text embeddings
        """
        tokens = self.tokenize(text)
        text_embed = self.text_encoder(tokens)
        return text_embed

    def get_video_embeds(self, path):
        """
        Embeds video one video tensor at a time
        TODO: See if we should change to encode batches of videos
        :param path: Path to the video
        :return:
        """
        if path in self.embed_cache:
            return torch.load(self.embed_cache[path])  # Note: May need to add .cuda() or change dtype

        video = self.open_video(path)
        video = self.transform(video)
        video_embed = self.video_encoder(video)
        self.cache(path, video_embed)

        return video_embed

    @abstractmethod
    def load_model(self, path):
        """
        Loads the model from the weights specified in `path`
        :param path:
        :return:
        """
        return

    @abstractmethod
    def tokenize(self, text):
        """
        Tokenizes text via tokenizer (likely variant of huggingface BertTokenizer)
        :param text:
        :return: Tokenized text
        """
        pass

    @abstractmethod
    def text_encoder(self, tokens):
        """
        Encodes tokenized text into joint text/video embedding space
        :param tokens:
        :return:
        """
        pass

    @abstractmethod
    def open_video(self, path):
        """
        Opens video and returns basic, non-transformed video tensor
        :param path:
        :return:
        """
        pass

    @abstractmethod
    def transform(self, video):
        """
        Transforms video using model-specific transforms
        :param video:
        :return:
        """
        pass

    @abstractmethod
    def video_encoder(self, video):
        """
        Encodes transformed video into joint text/video embedding space
        :param video:
        :return:
        """
        pass

    @abstractmethod
    def get_similarity(self, text_embed, video_embed):
        """
        Similarity score between text and video embeddings
        :param text_embed:
        :param video_embed:
        :return: Float where higher number --> more similar
        """
        # TODO: Add class for distance functions for easier configuration
        pass
