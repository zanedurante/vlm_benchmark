from SimilarityVLM import SimilarityVLM
from transformers import CLIPModel, CLIPTokenizer, CLIPProcessor
from similarity_metrics import Similarity
import numpy as np
import decord
import random


class ClipVLM(SimilarityVLM):
    """
    Similarity-based VLM that uses CLIP for frame and text encoders.  Currently, we use the hugging face implementation
    for CLIP since it is easier to set up.
    TODO: Implement the larger version of CLIP since this should get better performance.
    """

    def __init__(self, path="openai/clip-vit-base-patch32", num_frames=1, sample_strat='uniform', cache_file=None,
                 cache_dir=None, reset_cache=False):
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.num_frames = num_frames  # The number of frames CLIP will use to classify videos
        self.sample_strat = sample_strat  # 'rand' or 'uniform'

        decord.bridge.set_bridge("torch")  # Video loader

        super().__init__(path, cache_file=cache_file, cache_dir=cache_dir, reset_cache=reset_cache)

    def load_model(self, path="openai/clip-vit-base-patch32"):
        """
        Loads the model from the weights specified in `path`
        :param path:
        :return:
        """
        self.model = CLIPModel.from_pretrained(path)
        self.tokenizer = CLIPTokenizer.from_pretrained(path)
        self.processor = CLIPProcessor.from_pretrained(path)
        return

    def tokenize(self, text):
        """
        Tokenizes text via tokenizer (likely variant of huggingface BertTokenizer)
        :param text:, list of text to tokenize
        :return: Tokenized text
        """
        inputs = self.tokenizer(text, padding=True, return_tensors="pt")
        return inputs

    def text_encoder(self, tokens):
        """
        Encodes tokenized text into joint text/video embedding space
        :param tokens:
        :return:
        """
        text_features = self.model.get_text_features(**tokens)
        return text_features

    def open_video(self, path):
        """
        Opens video and returns basic, non-transformed video tensor
        :param path:
        :return:
        """
        # TODO: Figure out best way to subsample the video with CLIP (in the paper they just use one single frame)
        video_reader = decord.VideoReader(path, num_threads=1)
        vlen = len(video_reader)
        frame_idxs = self.get_frame_idxs(vlen)
        frames = video_reader.get_batch(frame_idxs)
        frames = frames.float() / 255
        frames = frames.permute(0, 3, 1, 2).squeeze()  # Get rid of single frame dimension
        return frames

    def transform(self, video):
        """
        Transforms video using model-specific transforms
        :param video:
        :return:
        """
        inputs = self.processor(images=video, return_tensors="pt")
        return inputs

    def video_encoder(self, video):
        """
        Encodes transformed video into joint text/video embedding space
        :param video:
        :return:
        """
        video_features = self.model.get_image_features(**video)  # Frame-level video features
        return video_features

    def default_similarity_metric(self) -> Similarity:
        """
        Returns a reference to the default similarity metric used by this VLM
        :return:
        """
        return Similarity.COSINE

    def get_frame_idxs(self, vlen):
        # Determine number of samples
        num_samples = min(self.num_frames, vlen)

        # Determine intervals of indices to sample from (e.g. [0, 3, 6, 10] for num_frames=3, vlen=10)
        intervals = np.linspace(start=0, stop=vlen, num=num_samples + 1).astype(int)
        ranges = []
        for idx, interval in enumerate(intervals[:-1]):
            ranges.append((interval, intervals[idx + 1] - 1))
        if self.sample_strat == 'rand':
            frame_idxs = [random.choice(range(x[0], x[1])) for x in ranges]
        elif self.sample_strat == 'uniform':
            frame_idxs = [(x[0] + x[1]) // 2 for x in ranges]
        else:
            raise NotImplementedError

        return frame_idxs

