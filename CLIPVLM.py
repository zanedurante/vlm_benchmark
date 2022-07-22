import SimilarityVLM
from transformers import CLIPModel, CLIPTokenizer


class ClipVLM(SimilarityVLM):
    """
    Similarity-based VLM that uses CLIP for frame and text encoders.  Currently, we use the hugging face implementation
    for CLIP since it is easier to set up.
    TODO: Implement the larger version of CLIP since this should get better performance.
    """

    def __init__(self, path, cache_file=None, cache_dir=None, reset_cache=False):
        self.model = None
        self.tokenizer = None
        self.processor = None
        super().__init__(self, path, cache_file=None, cache_dir=None, reset_cache=False)

    def load_model(self, path="openai/clip-vit-base-patch32"):
        """
        Loads the model from the weights specified in `path`
        :param path:
        :return:
        """
        self.model = CLIPModel.from_pretrained(path)
        self.tokenizer = CLIPTokenizer.from_pretrained(path)
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
        # TODO: Need to figure out how to interface with datasets to do this part
        pass

    def transform(self, video):
        """
        Transforms video using model-specific transforms
        :param video:
        :return:
        """
        # TODO: Figure out best way to subsample the video with CLIP (in the paper they just use one single frame)
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

    def get_similarity(self, text_embed, video_embed):
        """
        Similarity score between text and video embeddings
        :param text_embed:
        :param video_embed:
        :return: Float where higher number --> more similar
        """
        # TODO: Move this to new file that has similarity metrics
        # Code modified from: https://github.com/openai/CLIP/blob/main/clip/model.py
        # Uses cosine similarity (equivalent to dot product of normalized vectors)

        # normalize features + dot product
        video_features = video_embed / video_embed.norm(dim=1, keepdim=True)
        text_features = text_embed / text_embed.norm(dim=1, keepdim=True)
        similarity_per_video = video_features @ text_features.t()

        return similarity_per_video
