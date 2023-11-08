import numpy as np
import random
import os
from typing import Optional, List

import torch
import decord
from transformers import CLIPModel, CLIPTokenizer, CLIPProcessor
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from SimilarityVLM import SimilarityVLM
from similarity_metrics import Similarity



# Default cache locations
FILE_DIR = os.path.dirname(os.path.realpath(__file__))
CACHE_NAME = "cache"



class ClipVLM(SimilarityVLM):
    """
    Similarity-based VLM that uses CLIP for frame and text encoders.  Currently, we use the hugging face implementation
    for CLIP since it is easier to set up.
    TODO: Implement the larger version of CLIP since this should get better performance.
    """

    def __init__(self, path="openai/clip-vit-base-patch32", num_frames=1, sample_strat='uniform',
                 reset_cache=False):
        self.model = None
        self.tokenizer = None
        self.processor = None
        
        self.path = str(path)  # Pretrained CLIP identifier
        self.num_frames = int(num_frames)  # The number of frames CLIP will use to classify videos
        self.sample_strat = str(sample_strat)  # 'rand' or 'uniform'

        decord.bridge.set_bridge("torch")  # Video loader
        
        # Load model
        self.model = CLIPModel.from_pretrained(path)
        self.tokenizer = CLIPTokenizer.from_pretrained(path)
        self.processor = CLIPProcessor.from_pretrained(path)
        self.model.to(DEVICE)
        
        super().__init__(cache_file=os.path.join(FILE_DIR, CACHE_NAME), reset_cache=reset_cache)
        
    def params(self) -> dict:
        """
        Specify the value of all VLM-specific parameters which may affect prediction accuracy.
        This is used to differentiate test results which use different versions of the same VLM.
        :return:
        :rtype: dict
        """
        return {
            "path": self.path,
            "num_frames": self.num_frames,
            "sample_strat": self.sample_strat
        }
        
    def logit_scale(self) -> float:
        return 100
    
    def output_embed_dim(self) -> int:
        return 512
        
    def input_word_embed_dim(self) -> int:
        return 512
    
    def text_start_special_token_count(self) -> int:
        return 1
    
    def text_end_special_token_count(self) -> int:
        return 1
    
    def text_encoder(self, text):
        """
        Tokenize and encode text into a joint text/video embedding space
        :param tokens:
        :return:
        """
        tokens = self.tokenizer(text, padding=True, return_tensors="pt", max_length=77, truncation=True)
        for k,v in tokens.items():
            tokens[k] = v.to(DEVICE)
        with torch.no_grad():
            text_features = self.model.get_text_features(**tokens).cpu().numpy()[0]
        return text_features

    def get_input_word_embeddings(self, text_list: List[str]) -> torch.Tensor:
        """Converts a list of text string into a batched tensor of input word embeddings and a corresponding attention mask,
        including special tokens.

        Args:
            text_list (str): _description_

        Returns:
            torch.Tensor: input token embeddings for the text encoder. Shape (batch, sequence_len, token_dim)
            torch.Tensor: input sequence attention mask for the text encoder. Shape (batch, sequence_len)
        """
        text_input = self.tokenizer(text_list, padding=True, return_tensors="pt", max_length=77, truncation=True)
        token_ids = text_input["input_ids"].to(DEVICE)
        attn_mask = text_input["attention_mask"].to(DEVICE)
            
        text_input_embeds = self.model.text_model.embeddings.token_embedding(token_ids)
        
        return text_input_embeds, attn_mask

    def text_encoder_from_word_embeddings(self, input_word_embeds: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        """Converts a batch of token embeddings and corresponding attention masks into a batch of text embeddings.

        Args:
            token_embeds (torch.Tensor): Shape (batch, sequence_len, token_dim)
            attn_mask (torch.Tensor): Shape (batch, sequence_len)

        Returns:
            torch.Tensor: Shape (batch, embed_dim)
        """
        # Huggingface ClipTextTransformer doesn't allow inputs_embed argument
        # So we have to manually implement what the text transformer does
        hidden_states = self.model.text_model.embeddings(inputs_embeds=input_word_embeds)
        
        bsz, seq_len = input_word_embeds.shape[:2]
        
        # CLIP's text model uses causal mask, prepare it here.
        # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
        #causal_attention_mask = self.model.text_model._build_causal_attention_mask(bsz, seq_len, hidden_states.dtype).to(DEVICE)
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        causal_attention_mask = torch.empty(bsz, seq_len, seq_len, dtype=hidden_states.dtype)
        causal_attention_mask.fill_(torch.tensor(torch.finfo(hidden_states.dtype).min))
        causal_attention_mask.triu_(1)  # zero out the lower diagonal
        causal_attention_mask = causal_attention_mask.unsqueeze(1)  # expand mask
        causal_attention_mask = causal_attention_mask.to(DEVICE)
        
        # expand attention_mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        attention_mask = _expand_mask(attn_mask, hidden_states.dtype)
        
        encoder_outputs = self.model.text_model.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
        )
        
        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.model.text_model.final_layer_norm(last_hidden_state)
        
        # text_embeds.shape = [batch_size, sequence_length, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
        
        # New: Since we don't have access to token_ids, we will use the location of the final nonzero element
        # in the attention mask instead
        attn_mask_positions = attn_mask * torch.arange(seq_len, device=DEVICE).unsqueeze(0).expand(bsz, seq_len)
        final_nonzero_indices = attn_mask_positions.argmax(dim=-1)
        pooled_output = last_hidden_state[
            torch.arange(last_hidden_state.shape[0], device=DEVICE), final_nonzero_indices
        ]
        
        # ClipTextEncoder returns a lot of info, but ClipModel.get_text_features() only uses pooled_output
        text_features = self.model.text_projection(pooled_output)
        
        return text_features
        
    def text_encoder_over_embeds(self, text):
        with torch.no_grad():
            input_word_embeds, attn_mask = self.get_input_word_embeddings([text])
            return self.text_encoder_from_word_embeddings(input_word_embeds, attn_mask)[0].cpu().numpy()

    def video_encoder(self, video_path: str, subvideo_start_frame: Optional[int] = None, subvideo_end_frame: Optional[int] = None, random_augment: bool = False) -> np.ndarray:
        """
        Load, transform and encode a video file into a joint text/video embedding space
        :param video:
        :param subvideo_start_frame:
        :param subvideo_end_frame:
        :return:
        """
        # Load
        # TODO: Figure out best way to subsample the video with CLIP (in the paper they just use one single frame)
        video_reader = decord.VideoReader(video_path, num_threads=1)
        video_len = len(video_reader)
        frame_indices = self.sample_frame_indices(video_len, subvideo_start_frame, subvideo_end_frame, random_augment)
        frames = video_reader.get_batch(frame_indices)
        frames = frames.float() / 255
        frames = frames.permute(0, 3, 1, 2)
        # Convert frame batch axis into list
        frames = [frame for frame in frames]

        # Preprocess
        inputs = self.processor(images=frames, return_tensors="pt")
        
        for k, v in inputs.items():
            inputs[k] = v.to(DEVICE)
        
        # Encode
        with torch.no_grad():
            video_features = self.model.get_image_features(**inputs)  # Frame-level video features
            video_features = video_features.mean(dim=0) # Average over sampled frames from this video
            video_features = video_features.cpu().numpy()

        return video_features

    def default_similarity_metric(self) -> Similarity:
        """
        Returns a reference to the default similarity metric used by this VLM
        :return:
        """
        return Similarity.COSINE

    def sample_frame_indices(self, video_len: int, subvideo_start_frame: Optional[int] = None, subvideo_end_frame: Optional[int] = None, random_augment: bool = False) -> np.ndarray:
        start_frame = subvideo_start_frame or 0
        end_frame = subvideo_end_frame or video_len
        
        frame_indices = np.linspace(start_frame, end_frame, self.num_frames, endpoint=False)
        
        if self.sample_strat == "rand" or random_augment:
            frame_indices += np.random.choice((end_frame - start_frame) // self.num_frames)
        elif self.sample_strat == "uniform":
            frame_indices += (end_frame - start_frame) / self.num_frames / 2
        else:
            raise ValueError
        
        frame_indices = np.minimum(
            np.round(frame_indices),
            end_frame - 1
        )
        
        return frame_indices



    def get_feature_shape(self):
        return 512

    # Can this even be different from above?
    def get_text_token_dim(self):
        return 512

    def get_max_text_tokens(self):
        return 77


    
    
# Helper Function from Huggingface Clip Implementation
# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)
