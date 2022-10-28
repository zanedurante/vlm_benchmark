from typing import Optional, Tuple
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from SimilarityVLM import SimilarityVLM
from similarity_metrics import Similarity
from .base import FewShotClassifier

'''
Implementation of Visual Prompt Tuning for our framework.
Similar to adding extra prompt vision tokens (https://arxiv.org/abs/2203.12119), 
we instead learn spatial and temporal position embeddings to add to the default
video tokens.
'''
class VisualPromptFewShotClassifier(FewShotClassifier):
    def __init__(self, vlm: SimilarityVLM, epochs: int = 100, lr: float = 1e-3, weight_decay: float = 1e-2):
        self.vlm = vlm
        
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        
    '''
    Returns a dict with the value of all classifier-specific parameters which may affect prediction
    accuracy (apart from the underlying VLM object used).
    This is used to differentiate test results which use different classifier parameters.
    '''
    def params(self) -> dict:
        return {
            "epochs": self.epochs,
            "lr": self.lr,
            "weight_decay": self.weight_decay
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
        n_way = len(category_names)
        n_predict = query_video_paths.shape[0]
        if support_video_paths is None:
            n_support = 0
        else:
            n_support = support_video_paths.shape[1]
        
        # Use default similarity to text embeds if zero-shot
        if n_support == 0:
            # Text Embeddings
            text_embeds = np.array([self.vlm.get_text_embeds(name) for name in category_names])
            
            # Query Vid Embeddings
            query_vid_embeds = np.array([self.vlm.get_video_embeds(vid_path) for vid_path in query_video_paths])
            
            # Similarity
            query_to_text_similarities = self.vlm.default_similarity_metric()(query_vid_embeds, text_embeds)
            query_predictions = np.argmax(query_to_text_similarities, axis=1)
            return query_predictions
        
        # Load all class text embeddings as matrix/transform from vid embedding to similarity per class
        class_text_embeds = torch.vstack([
            torch.from_numpy(self.vlm.get_text_embeds(name))
            for name in category_names
        ])
        
        # Load all support videos as video token tensors
        support_labels = torch.repeat_interleave(torch.arange(n_way), n_support)
        with torch.no_grad():
            support_inputs = torch.concat([
                self.vlm.video_encoder_to_tokens(vid_path)
                for vid_paths_per_class in support_video_paths for vid_path in vid_paths_per_class
            ])
        support_dataset = torch.utils.data.TensorDataset(support_inputs, support_labels)
        support_dataloader = torch.utils.data.DataLoader(support_dataset, batch_size=32, num_workers=8, shuffle=True)

        # Create module and setup training params
        prompt_module = VisionPromptModule(self.vlm, class_text_embeds, )
        prompt_module.to(DEVICE)
        print(list(prompt_module.parameters()))
        
        optimizer = torch.optim.AdamW(prompt_module.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        # Train visual prompt
        for epoch_idx in range(self.epochs):
            total_queries = 0
            total_correct = 0
            total_loss = 0
            
            for batch_idx, (vid_embeds, vid_labels) in enumerate(support_dataloader):
                logits = prompt_module(vid_embeds)
                loss = F.cross_entropy(logits, vid_labels)
                
                total_queries += len(logits)
                total_correct += (torch.argmax(logits, dim=-1) == vid_labels).sum()
                total_loss += loss * len(logits)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            print(f"Acc: {total_correct / total_queries:20.4f}, Loss: {total_loss / total_queries:20.4f}")
                
        # Predict for query videos
        with torch.no_grad():
            query_inputs = torch.concat([
                self.vlm.video_encoder_to_tokens(vid_path)
                for vid_path in query_video_paths
            ])
            query_logits = prompt_module(query_inputs)
        query_predictions = torch.argmax(query_logits, dim=1).cpu().numpy()
        return query_predictions
          
        
        
class VisionPromptModule(nn.Module):
    def __init__(self, vlm: SimilarityVLM, text_embeds: torch.Tensor):
        super().__init__()
        
        self.vlm = vlm
        
        # n_way x embed_dim
        self.register_buffer(
            "text_embeds",
            text_embeds / torch.norm(text_embeds, dim=-1, keepdim=True)
        )
        
        self.constant = nn.Parameter(torch.zeros(1, 1, 1, vlm.video_token_dim()))
        self.temporal = nn.Parameter(torch.zeros(1, vlm.video_num_frames(), 1, vlm.video_token_dim()))
        self.positional = nn.Parameter(torch.zeros(1, 1, vlm.video_num_frame_patches(), vlm.video_token_dim()))
        

        
    def forward(self, video_tokens: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            video_tokens (torch.Tensor): Shape (batch, frames, patches, token_dim)

        Returns:
            torch.Tensor: video embeddings, shape (batch, embed_dim)
        """
        # Add positional/temporal embedding perturbations
        video_tokens += self.constant + self.temporal + self.positional
        
        # Compute embeddings
        video_embeds = self.vlm.video_encoder_from_tokens(video_tokens)
        
        # Compute class prediction logits
        video_embeds = F.normalize(video_embeds, dim=-1)
        class_logits = video_embeds @ self.text_embeds.T
        
        return class_logits