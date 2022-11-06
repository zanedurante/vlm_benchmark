from typing import Optional
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from SimilarityVLM import SimilarityVLM
from similarity_metrics import Similarity
from .base import FewShotClassifier

'''
Implementation of Tip-Adapter (https://arxiv.org/abs/2111.03930) for our framework.
'''
class TipAdapterFewShotClassifier(FewShotClassifier):
    def __init__(self, vlm: SimilarityVLM, alpha: float, beta: float, finetune_epochs: int = 0, finetune_lr: float = 1e-3, weight_decay: float = 1e-2):
        self.vlm = vlm
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.finetune_epochs = int(finetune_epochs)
        self.finetune_lr = float(finetune_lr)
        self.weight_decay = float(weight_decay)
        
        if self.finetune_epochs == 0:
            self.finetune_lr = 0.0
        
    '''
    Returns a dict with the value of all classifier-specific parameters which may affect prediction
    accuracy (apart from the underlying VLM object used).
    This is used to differentiate test results which use different classifier parameters.
    '''
    def params(self) -> dict:
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "finetune_epochs": self.finetune_epochs,
            "finetune_lr": self.finetune_lr,
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
        
        # Text Embeddings
        text_embeds = np.array([self.vlm.get_text_embeds(name) for name in category_names])
        
        # Query Vid Embeddings
        query_vid_embeds = np.array([self.vlm.get_video_embeds(vid_path) for vid_path in query_video_paths])
        
        # Use default similarity to text embeds if zero-shot
        if n_support == 0:
            query_to_text_similarities = self.vlm.default_similarity_metric()(query_vid_embeds, text_embeds)
            query_predictions = np.argmax(query_to_text_similarities, axis=1)
            return query_predictions
        
        
        
        # Support Vid Embeddings
        # TODO: TipAdapter averages each support embedding over 10 augments of the image (random recrop, 50% horizontal flip). Is this viable for videos?
        flat_support_vid_embeds = np.array([self.vlm.get_video_embeds(vid_path) for vid_path in support_video_paths.flatten()])
        
        # Normalize all embeddings so we can use both dot-product and euclid distance as Tip-Adapter does
        text_embeds /= np.linalg.norm(text_embeds, axis=-1, keepdims=True)
        query_vid_embeds /= np.linalg.norm(query_vid_embeds, axis=-1, keepdims=True)
        flat_support_vid_embeds /= np.linalg.norm(flat_support_vid_embeds, axis=-1, keepdims=True)
        
        text_embeds = torch.from_numpy(text_embeds)
        query_vid_embeds = torch.from_numpy(query_vid_embeds)
        flat_support_vid_embeds = torch.from_numpy(flat_support_vid_embeds)
        flat_support_vid_labels = torch.repeat_interleave(torch.arange(n_way), n_support)
        
        # Torch module for tip adapter
        adapter_module = TipAdapterModule(text_embeds, flat_support_vid_embeds, flat_support_vid_labels, self.alpha, self.beta)
        adapter_module.to(DEVICE)
        
        if self.finetune_epochs > 0:
            # Copy support vid embeddings as a training dataset for finetuning
            train_dataset = torch.utils.data.TensorDataset(flat_support_vid_embeds, flat_support_vid_labels)
            train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=256, num_workers=8, shuffle=True)
            
            optimizer = torch.optim.AdamW(adapter_module.parameters(), lr=self.finetune_lr, eps=1e-4, weight_decay=self.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.finetune_epochs * len(train_dataloader))
            
            for epoch_idx in range(self.finetune_epochs):
                for batch_idx, (vid_embeds, vid_labels) in enumerate(train_dataloader):
                    vid_embeds = vid_embeds.to(DEVICE)
                    vid_labels = vid_labels.to(DEVICE)
                    
                    logits = adapter_module(vid_embeds)
                    loss = F.cross_entropy(logits, vid_labels)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    
        query_vid_embeds = query_vid_embeds.to(DEVICE)
        with torch.no_grad():
            query_logits = adapter_module(query_vid_embeds).cpu().numpy()
        query_predictions = np.argmax(query_logits, axis=1)
        return query_predictions
        
                
        
        
class TipAdapterModule(nn.Module):
    def __init__(self, text_embeds: torch.tensor,
                 flat_support_vid_embeds: torch.tensor, flat_support_vid_labels: torch.tensor,
                 alpha: float, beta: float):
        super().__init__()
        
        # Hyperparameters
        self.alpha = alpha
        self.beta = beta
        
        # "Cache keys", computes affinity between query video embedding and support video embeddings
        self.cache_keys = nn.Parameter(flat_support_vid_embeds)
        
        # "Cache values", records the true class for each support video
        self.register_buffer(
            "cache_values",
            F.one_hot(flat_support_vid_labels, num_classes=text_embeds.shape[0]).float()
        )
        
        # "CLIP weights", for computing the similarity between query video embedding and class text embeddings
        # to produce the logits that would be applicable for zero-shot classification
        self.register_buffer(
            "text_embeds",
            text_embeds
        )
        
    def forward(self, embeds: torch.tensor) -> torch.tensor:
        affinity = embeds @ self.cache_keys.T
        cache_logits = (-1 * self.beta * (1 - affinity)).exp() @ self.cache_values
        
        text_logits = 100 * embeds @ self.text_embeds.T
        
        tip_logits = text_logits + self.alpha * cache_logits
        
        return tip_logits