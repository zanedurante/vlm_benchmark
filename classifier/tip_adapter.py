from typing import Optional
import numpy as np
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from SimilarityVLM import SimilarityVLM
from similarity_metrics import Similarity
from .base import FewShotClassifier
from .prompt_ensembles import PROMPT_ENSEMBLES

QUERY_BATCH_SIZE = 2048 # Batch size used for iterating through non-training data

'''
Implementation of Tip-Adapter (https://arxiv.org/abs/2111.03930) for our framework.
'''
class TipAdapterFewShotClassifier(FewShotClassifier):
    def __init__(self, vlm: SimilarityVLM, alpha: float, beta: float,
                 finetune_epochs: int = 0, finetune_lr: float = 1e-3,
                 batch_size: int = 256, random_augment: bool = True,
                 prompt_ensemble_id: Optional[str] = None):
        self.vlm = vlm
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.finetune_epochs = int(finetune_epochs)
        self.finetune_lr = float(finetune_lr)
        self.batch_size = int(batch_size)
        self.random_augment = bool(random_augment)
        self.prompt_ensemble_id = str(prompt_ensemble_id)
        
        assert prompt_ensemble_id in PROMPT_ENSEMBLES.keys(), "Unrecognized prompt_ensemble_id."
        
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
            "batch_size": self.batch_size,
            "random_augment": self.random_augment,
            "prompt_ensemble_id": self.prompt_ensemble_id
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
        n_way = len(category_names)
        n_predict = query_video_paths.shape[0]
        if support_video_paths is None:
            n_support = 0
        else:
            n_support = support_video_paths.shape[1]
        
        # Text Embeddings
        text_embeds = np.array([
            [
                self.vlm.get_text_embeds(template.format(name))
                for template in PROMPT_ENSEMBLES[self.prompt_ensemble_id]
            ]
            for name in category_names
        ]).mean(axis=1)
        
        # Query Vid Embeddings
        query_vid_embeds = np.array([self.vlm.get_video_embeds(vid_path) for vid_path in query_video_paths])
        
        # Use default similarity to text embeds if zero-shot
        if n_support == 0:
            query_to_text_similarities = self.vlm.default_similarity_metric()(query_vid_embeds, text_embeds)
            query_predictions = np.argmax(query_to_text_similarities, axis=1)
            return query_predictions
        
        
        
        # Support Vid Embeddings
        # TipAdapter averages each support embedding over 10 augments of the image
        if self.random_augment:
            with torch.no_grad():
                flat_support_vid_embeds = np.array([
                    [
                        self.vlm.video_encoder(vid_path, random_augment=True)
                        for _ in range(10)
                    ]
                    for vid_path in support_video_paths.flatten()
                ]).mean(axis=1)
        else:
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
        adapter_module = TipAdapterModule(text_embeds, flat_support_vid_embeds, flat_support_vid_labels, self.alpha, self.beta, self.vlm.logit_scale())
        adapter_module.to(DEVICE)
        
        if self.finetune_epochs > 0:
            # Copy support vid embeddings as a training dataset for finetuning
            train_dataset = torch.utils.data.TensorDataset(flat_support_vid_embeds, flat_support_vid_labels)
            train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, num_workers=0, shuffle=True)
            
            # Setup embeds for Val-Tuning Dataset
            if val_tuning_video_paths is None or val_tuning_video_labels is None:
                val_tuning_dataloader = None
            else:
                val_tuning_vid_embeds = torch.from_numpy(np.array([self.vlm.get_video_embeds(vid_path) for vid_path in val_tuning_video_paths]))
                val_tuning_vid_embeds = F.normalize(val_tuning_vid_embeds, dim=-1)
                val_tuning_vid_labels = torch.from_numpy(val_tuning_video_labels)
                val_tuning_dataset = torch.utils.data.TensorDataset(val_tuning_vid_embeds, val_tuning_vid_labels)
                val_tuning_dataloader = torch.utils.data.DataLoader(
                    val_tuning_dataset,
                    batch_size=QUERY_BATCH_SIZE, num_workers=0, shuffle=False
                )
            
            optimizer = torch.optim.AdamW(adapter_module.parameters(), lr=self.finetune_lr, eps=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.finetune_epochs * len(train_dataloader)) # Tip Adapter does scheduling steps with each optimizer step
            
            # Save best-performing model (if val tuning dataset is provided)
            val_tuning_best_acc = None
            val_tuning_best_model_state = None
            for epoch_idx in range(self.finetune_epochs):
                total_loss = 0
                total_correct = 0
                total_count = 0
                
                for batch_idx, (vid_embeds, vid_labels) in enumerate(train_dataloader):
                    vid_embeds = vid_embeds.to(DEVICE)
                    vid_labels = vid_labels.to(DEVICE)
                    
                    logits = adapter_module(vid_embeds)
                    loss = F.cross_entropy(logits, vid_labels)
                    
                    total_loss += loss.item() * len(vid_embeds)
                    total_correct += (logits.argmax(dim=1) == vid_labels).sum()
                    total_count += len(vid_embeds)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    
                # Check val-tuning performance
                # Check val-tuning performance
                if val_tuning_dataloader is not None:
                    total_val_correct = 0
                    total_val_count = 0
                    for batch_idx, (vid_embeds, vid_labels) in enumerate(val_tuning_dataloader):
                        vid_embeds = vid_embeds.to(DEVICE)
                        vid_labels = vid_labels.to(DEVICE)
                        
                        with torch.no_grad():
                            logits = adapter_module(vid_embeds)
                        total_val_correct += (logits.argmax(dim=1) == vid_labels).sum()
                        total_val_count += len(vid_embeds)
                        
                    val_acc = total_val_correct / total_val_count
                    if val_tuning_best_acc is None or val_acc >= val_tuning_best_acc:
                        val_tuning_best_acc = val_acc
                        val_tuning_best_model_state = deepcopy(adapter_module.state_dict())
                    print(f"Epoch {epoch_idx:5}: Support Acc = {total_correct / total_count:5.3f}, Val-Tune Acc = {val_acc:5.3f}, Loss = {total_loss / total_count:5.3f}")
                else:
                    print(f"Epoch {epoch_idx:5}: Support Acc = {total_correct / total_count:5.3f}, Loss = {total_loss / total_count:5.3f}")
                       
            # Reload best val-tuning model state
            if val_tuning_best_model_state is not None:
                adapter_module.load_state_dict(val_tuning_best_model_state)
            
            
                    
        query_embed_dataloader = torch.utils.data.DataLoader(query_vid_embeds, batch_size=QUERY_BATCH_SIZE, num_workers=0, shuffle=False)
        query_predictions = []
        with torch.no_grad():
            for batch_idx, vid_embeds in enumerate(query_embed_dataloader):
                vid_embeds = vid_embeds.to(DEVICE)
                logits = adapter_module(vid_embeds)
                query_predictions.append(logits.argmax(dim=1))
            query_predictions = torch.cat(query_predictions, dim=0)
        return query_predictions.cpu().numpy()
        
                
        
        
class TipAdapterModule(nn.Module):
    def __init__(self, text_embeds: torch.Tensor,
                 flat_support_vid_embeds: torch.Tensor, flat_support_vid_labels: torch.Tensor,
                 alpha: float, beta: float, vlm_logit_scale: float):
        super().__init__()
        
        # Hyperparameters
        self.alpha = alpha
        self.beta = beta
        
        # VLM-specific value
        self.vlm_logit_scale = vlm_logit_scale
        
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
        
        text_logits = self.vlm_logit_scale * embeds @ self.text_embeds.T
        
        tip_logits = text_logits + self.alpha * cache_logits
        
        return tip_logits