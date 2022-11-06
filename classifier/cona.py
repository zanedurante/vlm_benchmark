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
Implementation of our algorithm: Context Name Optimization.
'''
class CoNaFewShotClassifier(FewShotClassifier):
    def __init__(self, vlm: SimilarityVLM, context_len: int = 16, 
                 opt_strat: str = "joint", name_regularization: float = 1e-2,
                 lr: float = 1e-3, epochs: int = 10,
                 warmup_lr: float = 1e-5, warmup_epochs: int = 1, 
                 batch_size: int = 1, random_augment: bool = True):
        self.vlm = vlm
        
        self.context_len = int(context_len)
        self.opt_strat = str(opt_strat)
        self.name_regularization = float(name_regularization)
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.warmup_lr = float(warmup_lr)
        self.warmup_epochs = float(warmup_epochs)
        self.batch_size = int(batch_size)
        self.random_augment = bool(random_augment)
        
        assert opt_strat in ["joint"], "Invalid optimization strategy."
        
    '''
    Returns a dict with the value of all classifier-specific parameters which may affect prediction
    accuracy (apart from the underlying VLM object used).
    This is used to differentiate test results which use different classifier parameters.
    '''
    def params(self) -> dict:
        return {
            "context_len": self.context_len,
            "opt_strat": self.opt_strat,
            "name_regularization": self.name_regularization,
            "lr": self.lr,
            "epochs": self.epochs,
            "warmup_lr": self.warmup_lr,
            "warmup_epochs": self.warmup_epochs,
            "batch_size": self.batch_size,
            "random_augment": self.random_augment
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
            
            query_to_text_similarities = self.vlm.default_similarity_metric()(query_vid_embeds, text_embeds)
            query_predictions = np.argmax(query_to_text_similarities, axis=1)
            return query_predictions
        
        
        
        train_dataloader = torch.utils.data.DataLoader(
            list(zip(support_video_paths.flatten(), torch.repeat_interleave(torch.arange(n_way), n_support))),
            batch_size=self.batch_size, num_workers=0, shuffle=True
        )
        
        cona_module = CoNaModule(self.vlm, category_names, self.context_len)
        cona_module.to(DEVICE)
        
        optimizer = torch.optim.SGD(cona_module.parameters(), lr=self.lr)
        
        # Constant warmup lr until we begin a cosine lr decay schedule
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            [
                torch.optim.lr_scheduler.ConstantLR(
                    optimizer,
                    self.warmup_lr / self.lr,
                    total_iters=self.warmup_epochs
                ),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=self.epochs - self.warmup_epochs
                )
            ],
            [self.warmup_epochs]
        )
        
        for epoch_idx in range(self.epochs):
            total_loss = 0
            total_reg_loss = 0
            total_correct = 0
            total_count = 0
            
            for batch_idx, (vid_paths, vid_labels) in enumerate(train_dataloader):
                if self.random_augment:
                    vid_embeds = torch.cat([
                        torch.from_numpy(self.vlm.video_encoder(vid_path, random_augment=True)).unsqueeze(0).to(DEVICE)
                        for vid_path in vid_paths
                    ], dim=0)
                else: # Use version of video encoder which can cache results for fast lookup
                    vid_embeds = torch.cat([
                        torch.from_numpy(self.vlm.get_video_embeds(vid_path)).unsqueeze(0).to(DEVICE)
                        for vid_path in vid_paths
                    ], dim=0)
                vid_labels = vid_labels.to(DEVICE)
                
                logits = cona_module(vid_embeds)
                loss = F.cross_entropy(logits, vid_labels)
                reg_loss = self.name_regularization * cona_module.name_perturbation.pow(2).mean()
                combined_loss = loss + reg_loss
                
                total_loss += loss.item() * len(vid_paths)
                total_reg_loss += reg_loss.item() * len(vid_paths)
                total_correct += (logits.argmax(dim=1) == vid_labels).sum()
                total_count += len(vid_paths)
                
                optimizer.zero_grad()
                combined_loss.backward()
                optimizer.step()
            scheduler.step()
                
            print(f"Epoch {epoch_idx:5}: Acc = {total_correct / total_count:5.3f}, Loss = {total_loss / total_count:5.3E}, Reg Loss = {total_reg_loss / total_count:5.3E}")
                
                
                
        query_dataloader = torch.utils.data.DataLoader(query_video_paths, batch_size=self.batch_size, num_workers=0, shuffle=False)
        with torch.no_grad():
            query_predictions = []
            for batch_idx, vid_paths in enumerate(query_dataloader):
                batch_query_vid_embeds = torch.cat([
                    torch.from_numpy(self.vlm.get_video_embeds(vid_path)).unsqueeze(0).to(DEVICE)
                    for vid_path in vid_paths
                ])
                batch_query_logits = cona_module(batch_query_vid_embeds)
                query_predictions.append(torch.argmax(batch_query_logits, dim=1))
            query_predictions = torch.cat(query_predictions, dim=0)
            return query_predictions.cpu().numpy()
        
                
        
        
class CoNaModule(nn.Module):
    def __init__(self, vlm: SimilarityVLM, category_names: np.ndarray, context_len: int = 16):
        super().__init__()
        
        self.vlm = vlm
        self.category_names = category_names
        
        # Hyperparameters
        self.context_len = context_len
        
        # Orig Name embeddings
        with torch.no_grad():
            category_name_input_embeds, category_name_attn_masks = vlm.get_input_word_embeddings(category_names.tolist())
        self.register_buffer("category_name_input_embeds", category_name_input_embeds)
        self.register_buffer("category_name_attn_masks", category_name_attn_masks)
        
        # Class-shared context embeddings
        context = torch.empty(1, self.context_len, vlm.input_word_embed_dim())
        nn.init.normal_(context, std=0.02)
        self.context = nn.Parameter(context)
        
        # Class-specific name embedding tweaks
        name_perturbation = torch.empty_like(category_name_input_embeds)
        nn.init.normal_(name_perturbation, std=0.02)
        self.name_perturbation = nn.Parameter(name_perturbation)
        
        # Mask for class embeddings which aren't special tokens
        name_token_mask = category_name_attn_masks.clone().type(torch.bool)
        name_token_mask[:, :self.vlm.text_start_special_token_count()] = False
        name_token_mask[:, -self.vlm.text_end_special_token_count():] = False
        self.register_buffer("name_token_mask", name_token_mask)
        
        
        
    def forward(self, vid_embeds: torch.Tensor) -> torch.Tensor:
        # Retain only name perturbations which correspond to actual words (not special tokens or padding)
        masked_name_perturbation = self.name_perturbation * self.name_token_mask.unsqueeze(2)
        perturbed_category_name_input_embeds = self.category_name_input_embeds + masked_name_perturbation
        
        text_input_embeds = torch.cat(
            [
                perturbed_category_name_input_embeds[:, :self.vlm.text_start_special_token_count(), :],
                self.context.expand(self.category_name_input_embeds.size(0), -1, -1),
                perturbed_category_name_input_embeds[:, self.vlm.text_start_special_token_count():, :]
            ], dim=1
        )
        text_input_attn_masks = torch.cat(
            [
                self.category_name_attn_masks[:, :self.vlm.text_start_special_token_count()],
                torch.ones(self.category_name_input_embeds.size(0), self.context_len, device=DEVICE),
                self.category_name_attn_masks[:, self.vlm.text_start_special_token_count():]
            ], dim=1
        )
        
        text_embeds = self.vlm.text_encoder_from_word_embeddings(text_input_embeds, text_input_attn_masks)
        
        if self.vlm.default_similarity_metric() == Similarity.COSINE:
            vid_embeds = F.normalize(vid_embeds, dim=1)
            text_embeds = F.normalize(text_embeds, dim=1)
            logits = self.vlm.logit_scale() * (vid_embeds @ text_embeds.T)
            
        elif self.vlm.default_similarity_metric() == Similarity.DOT:
            logits = self.vlm.logit_scale() * (vid_embeds @ text_embeds.T)
            
        else:
            raise NotImplementedError
        
        return logits
        