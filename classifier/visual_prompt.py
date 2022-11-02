from typing import Optional, Tuple
import numpy as np
import math

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
class VisualPromptAdditionFewShotClassifier(FewShotClassifier):
    def __init__(self, vlm: SimilarityVLM, epochs: int = 50, warmup_epochs: int = 10,
                 lr: float = 1e-3, weight_decay: float = 1e-2, momentum: float = 0.9, temperature: float = 1e-2,
                 random_augment: bool = True, batch_size: int = 16):
        """
        Args:
            vlm (SimilarityVLM):            Base VLM
            epochs (int, optional):         _description_. Defaults to 50.
            warmup_epochs (int, optional):  _description_. Defaults to 10.
            lr (float, optional):           _description_. Defaults to 1e-3.
            weight_decay (float, optional): _description_. Defaults to 1e-2.
            momentum (float, optional):     Momentum hyperparameter for SGD optimizer. Defaults to 0.9.
            temperature (float, optional):  Describes how sharply differences in similarities are judged. Higher temperature = higher entropy outputs.
                                            Generally set specifically to each VLM. Defaults to 1e-2 (CLIP).
        """
        self.vlm = vlm
        
        self.epochs = int(epochs)
        self.warmup_epochs = int(warmup_epochs)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.momentum = float(momentum)
        self.temperature = float(temperature)
        self.random_augment = bool(random_augment)
        self.batch_size = int(batch_size)
        
    '''
    Returns a dict with the value of all classifier-specific parameters which may affect prediction
    accuracy (apart from the underlying VLM object used).
    This is used to differentiate test results which use different classifier parameters.
    '''
    def params(self) -> dict:
        return {
            "epochs": self.epochs,
            "warmup_epochs": self.warmup_epochs,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "momentum": self.momentum,
            "temperature": self.temperature,
            "random_augment": self.random_augment,
            "batch_size": self.batch_size
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
        
        # Create support dataset of video paths (cannot preload as video tokens since they may use random augmentation)
        #support_dataset = torch.utils.data.TensorDataset(torch.from_numpy(support_video_paths.flatten()), torch.repeat_interleave(torch.arange(n_way), n_support))
        support_dataloader = torch.utils.data.DataLoader(
            list(zip(support_video_paths.flatten(), torch.repeat_interleave(torch.arange(n_way), n_support))),
            batch_size=self.batch_size, num_workers=0, shuffle=True
        )

        # Create module and setup training params
        prompt_module = VisionPromptAdditionModule(self.vlm, class_text_embeds)
        prompt_module.to(DEVICE)
        
        optimizer = torch.optim.SGD(prompt_module.parameters(), lr=self.lr, weight_decay=self.weight_decay, momentum=self.momentum)
        scheduler = WarmupCosineSchedule(optimizer, self.warmup_epochs, self.epochs)
        
        # Train visual prompt
        for epoch_idx in range(self.epochs):
            total_queries = 0
            total_correct = 0
            total_loss = 0
            
            for batch_idx, (vid_paths, vid_labels) in enumerate(support_dataloader):
                with torch.no_grad():
                    vid_embeds = torch.concat([
                        self.vlm.video_encoder_to_tokens(vid_path, random_augment=self.random_augment)
                        for vid_path in vid_paths
                    ])
                    
                vid_embeds = vid_embeds.to(DEVICE)
                vid_labels = vid_labels.to(DEVICE)
                
                logits = prompt_module(vid_embeds) / self.temperature
                loss = F.cross_entropy(logits, vid_labels)
                
                total_queries += len(logits)
                total_correct += (torch.argmax(logits, dim=-1) == vid_labels).sum()
                total_loss += loss * len(logits)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            scheduler.step()
                
            if epoch_idx % 5 == 0:
                print(f"Epoch: {epoch_idx:5}, Acc: {total_correct / total_queries:10.4f}, Loss: {total_loss / total_queries:10.4f}")
                
        # Predict for query videos
        with torch.no_grad():
            query_inputs = torch.concat([
                self.vlm.video_encoder_to_tokens(vid_path, random_augment=False).cpu()
                for vid_path in query_video_paths
            ])
            
            query_predictions = []
            for query_input_batch in torch.utils.data.DataLoader(query_inputs, batch_size=self.batch_size, num_workers=0):
                query_input_batch = query_input_batch.to(DEVICE)
                query_logits_batch = prompt_module(query_input_batch)
                query_predictions_batch = torch.argmax(query_logits_batch, dim=1).cpu()
                query_predictions.append(query_predictions_batch)
            query_predictions = torch.concat(query_predictions).numpy()
            
        return query_predictions
        
        
class VisionPromptAdditionModule(nn.Module):
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
    
    
    
    
    
    
    
    
    
    
'''
Implementation of Visual Prompt Tuning for our framework.
Adds extra prompt vision tokens (https://arxiv.org/abs/2203.12119)
before video tokens.
'''
class VisualPromptPrependFewShotClassifier(FewShotClassifier):
    def __init__(self, vlm: SimilarityVLM, prompt_token_count: int, epochs: int = 50, warmup_epochs: int = 10,
                 lr: float = 1e-3, weight_decay: float = 1e-2, momentum: float = 0.9, temperature: float = 1e-2,
                 random_augment: bool = True, batch_size: int = 16):
        """
        Args:
            vlm (SimilarityVLM):                Base VLM
            prompt_token_count (int):           Number of prompt tokens to learn and prepend to video tokens.
            epochs (int, optional):             _description_. Defaults to 50.
            warmup_epochs (int, optional):      _description_. Defaults to 10.
            lr (float, optional):               _description_. Defaults to 1e-3.
            weight_decay (float, optional):     _description_. Defaults to 1e-2.
            momentum (float, optional):         Momentum hyperparameter for SGD optimizer. Defaults to 0.9.
            temperature (float, optional):      Describes how sharply differences in similarities are judged. Higher temperature = higher entropy outputs.
                                                Generally set specifically to each VLM. Defaults to 1e-2 (CLIP)
            random_augment (bool, optional):    Whether support videos use randomized augmentation when training prompts
        """
        self.vlm = vlm
        
        self.prompt_token_count = int(prompt_token_count)
        self.epochs = int(epochs)
        self.warmup_epochs = int(warmup_epochs)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.momentum = float(momentum)
        self.temperature = float(temperature)
        self.random_augment = bool(random_augment)
        self.batch_size = int(batch_size)
        
    '''
    Returns a dict with the value of all classifier-specific parameters which may affect prediction
    accuracy (apart from the underlying VLM object used).
    This is used to differentiate test results which use different classifier parameters.
    '''
    def params(self) -> dict:
        return {
            "prompt_token_count": self.prompt_token_count,
            "epochs": self.epochs,
            "warmup_epochs": self.warmup_epochs,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "momentum": self.momentum,
            "temperature": self.temperature,
            "random_augment": self.random_augment,
            "batch_size": self.batch_size
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
        
        # Create support dataset of video paths (cannot preload as video tokens since they may use random augmentation)
        #support_dataset = torch.utils.data.TensorDataset(torch.from_numpy(support_video_paths.flatten()), torch.repeat_interleave(torch.arange(n_way), n_support))
        support_dataloader = torch.utils.data.DataLoader(
            list(zip(support_video_paths.flatten(), torch.repeat_interleave(torch.arange(n_way), n_support))),
            batch_size=self.batch_size, num_workers=0, shuffle=True
        )

        # Create module and setup training params
        prompt_module = VisionPromptPrependModule(self.vlm, class_text_embeds, self.prompt_token_count)
        prompt_module.to(DEVICE)
        
        optimizer = torch.optim.SGD(prompt_module.parameters(), lr=self.lr, momentum=self.momentum)
        scheduler = WarmupCosineSchedule(optimizer, self.warmup_epochs, self.epochs)
        
        # Train visual prompt
        for epoch_idx in range(self.epochs):
            total_queries = 0
            total_correct = 0
            total_loss = 0
            
            for batch_idx, (vid_paths, vid_labels) in enumerate(support_dataloader):
                with torch.no_grad():
                    vid_embeds = torch.concat([
                        self.vlm.video_encoder_to_tokens(vid_path, random_augment=self.random_augment)
                        for vid_path in vid_paths
                    ])
                
                vid_embeds = vid_embeds.to(DEVICE)
                vid_labels = vid_labels.to(DEVICE)
                
                logits = prompt_module(vid_embeds) / self.temperature
                loss = F.cross_entropy(logits, vid_labels) + self.weight_decay * prompt_module.prompt_tokens.pow(2).sum()
                
                total_queries += len(logits)
                total_correct += (torch.argmax(logits, dim=-1) == vid_labels).sum()
                total_loss += loss * len(logits)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            scheduler.step()
                
            if epoch_idx % 5 == 0:
                print(f"Epoch: {epoch_idx:5}, Acc: {total_correct / total_queries:10.4f}, Loss: {total_loss / total_queries:10.4f}")
                
        # Predict for query videos
        with torch.no_grad():
            query_inputs = torch.concat([
                self.vlm.video_encoder_to_tokens(vid_path, random_augment=False).cpu()
                for vid_path in query_video_paths
            ])
            
            query_predictions = []
            for query_input_batch in torch.utils.data.DataLoader(query_inputs, batch_size=self.batch_size, num_workers=0):
                query_input_batch = query_input_batch.to(DEVICE)
                query_logits_batch = prompt_module(query_input_batch)
                query_predictions_batch = torch.argmax(query_logits_batch, dim=1).cpu()
                query_predictions.append(query_predictions_batch)
            query_predictions = torch.concat(query_predictions).numpy()
            
        return query_predictions    
    
    
    
class VisionPromptPrependModule(nn.Module):
    def __init__(self, vlm: SimilarityVLM, text_embeds: torch.Tensor, prompt_token_count: int):
        super().__init__()
        
        self.vlm = vlm
        
        # n_way x embed_dim
        self.register_buffer(
            "text_embeds",
            text_embeds / torch.norm(text_embeds, dim=-1, keepdim=True)
        )
        
        self.prompt_tokens = nn.Parameter(torch.zeros(prompt_token_count, vlm.video_token_dim()))
        
        # Initialization variance - from https://arxiv.org/abs/2203.12119
        init_bound = math.sqrt(6 / float(3 * vlm.video_frame_patch_size()**2 + vlm.video_token_dim()))
        nn.init.uniform_(self.prompt_tokens.data, -init_bound, init_bound)
        
    def forward(self, video_tokens: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            video_tokens (torch.Tensor): Shape (batch, frames, patches, token_dim)

        Returns:
            torch.Tensor: video embeddings, shape (batch, embed_dim)
        """
        # Compute embeddings, with additional learnable prompts to prepend
        video_embeds = self.vlm.video_encoder_from_tokens(video_tokens, prompt_tokens=self.prompt_tokens)
        
        # Compute class prediction logits
        video_embeds = F.normalize(video_embeds, dim=-1)
        class_logits = video_embeds @ self.text_embeds.T
        
        return class_logits
    
    
    
    
'''
Cosine LR scheduler with warmup steps used by https://arxiv.org/abs/2203.12119
'''
class WarmupCosineSchedule(torch.optim.lr_scheduler.LambdaLR):
    """ Linear warmup and then cosine decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps`.
        Decreases learning rate from 1. to 0. over remaining
            `t_total - warmup_steps` steps following a cosine curve.
        If `cycles` (default=0.5) is different from default, learning rate
            follows cosine function after warmup.
    """
    def __init__(self, optimizer, warmup_steps, t_total, cycles=.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(
            optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(max(
            1, self.t_total - self.warmup_steps))
        return max(
            0.0,
            0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress))
        )