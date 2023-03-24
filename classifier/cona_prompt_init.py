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

QUERY_BATCH_SIZE = 2048 # Batch size used for iterating through non-training data

'''
Ablated version of CoNa, in which the class-shared context vector is initialized to a text prompt,
and regularized to stay near it.
'''
class CoNaPromptInitFewShotClassifier(FewShotClassifier):
    def __init__(self, vlm: SimilarityVLM,
                 context_prompt_init: str = "a photo of", opt_strat: str = "joint",
                 context_regularization: float = 1e-2, name_regularization: float = 1e-2,
                 lr: float = 1e-3, epochs: int = 10,
                 warmup_lr: float = 1e-5, warmup_epochs: int = 1, 
                 batch_size: int = 1, optimizer: str = "sgd",
                 random_augment: bool = True):
        self.vlm = vlm
        
        self.context_prompt_init = str(context_prompt_init).split("{}")[0] # Remove any template slots used in other prompt formats.
        self.opt_strat = str(opt_strat)
        self.context_regularization = float(context_regularization)
        self.name_regularization = float(name_regularization)
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.warmup_lr = float(warmup_lr)
        self.warmup_epochs = float(warmup_epochs)
        self.batch_size = int(batch_size)
        self.optimizer = str(optimizer)
        self.random_augment = bool(random_augment)
        
        assert opt_strat in ["joint"], "Invalid optimization strategy."
        assert optimizer in ["sgd", "adam", "adamw"], "Invalid optimizer choice."
        
        # Save the latest progression of tuned class embeddings over epochs for visualization
        # {class name -> [orig text embed, tuned text embed epoch 0, epoch 1, ...]}
        self.text_embed_training_record = {}
        
        # Save these parts of state after training and prediction on a few-shot task
        # - trained input word embeddings
        # - class probabilities for each query video
        self.tuned_input_embeds = None
        self.query_class_probabilities = None
        
    '''
    Returns a dict with the value of all classifier-specific parameters which may affect prediction
    accuracy (apart from the underlying VLM object used).
    This is used to differentiate test results which use different classifier parameters.
    '''
    def params(self) -> dict:
        return {
            "context_prompt_init": self.context_prompt_init,
            "opt_strat": self.opt_strat,
            "context_regularization": self.context_regularization,
            "name_regularization": self.name_regularization,
            "lr": self.lr,
            "epochs": self.epochs,
            "warmup_lr": self.warmup_lr,
            "warmup_epochs": self.warmup_epochs,
            "batch_size": self.batch_size,
            "optimizer": self.optimizer,
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
        
        # Use default similarity to text embeds if zero-shot
        if n_support == 0:
            # Text Embeddings
            text_embeds = np.array([self.vlm.get_text_embeds(name) for name in category_names])
            
            # Query Vid Embeddings
            query_vid_embeds = np.array([self.vlm.get_video_embeds(vid_path) for vid_path in query_video_paths])
            
            query_to_text_similarities = self.vlm.default_similarity_metric()(query_vid_embeds, text_embeds)
            
            # Save category probabilities for each class
            query_class_logits = (self.vlm.logit_scale() * query_to_text_similarities)
            self.query_class_probabilities = np.exp(query_class_logits) / np.sum(np.exp(query_class_logits), axis=1, keepdims=True)
            
            # Return direct predictions
            query_predictions = np.argmax(query_to_text_similarities, axis=1)
            return query_predictions
        
        # Save original text embeds for visualization
        # Tuned text embeds will be added to this list
        self.text_embed_training_record = {
            name: [self.vlm.get_text_embeds(name)]
            for name in category_names
        }
        
        train_dataloader = torch.utils.data.DataLoader(
            list(zip(support_video_paths.flatten(), torch.repeat_interleave(torch.arange(n_way), n_support))),
            batch_size=self.batch_size, num_workers=0, shuffle=True
        )
        
        # Check if able to use val tuning dataset (to select best-performing epoch)
        if val_tuning_video_paths is None or val_tuning_video_labels is None:
            val_tuning_dataloader = None
        else:
            val_tuning_dataloader = torch.utils.data.DataLoader(
                list(zip(val_tuning_video_paths, val_tuning_video_labels)),
                batch_size=QUERY_BATCH_SIZE, num_workers=0, shuffle=False
            )
        
        module = CoNaPromptInitModule(self.vlm, category_names, self.context_prompt_init)
        module.to(DEVICE)
        
        optim_input = [
            {"params": module.name_perturbation, "weight_decay": self.name_regularization},
            {"params": module.context_perturbation, "weight_decay": self.context_regularization}
        ]
        if self.optimizer == "sgd":
            optimizer = torch.optim.SGD(optim_input, lr=self.lr, weight_decay=0)
        elif self.optimizer == "adam":
            optimizer = torch.optim.Adam(optim_input, lr=self.lr, weight_decay=0)
        elif self.optimizer == "adamw":
            optimizer = torch.optim.AdamW(optim_input, lr=self.lr, weight_decay=0)
        else:
            raise NotImplementedError
        
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
        
        # Save best-performing model (if val tuning dataset is provided)
        val_tuning_best_acc = None
        val_tuning_best_model_state = None
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
                
                logits = module(vid_embeds)
                loss = F.cross_entropy(logits, vid_labels)
                
                total_loss += loss.item() * len(vid_paths)
                with torch.no_grad():
                    total_reg_loss += 0.5 * self.name_regularization * module.name_perturbation.pow(2).sum() * len(vid_paths)
                    total_reg_loss += 0.5 * self.context_regularization * module.context_perturbation.pow(2).sum() * len(vid_paths)
                total_correct += (logits.argmax(dim=1) == vid_labels).sum()
                total_count += len(vid_paths)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()
            
            # Check val-tuning performance
            if val_tuning_dataloader is not None:
                total_val_correct = 0
                total_val_count = 0
                for batch_idx, (vid_paths, vid_labels) in enumerate(val_tuning_dataloader):
                    vid_embeds = torch.cat([
                        torch.from_numpy(self.vlm.get_video_embeds(vid_path)).unsqueeze(0).to(DEVICE)
                        for vid_path in vid_paths
                    ], dim=0)
                    vid_labels = vid_labels.to(DEVICE)
                    
                    with torch.no_grad():
                        logits = module(vid_embeds)
                    total_val_correct += (logits.argmax(dim=1) == vid_labels).sum()
                    total_val_count += len(vid_paths)
                    
                val_acc = total_val_correct / total_val_count
                if val_tuning_best_acc is None or val_acc >= val_tuning_best_acc:
                    val_tuning_best_acc = val_acc
                    val_tuning_best_model_state = deepcopy(module.state_dict())
                    
                print(f"Epoch {epoch_idx:5}: Support Acc = {total_correct / total_count:5.3f}, Val-Tune Acc = {total_val_correct / total_val_count:5.3f}, Loss = {total_loss / total_count:5.3E}, Reg Loss = {total_reg_loss / total_count:5.3E}, Combined Loss = {(total_loss + total_reg_loss) / total_count:5.3E}")
            else:
                print(f"Epoch {epoch_idx:5}: Support Acc = {total_correct / total_count:5.3f}, Loss = {total_loss / total_count:5.3E}, Reg Loss = {total_reg_loss / total_count:5.3E}, Combined Loss = {(total_loss + total_reg_loss) / total_count:5.3E}")
                
            # Save tuned text output embeds into record
            with torch.no_grad():
                text_embeds = module.tuned_text_embeds().cpu().numpy()
                for i, name in enumerate(category_names):
                    self.text_embed_training_record[name].append(text_embeds[i])
                
                
        # Reload best val-tuning model state
        if val_tuning_best_model_state is not None:
            module.load_state_dict(val_tuning_best_model_state)
            
        # Save tuned class input text embeds
        with torch.no_grad():
            self.tuned_input_embeds = module.tuned_class_input_word_embeds()
                
        query_dataloader = torch.utils.data.DataLoader(query_video_paths, batch_size=QUERY_BATCH_SIZE, num_workers=0, shuffle=False)
        with torch.no_grad():
            query_logits = []
            for batch_idx, vid_paths in enumerate(query_dataloader):
                batch_query_vid_embeds = torch.cat([
                    torch.from_numpy(self.vlm.get_video_embeds(vid_path)).unsqueeze(0).to(DEVICE)
                    for vid_path in vid_paths
                ])
                query_logits.append(module(batch_query_vid_embeds))
            query_logits = torch.cat(query_logits, dim=0)
            
            # Save class probabilities
            self.query_class_probabilities = torch.softmax(query_logits, dim=1).cpu().numpy()
                
            query_predictions = torch.argmax(query_logits, dim=1).cpu().numpy()
            return query_predictions
        
                
        
        
class CoNaPromptInitModule(nn.Module):
    def __init__(self, vlm: SimilarityVLM, category_names: np.ndarray, context_prompt_init: str):
        super().__init__()
        
        self.vlm = vlm
        self.category_names = category_names
        
        # Orig context prompt embedding
        # We will also remove the special tokens from the start/end of this, since we will insert it into
        # the category name embeddings, which already have special tokens
        with torch.no_grad():
            context_input_embeds, context_attn_mask = vlm.get_input_word_embeddings([context_prompt_init])
        context_start = vlm.text_start_special_token_count()
        context_end = context_attn_mask[0].sum() - vlm.text_end_special_token_count()
        context_input_embeds = context_input_embeds[:, context_start : context_end]
        self.register_buffer("context_input_embeds", context_input_embeds)
        
        # Class-shared context tweaks
        context_perturbation = torch.zeros_like(context_input_embeds)
        self.context_perturbation = nn.Parameter(context_perturbation)
        
        # Orig Name embeddings
        with torch.no_grad():
            category_name_input_embeds, category_name_attn_masks = vlm.get_input_word_embeddings(category_names.tolist())
        self.register_buffer("category_name_input_embeds", category_name_input_embeds)
        self.register_buffer("category_name_attn_masks", category_name_attn_masks)
        
        # Class-specific name embedding tweaks
        #name_perturbation = torch.empty_like(category_name_input_embeds)
        #nn.init.normal_(name_perturbation, std=0.02)
        name_perturbation = torch.zeros_like(category_name_input_embeds)
        self.name_perturbation = nn.Parameter(name_perturbation)
        
        # Mask for class embeddings which aren't special tokens
        name_token_mask = category_name_attn_masks.clone().type(torch.bool)
        name_token_mask = torch.roll(name_token_mask, -self.vlm.text_end_special_token_count(), dims=1)
        name_token_mask[:, -self.vlm.text_end_special_token_count():] = False
        name_token_mask[:, :self.vlm.text_start_special_token_count()] = False
        self.register_buffer("name_token_mask", name_token_mask)
        
    def tuned_class_input_word_embeds(self):
        """Add the current perturbation parameter to input word embeddings
        for each class name.
        """
        # Retain only name perturbations which correspond to actual words (not special tokens or padding)
        masked_name_perturbation = self.name_perturbation * self.name_token_mask.unsqueeze(2)
        perturbed_category_name_input_embeds = self.category_name_input_embeds + masked_name_perturbation
        
        # No masking necessary for context perturbation, since special tokens and padding are already removed
        perturbed_context_input_embeds = self.context_input_embeds + self.context_perturbation
        perturbed_context_input_embeds = perturbed_context_input_embeds.expand(self.category_name_input_embeds.size(0), -1, -1)
        
        text_input_embeds = torch.cat(
            [
                perturbed_category_name_input_embeds[:, :self.vlm.text_start_special_token_count(), :],
                perturbed_context_input_embeds,
                perturbed_category_name_input_embeds[:, self.vlm.text_start_special_token_count():, :]
            ], dim=1
        )
        text_input_attn_masks = torch.cat(
            [
                self.category_name_attn_masks[:, :self.vlm.text_start_special_token_count()],
                torch.ones(*perturbed_context_input_embeds.shape[:-1], device=DEVICE),
                self.category_name_attn_masks[:, self.vlm.text_start_special_token_count():]
            ], dim=1
        )
        return text_input_embeds, text_input_attn_masks
        
    def tuned_text_embeds(self):
        text_input_embeds, text_input_attn_masks = self.tuned_class_input_word_embeds()
        text_embeds = self.vlm.text_encoder_from_word_embeddings(text_input_embeds, text_input_attn_masks)
        return text_embeds
        
    def forward(self, vid_embeds: torch.Tensor) -> torch.Tensor:
        text_embeds = self.tuned_text_embeds()
        
        if self.vlm.default_similarity_metric() == Similarity.COSINE:
            vid_embeds = F.normalize(vid_embeds, dim=1)
            text_embeds = F.normalize(text_embeds, dim=1)
            logits = self.vlm.logit_scale() * (vid_embeds @ text_embeds.T)
            
        elif self.vlm.default_similarity_metric() == Similarity.DOT:
            logits = self.vlm.logit_scale() * (vid_embeds @ text_embeds.T)
            
        else:
            raise NotImplementedError
        
        return logits
        