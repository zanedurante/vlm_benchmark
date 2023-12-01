from typing import Optional, List
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
MAX_QUERY_SAMPLES_PER_EPOCH = 2048 # Number of random samples to draw from val dataset if using val tuning

'''
Name-tuning and prompt-ensembling only
'''
class NameTuningFewShotClassifier(FewShotClassifier):
    def __init__(self, vlm: SimilarityVLM,
                 prompt_ensemble_id: Optional[str] = None,
                 name_regularization: float = 1e-2,
                 lr: float = 1e-3,
                 epochs: int = 20,
                 warmup_lr: float = 1e-5,
                 warmup_epochs: int = 1,
                 batch_size: int = 1,
                 optimizer: str = "sgd",
                 random_augment: bool = True,
                 low_memory_training: bool = False
                 ):
        self.vlm = vlm
        
        self.prompt_ensemble_id = prompt_ensemble_id
        self.name_regularization = float(name_regularization)
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.warmup_lr = float(warmup_lr)
        self.warmup_epochs = float(warmup_epochs)
        self.batch_size = int(batch_size)
        self.optimizer = str(optimizer)
        self.random_augment = bool(random_augment)
        self.low_memory_training = bool(low_memory_training)
        
        assert prompt_ensemble_id in PROMPT_ENSEMBLES.keys(), "Unrecognized prompt_ensemble_id."
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
            "prompt_ensemble_id": self.prompt_ensemble_id,
            "name_regularization": self.name_regularization,
            "lr": self.lr,
            "epochs": self.epochs,
            "warmup_lr": self.warmup_lr,
            "warmup_epochs": self.warmup_epochs,
            "batch_size": self.batch_size,
            "optimizer": self.optimizer,
            "random_augment": self.random_augment,
            "low_memory_training": self.low_memory_training
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
    def predict(self, category_names: np.ndarray, support_video_paths: Optional[np.ndarray], query_video_paths: np.ndarray, query_video_labels: np.ndarray,
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
            text_embeds = np.array([
                [
                    self.vlm.get_text_embeds(template.format(name))
                    for template in PROMPT_ENSEMBLES[self.prompt_ensemble_id]
                ]
                for name in category_names
            ]).mean(axis=1)
                
            
            # Query Vid Embeddings
            query_vid_embeds = np.array([self.vlm.get_video_embeds(vid_path) for vid_path in query_video_paths])
            
            query_to_text_similarities = self.vlm.default_similarity_metric()(query_vid_embeds, text_embeds)
            
            # Save category probabilities for each class
            query_class_logits = (self.vlm.logit_scale() * query_to_text_similarities)
            self.query_class_probabilities = np.exp(query_class_logits) / np.sum(np.exp(query_class_logits), axis=1, keepdims=True)
            
            # Return direct predictions
            query_predictions = np.argmax(query_to_text_similarities, axis=1)
            accuracy = (query_predictions == query_video_labels).mean()
            return accuracy
        
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
            
        module = NameTuningModule(self.vlm, category_names, PROMPT_ENSEMBLES[self.prompt_ensemble_id])
        module.to(DEVICE)
        
        optim_input = [
            {"params": module.name_perturbation, "weight_decay": self.name_regularization}
        ]
        if self.optimizer == "sgd":
            optimizer = torch.optim.SGD(optim_input, lr=self.lr, weight_decay=0)
        elif self.optimizer == "adam":
            optimizer = torch.optim.Adam(optim_input, lr=self.lr, weight_decay=0)
        elif self.optimizer == "adamw":
            optimizer = torch.optim.AdamW(optim_input, lr=self.lr, weight_decay=0)
        else:
            raise NotImplementedError
        
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
                   
                if self.low_memory_training:
                    prompt_indices = np.random.choice(len(PROMPT_ENSEMBLES[self.prompt_ensemble_id]), size=1)
                else:
                    prompt_indices = range(len(PROMPT_ENSEMBLES[self.prompt_ensemble_id]))
                    
                logits = module(vid_embeds, prompt_indices)
                loss = F.cross_entropy(logits, vid_labels)
                
                total_loss += loss.item() * len(vid_paths)
                with torch.no_grad():
                    total_reg_loss += 0.5 * self.name_regularization * module.name_perturbation.pow(2).sum() * len(vid_paths)
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

                    if (batch_idx + 1) * QUERY_BATCH_SIZE >= MAX_QUERY_SAMPLES_PER_EPOCH:
                        break
                    
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
            self.tuned_input_embeds = module.tuned_class_input_word_embeds(0)
                
        query_dataloader = torch.utils.data.DataLoader(list(zip(query_video_paths, query_video_labels)), batch_size=QUERY_BATCH_SIZE, num_workers=0, shuffle=False)
        query_correct = 0
        query_total = 0
        with torch.no_grad():
            for batch_idx, (vid_paths, vid_labels) in enumerate(query_dataloader):
                batch_query_vid_embeds = torch.cat([
                    torch.from_numpy(self.vlm.get_video_embeds(vid_path)).unsqueeze(0).to(DEVICE)
                    for vid_path in vid_paths
                ])
                logits = module(batch_query_vid_embeds)
                preds = torch.argmax(logits, dim=1).cpu()
                query_correct += (preds == vid_labels).sum().item()
                query_total += len(preds)
        
        accuracy = query_correct / query_total
        return accuracy
        
        
        
        
        
class NameTuningModule(nn.Module):
    def __init__(self, vlm: SimilarityVLM, category_names: np.ndarray, prompt_templates: List[str]):
        super().__init__()
        
        self.vlm = vlm
        self.category_names = category_names
        self.prompt_templates = prompt_templates

        # Class-specific name embedding perturbations
        with torch.no_grad():
            category_name_input_embeds, category_name_attn_masks = vlm.get_input_word_embeddings(category_names.tolist())
        name_perturbation = torch.zeros_like(category_name_input_embeds)
        self.name_perturbation = nn.Parameter(name_perturbation)
        
        # Create mask for which word embeddings can be perturbed (attn mask but with special tokens set to False)
        # attn mask is left-aligned, so end special tokens have no fixed index. 
        # Instead we use torch.roll to shorten each sequence
        name_token_mask = category_name_attn_masks.clone().type(torch.bool)
        name_token_mask = torch.roll(name_token_mask, -self.vlm.text_end_special_token_count(), dims=1)
        name_token_mask[:, -self.vlm.text_end_special_token_count():] = False
        name_token_mask[:, :self.vlm.text_start_special_token_count()] = False
        self.register_buffer("name_token_mask", name_token_mask)
        
        # Save fixed input word embeddings for each name and prompt template combination
        with torch.no_grad():
            prompt_name_input_embeds, prompt_name_attn_masks = vlm.get_input_word_embeddings([
                template.format(name)
                for template in prompt_templates
                for name in category_names
            ])
        seq_len = prompt_name_input_embeds.shape[1]
        word_embed_dim = prompt_name_input_embeds.shape[2]
        prompt_name_input_embeds = prompt_name_input_embeds.reshape(len(prompt_templates), len(category_names), seq_len, word_embed_dim)
        prompt_name_attn_masks = prompt_name_attn_masks.reshape(len(prompt_templates), len(category_names), seq_len)
        self.register_buffer("prompt_name_input_embeds", prompt_name_input_embeds)
        self.register_buffer("prompt_name_attn_masks", prompt_name_attn_masks)
        
        # Also save the length of the part of the template before the category name
        with torch.no_grad():
            pre_name_templates = [template.split("{}")[0] for template in prompt_templates]
            self.pre_name_template_lengths = [
                vlm.get_input_word_embeddings([pre_name_template])[0].shape[1] - self.vlm.text_start_special_token_count() - self.vlm.text_end_special_token_count()
                for pre_name_template in pre_name_templates
            ]
            
    def tuned_class_input_word_embeds(self, prompt_index: int):
        """Add the current perturbation parameter to input word embeddings
        for each class name for a single prompt.

        Args:
            prompt_index (int)
        """
        masked_name_perturbation = self.name_perturbation * self.name_token_mask.unsqueeze(2)
        name_seq_len = masked_name_perturbation.shape[1]
        
        input_embeds = self.prompt_name_input_embeds[prompt_index].clone()
        offset = self.pre_name_template_lengths[prompt_index]
        input_embeds[:, offset : offset + name_seq_len] += masked_name_perturbation
        
        attn_mask = self.prompt_name_attn_masks[prompt_index]
        return input_embeds, attn_mask
        
        
    def tuned_text_embeds(self, prompt_indices: Optional[List[int]] = None):
        if prompt_indices is None:
            prompt_indices = range(len(self.prompt_templates))
        
        prompt_summed_name_embeds = None
        for i in prompt_indices:
            tuned_input_embeds, attn_mask = self.tuned_class_input_word_embeds(i)
            output_embeds = self.vlm.text_encoder_from_word_embeddings(tuned_input_embeds, attn_mask)
            if prompt_summed_name_embeds is None:
                prompt_summed_name_embeds = output_embeds
            else:
                prompt_summed_name_embeds += output_embeds
        name_embeds = prompt_summed_name_embeds / len(prompt_indices)
        return name_embeds
    
    def forward(self, vid_embeds: torch.Tensor, prompt_indices: Optional[List[int]] = None) -> torch.Tensor:
        text_embeds = self.tuned_text_embeds(prompt_indices)
        
        if self.vlm.default_similarity_metric() == Similarity.COSINE:
            vid_embeds = F.normalize(vid_embeds, dim=1)
            text_embeds = F.normalize(text_embeds, dim=1)
            logits = self.vlm.logit_scale() * (vid_embeds @ text_embeds.T)
            
        elif self.vlm.default_similarity_metric() == Similarity.DOT:
            logits = self.vlm.logit_scale() * (vid_embeds @ text_embeds.T)
            
        else:
            raise NotImplementedError
        
        return logits