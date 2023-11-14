from typing import Optional, List
import numpy as np
from copy import deepcopy
from tqdm.autonotebook import tqdm, trange
from functools import partial
from collections.abc import Mapping, Sequence
from torch.utils.data.dataloader import default_collate
from mmcv.parallel import collate
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

import torch
from torch import nn
from torch.nn import functional as F
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from SimilarityVLM import SimilarityVLM
from similarity_metrics import Similarity
from .base import FewShotClassifier
from VIFI_CLIP.wrapper import ViFiCLIP_SimilarityVLM

"""
Reimplementation of VL-Prompting method in ViFiCLIP (https://arxiv.org/abs/2212.03640).
Assumes VIFI_CLIP VLM, since it inserts trainable context vectors into each transformer layer
in both vision and text branches.
"""



from vifi_utils.pipeline import Compose
from vifi_utils.blending import CutmixMixupBlending

# Default augmentation settings
LABEL_SMOOTH = 0.1
COLOR_JITTER = 0.8
GRAY_SCALE = 0.2
MIXUP = 0.8
CUTMIX = 1.0
MIXUP_SWITCH_PROB = 0.5
INPUT_SIZE = 224

class VideoLoadingDataset(torch.utils.data.Dataset):
    def __init__(self,
                 video_filepaths: List[str],
                 labels: Optional[List[int]],
                 num_frames: int,
                 eval_mode: bool):
        # Pack filenames and labels into a list of dictionaries for pipeline input
        self.item_info = [
            dict(
                filename = f,
                label = 0,
                tar = False,
                start_index = 0,
                modality = "RGB"
            )
            for f in video_filepaths
        ]
        
        # Add real labels if included
        if labels is not None:
            for i in range(len(video_filepaths)):
                self.item_info[i]["label"] = labels[i]
                
        # Check if filenames include start/end frame information appended ("{path}:{start}:{end}")
        # If so, correct filenames and add start/end frame information to item_info
        for item in self.item_info:
            filename_split = item["filename"].split(":")
            if len(filename_split) == 3:
                filename = filename_split[0]
                start_frame = int(filename_split[1])
                end_frame = int(filename_split[2])
                item["filename"] = filename
                item["start_frame"] = start_frame
                item["end_frame"] = end_frame
        
        # Video loading pipelines
        scale_resize = int(256 / 224 * INPUT_SIZE)
        if eval_mode:
            self.pipeline = Compose([
                dict(type='DecordInit'),
                dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=num_frames, test_mode=True),
                dict(type='DecordDecode'),
                dict(type='Resize', scale=(-1, scale_resize)),
                dict(type='CenterCrop', crop_size=INPUT_SIZE),
                dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False),
                dict(type='FormatShape', input_format='NCHW'),
                dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
                dict(type='ToTensor', keys=['imgs'])
            ])
        else:
            self.pipeline = Compose([
                dict(type='DecordInit'),
                dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=num_frames),
                dict(type='DecordDecode'),
                dict(type='Resize', scale=(-1, scale_resize)),
                dict(
                    type='MultiScaleCrop',
                    input_size=INPUT_SIZE,
                    scales=(1, 0.875, 0.75, 0.66),
                    random_crop=False,
                    max_wh_scale_gap=1),
                dict(type='Resize', scale=(INPUT_SIZE, INPUT_SIZE), keep_ratio=False),
                dict(type='Flip', flip_ratio=0.5),
                dict(type='ColorJitter', p=COLOR_JITTER),
                dict(type='GrayScale', p=GRAY_SCALE),
                dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False),
                dict(type='FormatShape', input_format='NCHW'),
                dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
                dict(type='ToTensor', keys=['imgs', 'label']),
            ])
            
    def __len__(self):
        return len(self.item_info)
    
    def __getitem__(self, idx):
        return self.pipeline(deepcopy(self.item_info[idx]))
    
def mmcv_collate(batch, samples_per_gpu=1): 
    if not isinstance(batch, Sequence):
        raise TypeError(f'{batch.dtype} is not supported.')
    if isinstance(batch[0], Sequence):
        transposed = zip(*batch)
        return [collate(samples, samples_per_gpu) for samples in transposed]
    elif isinstance(batch[0], Mapping):
        return {
            key: mmcv_collate([d[key] for d in batch], samples_per_gpu)
            for key in batch[0]
        }
    else:
        return default_collate(batch)
    
class SubsetRandomSampler(torch.utils.data.Sampler):
    def __init__(self, indices):
        self.epoch = 0
        self.indices = indices
        
    def __iter__(self):
        return (self.indices[i] for i in torch.randperm(len(self.indices)))
    
    def __len__(self):
        return len(self.indices)
    
    def set_epoch(self, epoch):
        self.epoch = epoch
        
class Accumulator:
    def __init__(self):
        self.total = 0
    
    def add(self, value):
        self.total += value
        
    def get(self):
        if torch.distributed.is_initialized():
            total = torch.tensor(self.total).to(DEVICE)
            torch.distributed.all_reduce(total, op=torch.distributed.ReduceOp.SUM)
            return total.item()
        else:
            return self.total
    



class VLPromptNameTuningFewShotClassifier(FewShotClassifier):
    def __init__(self, vlm: SimilarityVLM,
                 name_regularization: float = 1.0,
                 text_context_len: int = 10,
                 text_context_depth: int = 12,
                 vision_context_len: int = 10,
                 vision_context_depth: int = 12,
                 lr: float = 8e-3,
                 epochs: int = 50,
                 warmup_lr: float = 1e-5,
                 warmup_epochs: int = 1,
                 batch_size: int = 4,
                 accumulation_steps: int = 2,
                 optimizer: str = "adamw"):
        assert isinstance(vlm, ViFiCLIP_SimilarityVLM)
        assert optimizer in ["sgd", "adam", "adamw"], "Invalid optimizer choice."
        self.vlm = vlm
        
        self.name_regularization = float(name_regularization)
        self.text_context_len = int(text_context_len)
        self.text_context_depth = int(text_context_depth)
        self.vision_context_len = int(vision_context_len)
        self.vision_context_depth = int(vision_context_depth)
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.warmup_lr = float(warmup_lr)
        self.warmup_epochs = float(warmup_epochs)
        self.batch_size = int(batch_size)
        self.accumulation_steps = int(accumulation_steps)
        self.optimizer = str(optimizer)
        
        
        
        
    '''
    Returns a dict with the value of all classifier-specific parameters which may affect prediction
    accuracy (apart from the underlying VLM object used).
    This is used to differentiate test results which use different classifier parameters.
    '''
    def params(self) -> dict:
        return {
            "name_regularization": self.name_regularization,
            "text_context_len": self.text_context_len,
            "text_context_depth": self.text_context_depth,
            "vision_context_len": self.vision_context_len,
            "vision_context_depth": self.vision_context_depth,
            "lr": self.lr,
            "epochs": self.epochs,
            "warmup_lr": self.warmup_lr,
            "warmup_epochs": self.warmup_epochs,
            "batch_size": self.batch_size,
            "accumulation_steps": self.accumulation_steps,
            "optimizer": self.optimizer
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
    def predict(self, category_names: np.ndarray, support_video_paths: Optional[np.ndarray],
                query_video_paths: np.ndarray, query_video_labels: np.ndarray,
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
            accuracy = (query_predictions == query_video_labels).mean()
            return accuracy
        
        """
        Set up dataloaders
        """
        train_data = VideoLoadingDataset(
            support_video_paths.flatten().tolist(),
            labels=torch.repeat_interleave(torch.arange(n_way), n_support).tolist(),
            num_frames=self.vlm.num_frames,
            eval_mode=False
        )
        if torch.distributed.is_initialized():
            train_sampler = torch.utils.data.DistributedSampler(
                train_data,
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                drop_last=False,
                shuffle=True
            )
        else:
            train_sampler = None
        train_dataloader = torch.utils.data.DataLoader(
            train_data,
            sampler=train_sampler,
            batch_size=self.batch_size,
            num_workers=0,
            pin_memory=True,
            shuffle=(train_sampler is None),
            collate_fn=partial(mmcv_collate, samples_per_gpu=self.batch_size)
        )
        
        # Check if able to use val tuning dataset (to select best-performing epoch)
        if val_tuning_video_paths is None or val_tuning_video_labels is None:
            val_tuning_dataloader = None
        else:
            val_data = VideoLoadingDataset(
                val_tuning_video_paths,
                labels=val_tuning_video_labels,
                num_frames=self.vlm.num_frames,
                eval_mode=True
            )
            if torch.distributed.is_initialized():
                val_sampler = torch.utils.data.DistributedSampler(
                    val_data,
                    num_replicas=torch.distributed.get_world_size(),
                    rank=torch.distributed.get_rank(),
                    shuffle=False,
                    drop_last=True
                )
            else:
                val_sampler = None
            val_tuning_dataloader = torch.utils.data.DataLoader(
                val_data,
                sampler=val_sampler,
                batch_size=self.batch_size,
                num_workers=0,
                pin_memory=True,
                shuffle=False,
                drop_last=True,
                collate_fn=partial(mmcv_collate, samples_per_gpu=self.batch_size)
            )
        
        query_data = VideoLoadingDataset(
            query_video_paths,
            labels=query_video_labels,
            num_frames=self.vlm.num_frames,
            eval_mode=True
        )
        if torch.distributed.is_initialized():
            query_sampler = torch.utils.data.DistributedSampler(
                query_data,
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                shuffle=False,
                drop_last=True
            )
        else:
            query_sampler = None
        query_dataloader = torch.utils.data.DataLoader(
            query_data,
            sampler=query_sampler,
            batch_size=self.batch_size,
            num_workers=0,
            pin_memory=True,
            shuffle=False,
            drop_last=True,
            collate_fn=partial(mmcv_collate, samples_per_gpu=self.batch_size)
        )
        
        """
        Initialize module
        """
        module = VLPrompt_Module(
            self.vlm,
            category_names,
            self.text_context_len,
            self.text_context_depth,
            self.vision_context_len,
            self.vision_context_depth
        )
        module.to(DEVICE)
        
        if torch.distributed.is_initialized():
            module = nn.parallel.DistributedDataParallel(
                module,
                device_ids=[torch.distributed.get_rank()],
                broadcast_buffers=False,
                find_unused_parameters=False
            )
        #print(list(module.named_parameters()))
        
        if self.optimizer == "sgd":
            optimizer = torch.optim.SGD(module.parameters(), lr=self.lr)
        elif self.optimizer == "adam":
            optimizer = torch.optim.Adam(module.parameters(), lr=self.lr, betas=(0.9, 0.98), eps=1e-8)
        elif self.optimizer == "adamw":
            optimizer = torch.optim.AdamW(module.parameters(), lr=self.lr, betas=(0.9, 0.98), eps=1e-8)
            
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            [
                torch.optim.lr_scheduler.ConstantLR(
                    optimizer,
                    min(1, self.warmup_lr / self.lr),
                    total_iters=self.warmup_epochs
                ),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=self.epochs - self.warmup_epochs
                )
            ],
            [self.warmup_epochs]
        )
        
        grad_scaler = torch.cuda.amp.GradScaler(enabled=True)
        
        """
        Set up loss function and mixup function
        """
        mixup_fn = None
        if MIXUP > 0:
            criterion = SoftTargetCrossEntropy()
            mixup_fn = CutmixMixupBlending(num_classes=len(category_names),
                                           smoothing=LABEL_SMOOTH,
                                           mixup_alpha=MIXUP,
                                           cutmix_alpha=CUTMIX,
                                           switch_prob=MIXUP_SWITCH_PROB)
        elif LABEL_SMOOTH > 0:
            criterion = LabelSmoothingCrossEntropy(smoothing=LABEL_SMOOTH)
        else:
            criterion = nn.CrossEntropyLoss()
        
        """
        Train
        """
        # Save best-performing model (if val tuning dataset is provided)
        val_tuning_best_acc = None
        val_tuning_best_model_state = None
        for epoch_idx in range(self.epochs):
            total_loss = Accumulator()
            total_correct = Accumulator()
            total_count = Accumulator()
            
            if torch.distributed.is_initialized():
                train_dataloader.sampler.set_epoch(epoch_idx)
            optimizer.zero_grad()
            
            # Compute how many batches don't divide by accumulation steps (to be scaled differently at end of epoch)
            accumulation_remainder_batches = len(train_dataloader) % self.accumulation_steps
            
            train_iter = enumerate(train_dataloader)
            if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                train_iter = tqdm(train_iter, total=len(train_dataloader), desc="train")
                
            for batch_idx, batch in train_iter:                
                vid_frames = batch["imgs"].to(DEVICE, non_blocking=True)
                vid_labels = batch["label"].flatten().to(DEVICE, non_blocking=True)
                orig_vid_labels = vid_labels # vid_labels as int, before smoothing
                
                if mixup_fn is not None:
                    vid_frames, vid_labels = mixup_fn(vid_frames, vid_labels)
                
                with torch.autocast(device_type=DEVICE, dtype=torch.float16, enabled=True):
                    logits = module(vid_frames)
                    loss = criterion(logits, vid_labels)
                    
                    # Add name regularization term
                    reg_loss = 0.5 * self.name_regularization * module.name_perturbation.pow(2).sum()
                    loss = loss + reg_loss
                    
                    # Scale down loss for accumulation steps
                    # Number of accumulation steps may be smaller for final batches (if not divisible by self.accumulation_steps)
                    if batch_idx >= len(train_dataloader) - accumulation_remainder_batches:
                        loss = loss / accumulation_remainder_batches
                    else:
                        loss = loss / self.accumulation_steps
                
                total_loss.add(loss.item() * len(vid_frames))
                total_correct.add((logits.argmax(dim=1) == orig_vid_labels).sum().detach().item())
                total_count.add(len(vid_frames))
                
                grad_scaler.scale(loss).backward()
                #loss.backward()
                if (batch_idx + 1) % self.accumulation_steps == 0 or (batch_idx + 1) == len(train_dataloader):
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                    #optimizer.step()                 
                    optimizer.zero_grad()
                    
                torch.cuda.synchronize()
            
            scheduler.step()
            
            train_loss = total_loss.get() / total_count.get()
            train_acc = total_correct.get() / total_count.get()
            
            # Check val-tuning performance
            if val_tuning_dataloader is not None:
                total_val_correct = Accumulator()
                total_val_count = Accumulator()
                
                val_iter = enumerate(val_tuning_dataloader)
                if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                    val_iter = tqdm(val_iter, total=len(val_tuning_dataloader), desc="val")
                
                with torch.no_grad():
                    for batch_idx, batch in val_iter:
                        vid_frames = batch["imgs"].to(DEVICE, non_blocking=True)
                        vid_labels = batch["label"].flatten().to(DEVICE, non_blocking=True)
                        
                        with torch.autocast(device_type=DEVICE, dtype=torch.float16, enabled=True):
                            logits = module(vid_frames)
                            
                        total_val_correct.add((logits.argmax(dim=1) == vid_labels).sum().item())
                        total_val_count.add(len(vid_frames))
                
                val_acc = total_val_correct.get() / total_val_count.get()
                if val_tuning_best_acc is None or val_acc >= val_tuning_best_acc:
                    val_tuning_best_acc = val_acc
                    val_tuning_best_model_state = deepcopy(module.state_dict())
                
                if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                    print(f"Epoch {epoch_idx:5}: Support Acc = {train_acc:5.3f}, Val-Tune Acc = {val_acc:5.3f}, Loss = {train_loss:5.3f}")
            else:
                if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                    print(f"Epoch {epoch_idx:5}: Support Acc = {train_acc:5.3f}, Loss = {train_loss:5.3f}")
        
        """
        Compute query-set outputs with trained module
        """        
        # Reload best val-tuning model state
        if val_tuning_best_model_state is not None:
            module.load_state_dict(val_tuning_best_model_state)
        
        query_correct = Accumulator()
        query_count = Accumulator()
        
        query_iter = enumerate(query_dataloader)
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            query_iter = tqdm(query_iter, total=len(query_dataloader), desc="query")
        
        with torch.no_grad():
            for batch_idx, batch in query_iter:
                vid_frames = batch["imgs"].to(DEVICE, non_blocking=True)
                vid_labels = batch["label"].to(DEVICE, non_blocking=True)
                
                with torch.autocast(device_type=DEVICE, dtype=torch.float16, enabled=True):
                    logits = module(vid_frames)
                predictions = logits.argmax(dim=1)
                
                query_correct.add((predictions == vid_labels).sum().item())
                query_count.add(len(predictions))
                        
                        
                
        query_accuracy = query_correct.get() / query_count.get()
        return query_accuracy



from VIFI_CLIP import clip
class VLPrompt_Module(nn.Module):
    def __init__(self, vlm: SimilarityVLM,
                 category_names: np.ndarray,
                 text_context_len: int,
                 text_context_depth: int,
                 vision_context_len: int,
                 vision_context_depth: int):
        super().__init__()
        
        assert isinstance(vlm, ViFiCLIP_SimilarityVLM)
        self.vlm = vlm
        self.clip_model = vlm.model
        
        
        # Perform initial tokenization and add positional embedding (everything before transformer layer) for each category name, using placeholder words to set aside space for context vectors
        if text_context_depth > 0:
            context_placeholder_string = " ".join(["X"] * text_context_len)
            category_names = [
                f"{context_placeholder_string} {name}"
                for name in category_names
            ]
        else:
            category_names = category_names.tolist()
        category_name_tokens = clip.tokenize(category_names).to(DEVICE)
        pre_transformer_text_embeds = self.encode_text_pre_transformer(category_name_tokens)
        self.register_buffer("pre_transformer_text_embeds", pre_transformer_text_embeds)
        
        eot_token_inds = category_name_tokens.argmax(dim=-1)
        self.register_buffer("eot_token_inds", eot_token_inds)
        
        
        # Create name perturbation parameter (seq_len, categories, dim)
        name_perturbation = torch.zeros_like(pre_transformer_text_embeds)
        self.name_perturbation = nn.Parameter(name_perturbation)
        
        # Create name perturbation mask as buffer 
        name_token_inds = torch.arange(category_name_tokens.size(1), device=DEVICE)[None, :]
        name_token_mask = (name_token_inds >= 1 + text_context_len) & (name_token_inds < eot_token_inds[:, None])
        name_perturbation_mask = name_token_mask.transpose(0, 1) # (categories, seq_len) -> (seq_len, categories)
        self.register_buffer("name_perturbation_mask", name_perturbation_mask)
        
        
        
        # Create text context parameter
        if text_context_depth > 0:
            text_context = torch.empty(text_context_depth, text_context_len, self.vlm.model.token_embedding.embedding_dim, dtype=self.vlm.model.dtype)
            nn.init.normal_(text_context, std=0.02)
            self.text_context = nn.Parameter(text_context)
        else:
            self.text_context = None
            
            
        
        # Create vision context parameter
        if vision_context_depth > 0:
            vision_width = self.vlm.model.visual.conv1.weight.size(0)
            vision_context = torch.empty(vision_context_depth, vision_context_len, vision_width)
            nn.init.normal_(vision_context)
            self.vision_context = nn.Parameter(vision_context)
        else:
            self.vision_context = None

    # Compute the text branch for all category names
    def tuned_text_embeds(self):
        x = self.pre_transformer_text_embeds # Shape = (seq_len, categories, dim)
        
        # Add name perturbation
        x = x + self.name_perturbation * self.name_perturbation_mask[:, :, None]
        
        for i, resblock in enumerate(self.vlm.model.transformer.resblocks):
            if self.text_context is not None and i < len(self.text_context):
                context = self.text_context[i].unsqueeze(1).expand(-1, x.size(1), -1).type(x.dtype)
                prefix = x[:1, :, :]
                suffix = x[1 + len(context):, :, :]
                x = torch.cat([prefix, context, suffix], dim=0)
            x = resblock(x)
        embeds = self.encode_text_post_transformer(x, self.eot_token_inds)
        return embeds
    
    def tuned_video_embeds(self, video_frames: torch.Tensor):
        bsz, num_frames, c, h, w = video_frames.shape
        
        x = video_frames.reshape(bsz * num_frames, c, h, w)
        x = self.encode_image_pre_transformer(x) # Shape = (seq_len, bsz * num_frames, dim)
        
        for i, resblock in enumerate(self.vlm.model.visual.transformer.resblocks):
            if self.vision_context is not None and i < len(self.vision_context):
                context = self.vision_context[i].unsqueeze(1).expand(-1, x.size(1), -1).type(x.dtype)
                if i == 0:
                    # First layer requires appending visual context elements
                    prefix = x
                else:
                    # Later layers have already been extended by previous context
                    prefix = x[:-len(context)]
                x = torch.cat([prefix, context], dim=0)
            x = resblock(x)
        
        frame_embeds = self.encode_image_post_transformer(x)
        frame_embeds = frame_embeds.reshape(bsz, num_frames, -1)
        video_embeds = frame_embeds.mean(dim=1)
        return video_embeds
    
    def forward(self, video_frames: torch.Tensor):
        bsz, num_frames, c, h, w = video_frames.shape
        
        text_embeds = self.tuned_text_embeds()
        video_embeds = self.tuned_video_embeds(video_frames)
        
        # Compute logits using cosine similarity
        text_embeds = F.normalize(text_embeds, dim=1)
        video_embeds = F.normalize(video_embeds, dim=1)
        logits = self.vlm.logit_scale() * (video_embeds @ text_embeds.T)
        return logits
                
        
    

    """
    Helper functions to copy parts of the vlm.mode text/video encoding procedure, isolating transformer use
    """
    @torch.no_grad()
    def encode_text_pre_transformer(self, tokens):
        x = self.vlm.model.token_embedding(tokens).type(self.vlm.model.dtype)
        x = x + self.vlm.model.positional_embedding.type(self.vlm.model.dtype)
        x = x.permute(1, 0, 2) # NLD -> LND
        return x
    
    def encode_text_post_transformer(self, x, eot_token_inds):
        x = x.permute(1, 0, 2) # LND -> NLD
        x = self.vlm.model.ln_final(x).type(self.vlm.model.dtype)
        
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), eot_token_inds] @ self.vlm.model.text_projection
        return x
    
    @torch.no_grad()
    def encode_image_pre_transformer(self, frames):
        x = frames.type(self.vlm.model.dtype)
        
        x = self.vlm.model.visual.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.vlm.model.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.vlm.model.visual.positional_embedding.to(x.dtype)
        x = self.vlm.model.visual.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        return x
    
    def encode_image_post_transformer(self, x):
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.vlm.model.visual.ln_post(x[:, 0, :])
        if self.vlm.model.visual.proj is not None:
            x = x @ self.vlm.model.visual.proj
        return x