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
        if labels is not None:
            for i in range(len(video_filepaths)):
                self.item_info[i]["label"] = labels[i]
        
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
        return self.pipeline(self.item_info[idx])
    
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



class VLPromptFewShotClassifier(FewShotClassifier):
    def __init__(self, vlm: SimilarityVLM,
                 text_context_len: int = 16,
                 text_context_depth: int = 10,
                 vision_context_len: int = 16,
                 vision_context_depth: int = 10,
                 lr: float = 1e-3,
                 epochs: int = 20,
                 warmup_lr: float = 1e-5,
                 warmup_epochs: int = 1,
                 batch_size: int = 1,
                 accumulation_steps: int = 1,
                 optimizer: str = "adamw"):
        assert isinstance(vlm, ViFiCLIP_SimilarityVLM)
        assert optimizer in ["sgd", "adam", "adamw"], "Invalid optimizer choice."
        self.vlm = vlm
        
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
        
        """
        Set up dataloaders
        """
        train_data = VideoLoadingDataset(
            support_video_paths.flatten().tolist(),
            labels=torch.repeat_interleave(torch.arange(n_way), n_support).tolist(),
            num_frames=self.vlm.num_frames,
            eval_mode=False
        )
        train_dataloader = torch.utils.data.DataLoader(
            train_data, batch_size=self.batch_size, num_workers=0, shuffle=True,
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
            val_tuning_dataloader = torch.utils.data.DataLoader(
                val_data, batch_size=self.batch_size, num_workers=0, shuffle=True,
                collate_fn=partial(mmcv_collate, samples_per_gpu=self.batch_size)
            )
        
        query_data = VideoLoadingDataset(
            query_video_paths,
            labels=None,
            num_frames=self.vlm.num_frames,
            eval_mode=True
        )
        query_dataloader = torch.utils.data.DataLoader(
            query_data, batch_size=self.batch_size, num_workers=0, shuffle=True,
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
        #print(list(module.named_parameters()))
        
        if self.optimizer == "sgd":
            optimizer = torch.optim.SGD(module.parameters(), lr=self.lr)
        elif self.optimizer == "adam":
            optimizer = torch.optim.Adam(module.parameters(), lr=self.lr, betas=(0.9, 0.98), eps=1e-8)
        elif self.optimizer == "adamw":
            optimizer = torch.optim.AdamW(module.parameters(), lr=self.lr)#, betas=(0.9, 0.98), eps=1e-8)
            
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
            total_loss = 0
            total_correct = 0
            total_count = 0
            optimizer.zero_grad()
            
            train_iter = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc="train")
            for batch_idx, batch in train_iter:
                vid_frames = batch["imgs"].to(DEVICE, non_blocking=True)
                vid_labels = batch["label"].flatten().to(DEVICE, non_blocking=True)
                orig_vid_labels = vid_labels # vid_labels as int, before smoothing
                
                if mixup_fn is not None:
                    vid_frames, vid_labels = mixup_fn(vid_frames, vid_labels)
                
                logits = module(vid_frames)
                loss = criterion(logits, vid_labels) / self.accumulation_steps
                
                total_loss += loss.item() * len(vid_frames)
                total_correct += (logits.argmax(dim=1) == orig_vid_labels).sum()
                total_count += len(vid_frames)
                train_iter.set_postfix_str(f"train acc: {total_correct / total_count:.2f}, train loss: {total_loss / total_count:.2f}, total count: {total_count}")
                
                loss.backward()
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    optimizer.step()                 
                    optimizer.zero_grad()
                    
                torch.cuda.synchronize()
                
            scheduler.step()
            
            # Check val-tuning performance
            if val_tuning_dataloader is not None:
                total_val_correct = 0
                total_val_count = 0
                with torch.no_grad():
                    for batch_idx, batch in tqdm(enumerate(val_tuning_dataloader), total=len(val_tuning_dataloader), desc="val"):
                        vid_frames = batch["imgs"].to(DEVICE, non_blocking=True)
                        vid_labels = batch["label"].flatten().to(DEVICE, non_blocking=True)
                        
                        logits = module(vid_frames)
                            
                        total_val_correct += (logits.argmax(dim=1) == vid_labels).sum()
                        total_val_count += len(vid_frames)
                    
                val_acc = total_val_correct / total_val_count
                if val_tuning_best_acc is None or val_acc >= val_tuning_best_acc:
                    val_tuning_best_acc = val_acc
                    val_tuning_best_model_state = deepcopy(module.state_dict())
                print(f"Epoch {epoch_idx:5}: Support Acc = {total_correct / total_count:5.3f}, Val-Tune Acc = {val_acc:5.3f}, Loss = {total_loss / total_count:5.3f}")
            else:
                print(f"Epoch {epoch_idx:5}: Support Acc = {total_correct / total_count:5.3f}, Loss = {total_loss / total_count:5.3f}")
        
        """
        Compute query-set outputs with trained module
        """        
        # Reload best val-tuning model state
        if val_tuning_best_model_state is not None:
            module.load_state_dict(val_tuning_best_model_state)
        
        query_logits = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(query_dataloader):
                vid_frames = batch["imgs"].to(DEVICE, non_blocking=True)
                query_logits.append(module(vid_frames))
        query_logits = torch.cat(query_logits, dim=0)
        
        query_predictions = torch.argmax(query_logits, dim=1).cpu.numpy()
        return query_predictions



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
        for i, resblock in enumerate(self.vlm.model.transformer.resblocks):
            if i < len(self.text_context):
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
            if i < len(self.vision_context):
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