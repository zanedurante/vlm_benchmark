import os, sys
from typing import Optional
from transformers import AutoModel

import torch
import torch.nn as nn
import torch.nn.functional as F



FILE_DIR = os.path.dirname(os.path.realpath(__file__))
REPO_DIR = os.path.join(FILE_DIR, "MCQ")
sys.path.append(REPO_DIR)
from .MCQ.MILES.model.model_MILES import MILES
from .MCQ.MILES.model.video_encoder_MILES import SpaceTimeTransformer
from .MCQ.MILES.utils.util import state_dict_data_parallel_fix


class SpaceTimeTransformer_ExposedTokens(SpaceTimeTransformer):
    def forward_features_to_tokens(self, x: torch.Tensor) -> torch.Tensor:
        b, curr_frames, channels, _, _ = x.shape
        
        # If given more frames than trained on (set at init), pass through patch embedding in self.num_frames-size batches
        x_list = []
        for f in range(0, curr_frames, self.num_frames):
            f_width = min(self.num_frames, curr_frames - f)
            x_list.append(self.patch_embed(x[:, f : f + f_width]).reshape(b, f_width, -1, self.patch_embed.embed_dim))
        x = torch.concat(x_list, dim=1)
        x = x.flatten(2).transpose(2, 1)
        x = x.reshape(b, -1, self.patch_embed.embed_dim)
        
        # Not from original implementation: Return tokens in shape (batch, frame, patch, embed)
        x = x.reshape(b, curr_frames, -1, self.patch_embed.embed_dim)
        
        return x
    
    def forward_features_from_tokens(self, x: torch.Tensor, prompt_tokens: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Portion of SpaceTimeTransformer.forward_features() which occurs after frames are converted into patch embeddings / tokens.

        Args:
            x (torch.Tensor):                                   Patch token embeddings. Shape (batch, frame, patch, embed_dim)
            prompt_tokens (Optional[torch.Tensor], optional):   Optional token embeddings which will be inserted between [CLS] token
                                                                and video tokens. Shape = (prompt_tokens, token_dim). Defaults to None.

        Returns:
            torch.Tensor: SpaceTimeTransformer output.
        """
        BF = x.shape[0]
        curr_frames = x.shape[1] # Not original: Compute number of frames, so we can interpolate temporal embeddings if there are more than 4 frames
        
        # Not from original implementation: Convert from tokens received in shape (batch, frame, patch, embed)
        x = x.reshape(x.shape[0], -1, self.patch_embed.embed_dim)
        
        cls_tokens = self.cls_token.expand(BF, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        
        # positional embed needs to be tiled for each frame (this does [1,2,3] --> [1,2,3,1,2,3]...)
        cls_embed = self.pos_embed[:, 0, :].unsqueeze(1)
        tile_pos_embed = self.pos_embed[:, 1:, :].repeat(1, curr_frames, 1)
        
        # Not from original implementation: interpolate temporal embeddings if we are seeing more than the training number of frames (4)
        # This is mentioned in the MILES paper as being used for action recognition (4 frames training to 16 frames 0-shot testing), but isn't in repo
        if curr_frames == self.num_frames:
            temporal_embed = self.temporal_embed
        else:
            temporal_embed = F.interpolate(self.temporal_embed.unsqueeze(0), size=(curr_frames, self.patch_embed.embed_dim), mode="bilinear")[0]
            
        print(f"orig: {self.temporal_embed.shape}\nnew: {temporal_embed.shape}")
        print(f"orig: {self.temporal_embed[0, :, 0]}\nnew: {temporal_embed[0, :, 0]}")
        
        # temporal embed needs to be repeated within each frame (this does [1,2,3] --> [1,1,1,2,2,2,3,3,3]...)
        tile_temporal_embed = temporal_embed.repeat_interleave(self.patches_per_frame, 1)
        
        total_pos_embed = tile_pos_embed + tile_temporal_embed
        total_pos_embed = torch.cat([cls_embed, total_pos_embed], dim=1)

        curr_patches = x.shape[1]
        x = x + total_pos_embed[:, :curr_patches]
        
        # Not from original implementation: Add prompt tokens (as a perturbation on top of repeated CLS tokens)
        if prompt_tokens is not None:
            prompt_tokens = prompt_tokens.unsqueeze(0).expand(BF, -1, -1) + self.cls_token
            x = torch.cat(
                [x[:, :1], prompt_tokens, x[:, 1:]],
                dim=1
            )
            curr_patches = x.shape[1]
        
        x = self.pos_drop(x)
        n = self.patches_per_frame
        f = int((curr_patches - 1) / self.patches_per_frame) # Not from original implementation. Determine frame count from current patch count
        assert (curr_patches - 1) % self.patches_per_frame == 0, "Non-CLS Visual patch embeddings must be reshapable into temporal and spatial dimensions"

        for blk in self.blocks:
            x = blk(x, self.einops_from_space, self.einops_to_space, self.einops_from_time,
                    self.einops_to_time,
                    time_n=n, space_f=f)

        x = self.norm(x)[:, 0]
        x = self.pre_logits(x)

        return x
    
    def forward_to_tokens(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features_to_tokens(x)
    
    def forward_from_tokens(self, x: torch.Tensor, prompt_tokens: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Portion of SpaceTimeTransformer.forward() which occurs after frames are converted into patch embeddings / tokens.

        Args:
            x (torch.Tensor):                                   Patch token embeddings. Shape (batch, frame, patch, embed_dim)
            prompt_tokens (Optional[torch.Tensor], optional):   Optional token embeddings which will be inserted between [CLS] token
                                                                and video tokens. Shape = (prompt_tokens, token_dim). Defaults to None.

        Returns:
            torch.Tensor: SpaceTimeTransformer output.
        """
        x = self.forward_features_from_tokens(x, prompt_tokens)
        x = self.head(x)
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_to_tokens(x)
        x = self.forward_from_tokens(x)
        return x
    
    
    
class MILES_ExposedTokens(MILES):
    def __init__(self,
                 video_params,
                 text_params,
                 projection_dim=256,
                 load_checkpoint=None,
                 projection='minimal'):
        super().__init__(video_params, text_params, projection_dim, load_checkpoint, projection)

        self.video_params = video_params
        self.text_params = text_params
        if not text_params['pretrained']:
            raise NotImplementedError("Huggingface text models require pretrained init.")

        self.text_model = AutoModel.from_pretrained(text_params['model'])
        self.text_model.train()
         
        pretrained = video_params['pretrained']
        num_frames = video_params.get('num_frames', 4)
        arch_config = video_params.get('arch_config', 'base_patch16_224_temporal')
        vit_init = video_params.get('vit_init', 'imagenet-21k')
        if arch_config == 'base_patch16_224_temporal':
            model = SpaceTimeTransformer_ExposedTokens(num_frames=num_frames)
        else:
            raise NotImplementedError

        model.head = nn.Identity()
        model.pre_logits = nn.Identity()
        ftr_dim = model.embed_dim
        self.video_model = model

        # for backwards compatibility (old models)
        self.video_model.fc = nn.Identity()

        # Project to a common embedding
        if projection == 'minimal':
            text_proj = nn.Sequential(nn.ReLU(),
                                     nn.Linear(self.text_model.config.hidden_size, projection_dim),
                                     )

            vid_proj = nn.Sequential(
                nn.Linear(ftr_dim, projection_dim)
            )
        self.text_proj = text_proj
        self.vid_proj = vid_proj

        if load_checkpoint not in ["", None]:
            checkpoint = torch.load(load_checkpoint)
            state_dict = checkpoint['state_dict']
            new_state_dict = state_dict_data_parallel_fix(state_dict, self.state_dict())
            self.load_state_dict(new_state_dict, strict=False)
        
        
        
    def compute_video_to_tokens(self, video_data: torch.Tensor) -> torch.Tensor:
        return self.video_model.forward_to_tokens(video_data)
    
    def compute_video_from_tokens(self, video_tokens: torch.Tensor, prompt_tokens: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Portion of MILES.compute_video() which occurs after frames are converted into patch embeddings / tokens.

        Args:
            x (torch.Tensor):                                   Patch token embeddings. Shape (batch, frame, patch, embed_dim)
            prompt_tokens (Optional[torch.Tensor], optional):   Optional token embeddings which will be inserted between [CLS] token
                                                                and video tokens. Shape = (prompt_tokens, token_dim). Defaults to None.

        Returns:
            torch.Tensor: SpaceTimeTransformer output.
        """
        video_embeds = self.video_model.forward_from_tokens(video_tokens, prompt_tokens)
        video_embeds = self.vid_proj(video_embeds)
        return video_embeds
        
    def compute_video(self, video_data: torch.Tensor) -> torch.Tensor:
        video_tokens = self.compute_video_to_tokens(video_data)
        video_embeds = self.compute_video_from_tokens(video_tokens)
        return video_embeds