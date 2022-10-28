import os.path as osp
from pytorchvideo.models.head import create_vit_basic_head
from pytorchvideo.models.vision_transformers import create_multiscale_vision_transformers
import torch
import torch.nn as nn
from torchvision.datasets.utils import download_url

from multi_head import MultiHead


def create_multi_head(in_features, out_features, seq_pool_type, dropout_rate, activation):
  heads = []
  for out_feature in out_features:
    heads.append(create_vit_basic_head(in_features=in_features,
                                       out_features=out_feature,
                                       seq_pool_type=seq_pool_type,
                                       dropout_rate=dropout_rate,
                                       activation=activation)
                 )
  multi_head = MultiHead(nn.ModuleList(heads))
  return multi_head


# Reference: https://github.com/facebookresearch/pytorchvideo/blob/main/pytorchvideo_trainer/pytorchvideo_trainer/conf/module/model/mvit_base_16x4.yaml
def get_mvit(cfg):
  net = create_multiscale_vision_transformers(
    spatial_size=224,
    temporal_size=16,
    dropout_rate_block=0.0,
    droppath_rate_block=0.2,
    embed_dim_mul=[[1, 2.0], [3, 2.0], [14, 2.0]],
    atten_head_mul=[[1, 2.0], [3, 2.0], [14, 2.0]],
    pool_q_stride_size=[[1, 1, 2, 2], [3, 1, 2, 2], [14, 1, 2, 2]],
    pool_kv_stride_size=None,
    pool_kv_stride_adaptive=[1, 8, 8],
    pool_kvq_kernel=[3, 3, 3],
    #head=None
    head=create_vit_basic_head if len(cfg.num_classes) == 1 else create_multi_head,
    head_dropout_rate=0.5,
    head_activation=None,
    head_num_classes=cfg.num_classes[0] if len(cfg.num_classes) == 1 else cfg.num_classes
  )

  if cfg.mode == 'from_scratch':
    print('Initializing randomly')

  elif cfg.mode == 'finetune':
    if cfg.weight == 'ckpt':
      print('Loading checkpoint')
      weight = torch.load(osp.join(cfg.dir_weights, cfg.rpath_ckpt))['state_dict']
      weight = {k.removeprefix('net.'): v for k, v in weight.items()}
      weight.pop('head.proj.weight')
      weight.pop('head.proj.bias')
      keys_missing, keys_unexpected = net.load_state_dict(weight, strict=False)
      assert len(keys_unexpected) == 0
      print(f'{keys_missing} will be trained from scratch')

    elif cfg.weight == 'pretrain':
      print('Loading Kinetics pre-trained weight')
      dir_pretrain = osp.join(cfg.dir_weights, 'pretrain')
      fname_pretrain = 'MVIT_B_16x4.pyth'
      if not osp.exists(osp.join(dir_pretrain, fname_pretrain)):
        download_url(f'https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/{fname_pretrain}', dir_pretrain)

      weight = torch.load(osp.join(dir_pretrain, fname_pretrain))['model_state']
      weight.pop('head.proj.weight')
      weight.pop('head.proj.bias')
      keys_missing, keys_unexpected = net.load_state_dict(weight, strict=False)
      assert len(keys_unexpected) == 0
      print(f'{keys_missing} will be trained from scratch')

    else:
      raise NotImplementedError

  else:
    raise NotImplementedError

  return net