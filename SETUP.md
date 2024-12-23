# General setup
Working for all VLMs as of 2023-04-03
(TODO: Add instructions for adding repositories)
```
git clone <PATH-TO-URL>
git submodule update --init
```
# CLIP installs
### NOTE: Needs to be python3.8 for huggingface compatibility
```
conda create --name vlm_env python=3.8 -y
conda activate vlm_env
```

# Install MOMA 
### (assumes you are in the same directory as this file)
```
cd dataset/moma
pip install -e .
pip install jsbeautifier==1.14.4
```

# Install pytorch
The main dependency here is compatibility with the transformers and tokenizers version below.  Otherwise, any version of torch or cuda should work.
### For V100 GPUs:
`conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch -y`
### For A100 GPUs:
`pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html`
### Alternative compatibility that has worked on some setupts
`conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia -y`

# More requirements:
```
conda install -c huggingface transformers==4.14.1 tokenizers==0.10.3 -y
conda install pandas -y
pip install decord ftfy tqdm regex filelock
conda install ipykernel ipywidgets matplotlib -y
pip install pytorchvideo scikit-optimize patool yacs mmcv-full
```

# MILES VLM installs
```
pip install psutil humanize scipy einops dominate sacred neptune-contrib opencv-python ftfy timm==0.4.5
OPTIONAL: mkdir MILES/pretrained
OPTIONAL: use wget or some other tool MILES/pretrained/MILES.pth <-- https://connecthkuhk-my.sharepoint.com/:u:/g/personal/yuyingge_connect_hku_hk/EewsJ8SvaetNjHBnaopKelkBpIhyARHKoHAFkHm9uAZhGA?e=K6XXEI
```

# VideoCLIP installs (optional)
```
cd video_clip
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install -e .
export MKL_THREADING_LAYER=GNU
cd ../..
cd video_clip
git clone https://github.com/zanedurante/MMPT_updated # updated version of VideoCLIP repo
cd MMPT_updated
pip install -e .
cd ../..
```


# install pre-trained video and text tokenizers
```
cd video_clip/MMPT_updated
mkdir pretrained_models
cd pretrained_models
wget https://www.rocq.inria.fr/cluster-willow/amiech/howto100m/s3d_howto100m.pth
wget https://www.rocq.inria.fr/cluster-willow/amiech/howto100m/s3d_dict.npy
cd ..

# Dry run to get config files
python locallaunch.py projects/retri/videoclip.yaml --dryrun

# install videoclip
mkdir -p runs/retri/videoclip
cd runs/retri/videoclip
wget https://dl.fbaipublicfiles.com/MMPT/retri/videoclip/checkpoint_best.pt
cd ../../../../..
```
