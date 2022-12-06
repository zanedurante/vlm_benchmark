# vlm_benchmark
Code for benchmarking contrastive VLMs on zero and few-shot activity recognition. 

## Current citation for the project
```
@InProceedings{durante2023cona,
  title={CoNa: Context-Name Tuning for Few-shot Activity Recognition},
  author={Durante, Zane and Harries, Robathan and Luo, Zelun and Sun, Adam and Agarwal, Pratyush and Adeli, Ehsan and Fei-Fei, Li},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={\noop{2023} in submission}
}
```

## Project Roadmap (before CVPR deadline):
- [x] Create unified framework for contrastive VLMs
- [x] Create caching system for faster embedding retrieval
- [x] CLIP Model
- [x] MILES 
- [x] VideoCLIP
- [x] UNIVL
- [x] VTTTwins
- [x] Implement general VL-prototype classifier
- [x] Implement general CoOp classifier
- [ ] Implement general Name-Optimizer classifier
- [ ] Add OpenAI clip implementation (instead of huggingface)

=========================================================================

## Project Roadmap (after CVPR deadline):
- [ ] Frozen-in-time
- [ ] Add linear probe classifier to the repo
- [ ] Update README
- [ ] Create repo setup instructions
- [ ] Create a quickstart guide 
- [ ] Merge into single conda environment
- [ ] Test on multiple GCP instances to ensure setup works easily
- [ ] Allow for easy replication of our results in the paper
- [ ] Add link to paper in README


## Future plans (unclear):
- [ ] Add caching support for video augmentations -- is this feasible by using RandAugment + RandomResizedCrop (only four added hyperparameters)?
- [ ] Add video-only models using similar framework
- [ ] Add non-contrastive VLMs
- [ ] Create video model training repo with easy interfacing with this repo (allows for full-finetuning)
