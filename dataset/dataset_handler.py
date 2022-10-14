import os, sys
from typing import Optional
import json, itertools
import numpy as np
from tqdm.autonotebook import tqdm

from SimilarityVLM import SimilarityVLM
from .dataset_types import SequentialVideoDataset, SequentialCategoryNameDataset, FewShotTaskDataset


'''
Handler which loads the information for all supported datasets, and can
produce various formats of iterable datasets for testing.

TODO: Remove moma repo as submodule, instead add instructions to clone it (anywhere) and install it into each VLM environment.
'''


FILE_DIR = os.path.dirname(os.path.realpath(__file__))
MOMA_REPO = os.path.join(FILE_DIR, "moma")

KINETICS_100_DIR = "/home/datasets/kinetics_100"
SMSM_DIR = "/home/datasets/smsm_cmn"
MOMA_DIR = "/home/datasets/moma"

class DatasetHandler:
    def __init__(self, name: str, split: str = "val", class_limit: Optional[int] = None):
        self.name = name
        self.split = split
        self.class_limit = class_limit
        
        if split not in ["train", "val", "test", "all"]:
            raise ValueError(f"Invalid dataset split: {split}")
        
        if class_limit is not None and class_limit <= 0:
            raise ValueError(f"Class limit must be positive or None. Got {class_limit}.")
        
        '''
        Populate self.data_dict.
            Keys are category names
            Values are lists of all video paths associated with that category name.
        '''
        self.data_dict = {}
        
        if name in ["kinetics_100", "smsm"]:
            # Both of these datasets come from the FSL-Video repo, and both use the splits from CMN: https://github.com/ffmpbgrnn/CMN
            # Both datasets are stored in the same format. One folder for each category (labeled <id>.<description>), containing video files.
            if name == "kinetics_100":
                dataset_dir = KINETICS_100_DIR
            else:
                dataset_dir = SMSM_DIR
            
            cls_folder_names = [f for f in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, f))]
            cls_folder_names.sort()
            
            if split == "train":
                class_indices = range(0, 64)
            elif split == "val":
                class_indices = range(64, 76)
            elif split == "test":
                class_indices = range(76, len(cls_folder_names))
            elif split == "all":
                class_indices = range(0, len(cls_folder_names))
                
            for i in class_indices:
                cls_folder_name = cls_folder_names[i]
                category_name = cls_folder_name.split(".")[-1].replace("_", " ").lower()
                
                cls_folder_path = os.path.join(dataset_dir, cls_folder_name)
                category_video_paths = [
                    os.path.join(cls_folder_path, f) for f in os.listdir(cls_folder_path)
                    if os.path.isfile(os.path.join(cls_folder_path, f)) and f[0] != "."
                ]
                
                self.data_dict[category_name] = category_video_paths
        
        elif name == "moma_act":
            sys.path.append(MOMA_REPO)
            from .moma.momaapi.moma import MOMA
            
            moma = MOMA(MOMA_DIR, paradigm="few-shot")
            cids = moma.get_cids(kind="act", threshold=1, split=split)
            category_names = moma.get_cnames(cids_act=cids)
            for category_name in category_names:
                ids = moma.get_ids_act(split=split if split != "all" else None, cnames_act=[category_name])
                category_video_paths = moma.get_paths(ids_act=ids)
                self.data_dict[category_name] = category_video_paths
        
        elif name == "moma_sact":
            sys.path.append(MOMA_REPO)
            from .moma.momaapi.moma import MOMA
            
            moma = MOMA(MOMA_DIR, paradigm="few-shot")
            cids = moma.get_cids(kind="sact", threshold=1, split=split)
            category_names = moma.get_cnames(cids_sact=cids)
            for category_name in category_names:
                ids = moma.get_ids_sact(split=split if split != "all" else None, cnames_sact=[category_name])
                category_video_paths = moma.get_paths(ids_sact=ids)
                self.data_dict[category_name] = category_video_paths
        
        else:
            raise ValueError(f"Unrecognized dataset name: {name}")
        
        # Artificially limit the number of classes after the fact
        if self.class_limit is not None and self.class_limit < len(self.data_dict):
            for extra_class in list(self.data_dict.keys())[self.class_limit:]:
                del self.data_dict[extra_class]
        
    def id(self) -> str:
        if self.class_limit is None:
            return f"{self.name}.{self.split}"
        else:
            return f"{self.name}.{self.split}.{self.class_limit}"
    
    def category_count(self) -> int:
        return len(self.data_dict)
    
    def video_count(self) -> int:
        return sum(len(vids) for vids in self.data_dict.values())
    
    def valid_for_few_shot(self, n_way: int, n_support: int, n_query: int) -> bool:
        """Check whether the dataset has enough categories with enough examples to successfully sample
        few-shot tasks with the given parameters.

        Args:
            n_way (int): _description_
            n_support (int): _description_
            n_query (int): _description_

        Returns:
            bool: _description_
        """
        valid_category_count = sum(
            len(videos) >= n_support + n_query
            for videos in self.data_dict.values()
        )
        return valid_category_count >= n_way
    
    
    def sequential_video(self) -> SequentialVideoDataset:
        return SequentialVideoDataset(self.data_dict)
    
    def sequential_category_name(self) -> SequentialCategoryNameDataset:
        return SequentialCategoryNameDataset(self.data_dict)
    
    def few_shot(self, n_episodes: int, n_way: int, n_support: int, n_query: int) -> FewShotTaskDataset:
        return FewShotTaskDataset(self.data_dict, n_episodes, n_way, n_support, n_query)
    
    def fill_cache(self, vlm: SimilarityVLM) -> None:
        """Triggers the given vlm to generate embeddings for every video and text referenced
        in this dataset split, saving the resulting cache both disk and mem.

        Args:
            vlm (SimilarityVLM): VLM to fill the cache for
        """
        
        video_dataset = self.sequential_video()
        for i, vid_path in enumerate(tqdm(video_dataset, leave=False)):
            vlm.get_video_embeds(vid_path)
                
        text_dataset = self.sequential_category_name()
        for i, text in enumerate(tqdm(text_dataset, leave=False)):
            vlm.get_text_embeds(text)
            
    def export_embeddings(self, vlm: SimilarityVLM, save_dir_path: str) -> None:
        """Computes embeddings for each text and video in the dataset, saving them
        as numpy arrays in .npy format.
        Output Files (in <save_dir_path>):
            category_names.npy:             1D array of text names for each category. Category index
                                            is consistent across all files.
            category_name_embeddings.npy:   2D array of text embeddings for each category. Dim 1 is category index,
                                            dim 2 is embedding dim.
            video_category_indices.npy:     1D array of category indices (corresponding to category_names.npy) for
                                            each video path/embedding.
            video_paths.npy:                1D array of video paths for each video.
            video_embeddings.npy:           2D array of video embeddings for each video.
                                            Dim 1 is video index, dim 32 is embedding dim.
            vlm_info.json:                  Class and parameters for the VLM instance used.

        Args:
            vlm (SimilarityVLM): _description_
            save_dir_path (str): _description_
        """
        self.fill_cache(vlm)
        
        os.makedirs(save_dir_path, exist_ok=True)
        
        category_names = np.array(list(self.data_dict.keys()))
        np.save(os.path.join(save_dir_path, "category_names.npy"), category_names)
        
        category_name_embeddings = np.array([
            vlm.get_text_embeds(name)
            for name in category_names
        ])
        np.save(os.path.join(save_dir_path, "category_name_embeddings.npy"), category_name_embeddings)
        
        video_category_indices = []
        video_paths = []
        video_embeddings = []
        for i, name in enumerate(category_names):
            video_category_indices += [i] * len(self.data_dict[name])
            video_paths += self.data_dict[name]
            video_embeddings += [
                vlm.get_video_embeds(path)
                for path in self.data_dict[name]
            ]
        video_category_indices = np.array(video_category_indices)
        video_paths = np.array(video_paths)
        video_embeddings = np.array(video_embeddings)
        
        np.save(os.path.join(save_dir_path, "video_category_indices.npy"), video_category_indices)
        np.save(os.path.join(save_dir_path, "video_paths.npy"), video_paths)
        np.save(os.path.join(save_dir_path, "video_embeddings.npy"), video_embeddings)
        
        vlm_info_dict = vlm.params()
        vlm_info_dict["class"] = vlm.__class__.__name__
        with open(os.path.join(save_dir_path, "vlm_info.json"), "w") as fp:
            json.dump(vlm_info_dict, fp, indent=2)