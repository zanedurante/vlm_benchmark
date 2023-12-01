import os, sys
from typing import Optional
import json, itertools
import numpy as np
from tqdm.autonotebook import tqdm
import torch
from patoolib import extract_archive
from collections import defaultdict
import re

from SimilarityVLM import SimilarityVLM


'''
Handler which loads the information for all supported datasets, and can
produce various formats of iterable datasets for testing.

TODO: Remove moma repo as submodule, instead add instructions to clone it (anywhere) and install it into each VLM environment.
'''


FILE_DIR = os.path.dirname(os.path.realpath(__file__))
MOMA_REPO = os.path.join(FILE_DIR, "moma")

KINETICS_100_DIR = None
SMSM_DIR = None
MOMA_DIR = None
HMDB_51_DIR = "/next/u/rharries/vlm_benchmark.data/hmdb_51"
UCF_101_DIR = "/next/u/rharries/vlm_benchmark.data/ucf_101"
SSV2_DIR = "/next/u/rharries/vlm_benchmark.data/ssv2"
IADL_DIR = "/next/u/rharries/vlm_benchmark.data/InteractADL_egoview_actions_subclips_resized"
IADL_ACTIVITIES_DIR = "/next/u/rharries/vlm_benchmark.data/InteractADL_egoview_activities_subclips"


DEFAULT_MIN_TRAIN_VIDS = 16



'''
Simple dataset for filling video embedding caches.
This just iterates through all videos referenced in the given dataset split.
Each element is a single video, referenced as a file path.
'''
class SequentialVideoDataset(torch.utils.data.Dataset):
    '''
    Args:
        data_dict ({str -> [str]}): Dictionary from class names to lists of video paths in that class.
    '''
    def __init__(self, data_dict: dict) -> None:
        super().__init__()
        
        self.video_paths = list(itertools.chain(*data_dict.values()))
    
    def __getitem__(self, i):
        return self.video_paths[i]
    
    def __len__(self):
        return len(self.video_paths)
    

'''
Simple dataset for filling text embedding caches.
This just iterates through all videos referenced in the given dataset split.
'''
class SequentialCategoryNameDataset(torch.utils.data.Dataset):
    '''
    Args:
        data_dict ({str -> [str]}): Dictionary from class names to lists of video paths in that class.
    '''
    def __init__(self, data_dict: dict) -> None:
        super().__init__()
        
        self.category_names = list(data_dict.keys())
    
    def __getitem__(self, i):
        return self.category_names[i]
    
    def __len__(self):
        return len(self.category_names)



class DatasetHandler:
    def __init__(self, name: str, split: str = "val", split_type: str = "video", class_limit: Optional[int] = None, min_train_videos: int = DEFAULT_MIN_TRAIN_VIDS):
        self.name = name
        self.split = split
        self.split_type = split_type
        self.class_limit = class_limit
        self.min_train_videos = min_train_videos
        
        if split not in ["train", "val", "test", "all"]:
            raise ValueError(f"Invalid dataset split: {split}")
        
        if split_type not in ["class", "video"]:
            raise ValueError(f"Invalid split type: {split_type}")
        
        if class_limit is not None and class_limit <= 0:
            raise ValueError(f"Class limit must be positive or None. Got {class_limit}.")
        
        '''
        Populate self.data_dict.
            Keys are category names
            Values are lists of all video paths associated with that category name.
        '''
        self.data_dict = defaultdict(list)
        
        if name in ["kinetics_100", "smsm"]:
            # Both of these datasets come from the FSL-Video repo, and both use the splits from CMN: https://github.com/ffmpbgrnn/CMN
            # Both datasets are stored in the same format. One folder for each category (labeled <id>.<description>), containing video files.
            if name == "kinetics_100":
                dataset_dir = KINETICS_100_DIR
            else:
                dataset_dir = SMSM_DIR
            
            cls_folder_names = [f for f in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, f))]
            cls_folder_names.sort()
            
            if split_type == "class":
                if split == "train":
                    class_indices = range(0, 64)
                elif split == "val":
                    class_indices = range(64, 76)
                elif split == "test":
                    class_indices = range(76, len(cls_folder_names))
                elif split == "all":
                    class_indices = range(0, len(cls_folder_names))
            elif split_type == "video":
                class_indices = range(0, len(cls_folder_names))
                
                
            for i in class_indices:
                cls_folder_name = cls_folder_names[i]
                category_name = cls_folder_name.split(".")[-1].replace("_", " ").lower()
                
                cls_folder_path = os.path.join(dataset_dir, cls_folder_name)
                category_video_paths = [
                    os.path.join(cls_folder_path, f) for f in sorted(os.listdir(cls_folder_path))
                    if os.path.isfile(os.path.join(cls_folder_path, f)) and f[0] != "."
                ]
                
                if split_type == "class":
                    vid_indices = range(0, len(category_video_paths))
                elif split_type == "video":
                    train_len = int(0.7648 * len(category_video_paths))
                    val_len = int(0.1122 * len(category_video_paths))
                    if split == "train":
                        vid_indices = range(0, train_len)
                    elif split == "val":
                        vid_indices = range(train_len, train_len + val_len)
                    elif split == "test":
                        vid_indices = range(train_len + val_len, len(category_video_paths))
                    elif split == "all":
                        vid_indices = range(0, len(category_video_paths))
                
                self.data_dict[category_name] = [category_video_paths[j] for j in vid_indices]
        
        elif name == "moma_act":
            sys.path.append(MOMA_REPO)
            from .moma.momaapi.moma import MOMA
            
            moma = MOMA(MOMA_DIR, paradigm="few-shot" if split_type == "class" else "standard")
            cids = moma.get_cids(kind="act", threshold=1, split=split)
            category_names = moma.get_cnames(cids_act=cids)
            for category_name in category_names:
                ids = moma.get_ids_act(split=split if split != "all" else None, cnames_act=[category_name])
                category_video_paths = moma.get_paths(ids_act=ids)
                self.data_dict[category_name] = category_video_paths
        
        elif name == "moma_sact":
            sys.path.append(MOMA_REPO)
            from .moma.momaapi.moma import MOMA
            
            moma = MOMA(MOMA_DIR, paradigm="few-shot" if split_type == "class" else "standard")
            cids = moma.get_cids(kind="sact", threshold=1, split=split)
            category_names = moma.get_cnames(cids_sact=cids)
            for category_name in category_names:
                ids = moma.get_ids_sact(split=split if split != "all" else None, cnames_sact=[category_name])
                category_video_paths = moma.get_paths(ids_sact=ids)
                self.data_dict[category_name] = category_video_paths
                
        elif name == "hmdb_51":
            # Extract video and split info from original compressed formats, if necessary
            if not os.path.exists(os.path.join(HMDB_51_DIR, "data")):
                print("Extracting HMDB-51 videos from original format")
                extract_archive(os.path.join(HMDB_51_DIR, "hmdb51_org.rar"), outdir=os.path.join(HMDB_51_DIR, "data"))
                for file in os.listdir(os.path.join(HMDB_51_DIR, "data")):
                    if file.endswith(".rar"):
                        extract_archive(os.path.join(HMDB_51_DIR, "data", file), outdir=os.path.join(HMDB_51_DIR, "data"))
                        os.remove(os.path.join(HMDB_51_DIR, "data", file))       
            if not os.path.exists(os.path.join(HMDB_51_DIR, "splits")):
                print("Extracting HMDB-51 split information from original format")
                extract_archive(os.path.join(HMDB_51_DIR, "test_train_splits.rar"), outdir=HMDB_51_DIR)

                # Collect train/val/test relative video paths (for each split number)
                # Original format only has train and test splits, so train set is split randomly into train and val splits (50 and 20 videos per category respectively)
                train_val_split_generator = torch.Generator().manual_seed(42)
                train_paths = defaultdict(list)
                val_paths = defaultdict(list)
                test_paths = defaultdict(list)
                for file in os.listdir(os.path.join(HMDB_51_DIR, "testTrainMulti_7030_splits")):
                    split_number = int(file[-5])
                    category_dir = "_".join(file.split("_")[:-2])
                    category_train_val_paths = defaultdict(list)
                    category_test_paths = defaultdict(list)
                    with open(os.path.join(HMDB_51_DIR, "testTrainMulti_7030_splits", file), "r") as fp:
                        for line in fp.readlines():
                            line = line.split(" ")
                            vid_filename = line[0]
                            vid_type = line[1]
                            if vid_type == "0":
                                continue
                            elif vid_type == "1":
                                category_train_val_paths[split_number].append(f"{category_dir}/{vid_filename}")
                            elif vid_type == "2":
                                category_test_paths[split_number].append(f"{category_dir}/{vid_filename}")
                            else:
                                raise ValueError
                    for split_number, paths in category_train_val_paths.items():
                        train, val = torch.utils.data.random_split(paths, [50, 20], generator=train_val_split_generator)
                        train_paths[split_number] += train
                        val_paths[split_number] += val
                    for split_number, paths in category_test_paths.items():
                        test_paths[split_number] += paths
                            
                    os.remove(os.path.join(HMDB_51_DIR, "testTrainMulti_7030_splits", file))
                os.rmdir(os.path.join(HMDB_51_DIR, "testTrainMulti_7030_splits"))
                
                # Save train/test video paths to split files
                os.makedirs(os.path.join(HMDB_51_DIR, "splits"), exist_ok=True)
                for split_number, video_paths in train_paths.items():
                    with open(os.path.join(HMDB_51_DIR, "splits", f"train_split{split_number}.txt"), "w") as fp:
                        fp.write("\n".join(video_paths))
                for split_number, video_paths in val_paths.items():
                    with open(os.path.join(HMDB_51_DIR, "splits", f"val_split{split_number}.txt"), "w") as fp:
                        fp.write("\n".join(video_paths))
                for split_number, video_paths in test_paths.items():
                    with open(os.path.join(HMDB_51_DIR, "splits", f"test_split{split_number}.txt"), "w") as fp:
                        fp.write("\n".join(video_paths))
                        
            # Construct data_dict from split 1
            if split_type == "video":
                if split in ["train", "val", "test"]:
                    with open(os.path.join(HMDB_51_DIR, "splits", f"{split}_split1.txt"), "r") as fp:
                        for line in fp.readlines():
                            line = line.strip().split("/")
                            category_folder_name = line[0]
                            category_name = category_folder_name.replace("_", " ")
                            video_file = line[1]
                            self.data_dict[category_name].append(os.path.join(HMDB_51_DIR, "data", category_folder_name, video_file))
                elif split == "all":
                    for partial_split in ["train", "val", "test"]:
                        with open(os.path.join(HMDB_51_DIR, "splits", f"{partial_split}_split1.txt"), "r") as fp:
                            for line in fp.readlines():
                                line = line.strip().split("/")
                                category_folder_name = line[0]
                                category_name = category_folder_name.replace("_", " ")
                                video_file = line[1]
                                self.data_dict[category_name].append(os.path.join(HMDB_51_DIR, "data", category_folder_name, video_file))
                
            elif split_type == "class":
                # TODO
                raise NotImplementedError
            
        elif name == "ucf_101":
            # Extract video and split info from original compressed formats, if necessary
            if not os.path.exists(os.path.join(UCF_101_DIR, "data")):
                print("Extracting UCF-101 videos from original format")
                extract_archive(os.path.join(UCF_101_DIR, "UCF101.rar"), outdir=os.path.join(UCF_101_DIR))
                os.rename(os.path.join(UCF_101_DIR, "UCF-101"), os.path.join(UCF_101_DIR, "data"))
            if not os.path.exists(os.path.join(UCF_101_DIR, "splits")):
                print("Extracting UCF-101 split information from original format")
                extract_archive(os.path.join(UCF_101_DIR, "UCF101TrainTestSplits-RecognitionTask.zip"), outdir=UCF_101_DIR)

                # Collect train/val/test relative video paths (for each split number)
                # Original format only has train and test splits, so train set is split randomly into train and val splits (75%/25% per category)
                train_val_split_generator = torch.Generator().manual_seed(42)
                train_paths = defaultdict(list)
                val_paths = defaultdict(list)
                test_paths = defaultdict(list)
                for split_number in [1, 2, 3]:
                    # Train/Val Split - Group entries per category, then randomly split into train and val sets
                    train_val_paths_per_category = defaultdict(list)
                    with open(os.path.join(UCF_101_DIR, "ucfTrainTestlist", f"trainlist0{split_number}.txt"), "r") as fp:
                        for line in fp.readlines():
                            line = line.strip().split(" ")[0].split("/")
                            category_dir = line[0]
                            video_file = line[1]
                            train_val_paths_per_category[category_dir].append(f"{category_dir}/{video_file}")
                    for category_dir, paths in train_val_paths_per_category.items():
                        train_len = int(0.75 * len(paths))
                        val_len = len(paths) - train_len
                        train, val = torch.utils.data.random_split(paths, [train_len, val_len], generator=train_val_split_generator)
                        train_paths[split_number] += train
                        val_paths[split_number] += val
                    
                    # Test split
                    with open(os.path.join(UCF_101_DIR, "ucfTrainTestlist", f"testlist0{split_number}.txt"), "r") as fp:
                        for line in fp.readlines():
                            line = line.strip().split(" ")[0].split("/")
                            category_dir = line[0]
                            video_file = line[1]
                            test_paths[split_number].append(f"{category_dir}/{video_file}")
                            
                # Remove extracted split folder
                for file in os.listdir(os.path.join(UCF_101_DIR, "ucfTrainTestlist")):
                    os.remove(os.path.join(UCF_101_DIR, "ucfTrainTestlist", file))
                os.rmdir(os.path.join(UCF_101_DIR, "ucfTrainTestlist"))
                
                # Save formatted splits to new "splits" folder
                os.makedirs(os.path.join(UCF_101_DIR, "splits"), exist_ok=True)
                for split_number, video_paths in train_paths.items():
                    with open(os.path.join(UCF_101_DIR, "splits", f"train_split{split_number}.txt"), "w") as fp:
                        fp.write("\n".join(video_paths))
                for split_number, video_paths in val_paths.items():
                    with open(os.path.join(UCF_101_DIR, "splits", f"val_split{split_number}.txt"), "w") as fp:
                        fp.write("\n".join(video_paths))
                for split_number, video_paths in test_paths.items():
                    with open(os.path.join(UCF_101_DIR, "splits", f"test_split{split_number}.txt"), "w") as fp:
                        fp.write("\n".join(video_paths))
                        
            # Construct data_dict from split 1
            if split_type == "video":
                if split in ["train", "val", "test"]:
                    with open(os.path.join(UCF_101_DIR, "splits", f"{split}_split1.txt"), "r") as fp:
                        for line in fp.readlines():
                            line = line.strip().split("/")
                            category_folder_name = line[0]
                            category_name = " ".join(re.findall('[A-Z][^A-Z]*', category_folder_name)).lower()
                            video_file = line[1]
                            self.data_dict[category_name].append(os.path.join(UCF_101_DIR, "data", category_folder_name, video_file))
                elif split == "all":
                    for partial_split in ["train", "val", "test"]:
                        with open(os.path.join(UCF_101_DIR, "splits", f"{partial_split}_split1.txt"), "r") as fp:
                            for line in fp.readlines():
                                line = line.strip().split("/")
                                category_folder_name = line[0]
                                category_name = " ".join(re.findall('[A-Z][^A-Z]*', category_folder_name)).lower()
                                video_file = line[1]
                                self.data_dict[category_name].append(os.path.join(UCF_101_DIR, "data", category_folder_name, video_file))
                
            elif split_type == "class":
                # TODO
                raise NotImplementedError
        
        elif name == "ssv2":
            # To match ViFi-CLIP (which provides results on validation set), we ignore the official test set,
            # and instead use the official validation set as our test split.
            # The official train set is randomly split into train and val sets
            # This is also helpful since it avoids the official test set, which only includes 154 / 174 classes.
            if split_type == "video":
                if split in ["train", "val"]:
                    with open(os.path.join(SSV2_DIR, "labels", "train.json"), "r") as fp:
                        label_info = json.load(fp)
                    train_val_data = defaultdict(list)
                    for vid_info in label_info:
                        category_name = vid_info["template"].replace("[something]", "something").lower()
                        video_path = os.path.join(SSV2_DIR, "20bn-something-something-v2", f"{vid_info['id']}.webm")
                        train_val_data[category_name].append(video_path)
                    # Entries in label json are already randomly ordered, so our train/val split can just use that order
                    for category_name, video_paths in train_val_data.items():
                        num_train = int(0.75 * len(video_paths))
                        if split == "train":
                            self.data_dict[category_name] = video_paths[:num_train]
                        else:
                            self.data_dict[category_name] = video_paths[num_train:]
                elif split == "test":
                    with open(os.path.join(SSV2_DIR, "labels", "validation.json"), "r") as fp:
                        label_info = json.load(fp)
                    for vid_info in label_info:
                        category_name = vid_info["template"].replace("[something]", "something").lower()
                        video_path = os.path.join(SSV2_DIR, "20bn-something-something-v2", f"{vid_info['id']}.webm")
                        self.data_dict[category_name].append(video_path)
                elif split == "all":
                    raise NotImplementedError
                
            elif split_type == "class":
                raise NotImplementedError
        
        elif name == "iadl":
            if split_type == "video":
                if split in ["train", "val", "test"]:
                    with open(os.path.join(IADL_DIR, "splits", f"{split}.json"), "r") as fp:
                        self.data_dict = json.load(fp)
                    for key in list(self.data_dict.keys()):
                        if len(self.data_dict[key]) == 0:
                            del self.data_dict[key]
                elif split == "all":
                    raise NotImplementedError

                # Prepend base data dir to relative video paths
                for category, vids in self.data_dict.items():
                    for i in range(len(vids)):
                        vids[i] = os.path.join(IADL_DIR, vids[i])
                
            elif split_type == "class":
                raise NotImplementedError
            
        elif name == "iadl_activities":
            # Force no min_train_videos filtering (use as many categories as possible given number of shots in test)
            min_train_videos = 0

            if split_type == "video":
                if split in ["train", "val", "test"]:
                    with open(os.path.join(IADL_ACTIVITIES_DIR, "splits", f"{split}.json"), "r") as fp:
                        self.data_dict = json.load(fp)
                    for key in list(self.data_dict.keys()):
                        if len(self.data_dict[key]) == 0:
                            del self.data_dict[key]
                elif split == "all":
                    raise NotImplementedError

                # Prepend base data dir to relative video paths
                for category, vids in self.data_dict.items():
                    for i in range(len(vids)):
                        vids[i] = os.path.join(IADL_ACTIVITIES_DIR, vids[i])
                
            elif split_type == "class":
                raise NotImplementedError
        
        else:
            raise ValueError(f"Unrecognized dataset name: {name}")
        
        # Artificially limit the number of classes after the fact
        if self.class_limit is not None and self.class_limit < len(self.data_dict):
            for extra_class in list(self.data_dict.keys())[self.class_limit:]:
                del self.data_dict[extra_class]
                
        # Remove classes which have too few training examples
        # min_train_videos field only has an effect on split_type="video" datasets, where the classes are the same across splits
        if self.split_type == "video" and min_train_videos > 1:
            # TODO: Determine better way to make this consistent across splits
            if split == "train":
                for cat in list(self.data_dict.keys()):
                    if len(self.data_dict[cat]) < min_train_videos:
                        del self.data_dict[cat]
            else:
                train_dataset = DatasetHandler(name, split="train", split_type=split_type, class_limit=class_limit, min_train_videos=min_train_videos)
                for cat in list(self.data_dict.keys()):
                    if cat not in train_dataset.data_dict.keys():
                        del self.data_dict[cat]
        
    def id(self) -> str:
        if self.split_type == "class":
            out = f"{self.name}.c.{self.split}"
        else:
            out = f"{self.name}.v.{self.split}"
            
        # Only include extra info if these uncommon vars are non-default
        if self.min_train_videos != DEFAULT_MIN_TRAIN_VIDS or self.class_limit is not None:
            out += f".vidmin_{self.min_train_videos}"
            out += f".classmax_{self.class_limit}"
            
        return out
            
    def category_count(self) -> int:
        return len(self.data_dict)
    
    def video_count(self) -> int:
        return sum(len(vids) for vids in self.data_dict.values())
    
    
    def sequential_video(self) -> SequentialVideoDataset:
        return SequentialVideoDataset(self.data_dict)
    
    def sequential_category_name(self) -> SequentialCategoryNameDataset:
        return SequentialCategoryNameDataset(self.data_dict)
    
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
