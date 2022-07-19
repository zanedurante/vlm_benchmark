import torch
import os
import numpy as np


'''
Dataloaders which read dataset split files in the format of FSL-Video,
returning video datapoints as filesystem paths to a folder of ordered image frames.

NOTE: Category names are currently extracted from dataset folder structure, in which
each category is associated with a folder labeled as "<category_index + 1>.<category_name>"
'''

def category_name_from_video_path(video_path):
    category_folder_name = video_path.split("/")[-2]
    category_name = category_folder_name.split(".")[-1]
    category_name = category_name.replace("_", " ")
    return category_name


'''
Simple dataset for filling video embedding caches.
This just iterates through all videos referenced in the given dataset split file.
Each element is a single video, referenced as a path to a folder of ordered frames.
'''
class SequentialVideoDataset(torch.utils.data.Dataset):
    '''
    Args:
        data_split_filepath (str):  Path to the txt file corresponding to a split of the dataset.
    '''
    def __init__(self, data_split_filepath: str) -> None:
        super().__init__()
        
        with open(data_split_filepath, 'r') as fp:
            data_split = [x.strip().split(' ') for x in fp]
            
        self.video_frame_paths = [x[0] for x in data_split]
    
    def __getitem__(self, i):
        return self.video_frame_paths[i]
    
    def __len__(self):
        return len(self.video_frame_paths)
    

'''
Simple dataset for filling text embedding caches.
This just iterates through all videos referenced in the given
'''
class SequentialCategoryNameDataset(torch.utils.data.Dataset):
    '''
    Args:
        data_split_filepath (str):  Path to the txt file corresponding to a split of the dataset.
    '''
    def __init__(self, data_split_filepath: str) -> None:
        super().__init__()
        
        with open(data_split_filepath, 'r') as fp:
            data_split = [x.strip().split(' ') for x in fp]
        
        category_names_by_index = {}
        for (video_frames_path, frame_count, category_index) in data_split:
            category_index = int(category_index)
            if category_index not in category_names_by_index:
                category_names_by_index[category_index] = category_name_from_video_path(video_frames_path)
                
        self.category_names = list(category_names_by_index.items())
    
    def __getitem__(self, i):
        return self.category_names[i]
    
    def __len__(self):
        return len(self.category_names)
        
        
        
'''
Few-Shot task dataset for sampling few-shot tasks.
Task sampling algorithm matches that used in FSL-Video.
Each element is a tuple with two elements:
    1.  An array of video references (frame folder path),
        with shape (n_way, n_support + n_query)
    2.  An array of names for the categories used in the task
        with shape (n_way,)
'''
class FewShotTaskDataset(torch.utils.data.IterableDataset):
    '''
    Args:
        data_split_filepath (str):  Path to the split file, which specifies frame folder and class index for each video in one some split of the dataset.
        n_episodes (int):           Number of few-shot tasks/datapoints that can be sampled from this dataset. This sets the dataset instance's length.
        n_way (int):                Number of categories given in each few-shot task/datapoint sampled.
        n_support (int):            Number of example videos for each category in each few-shot task/datapoint sampled.
        n_query (int):              Number of videos a model must predict for each category in each few-shot task/datapoint sampled.
    '''
    def __init__(self, data_split_filepath: str, n_episodes: int, n_way: int, n_support: int, n_query: int) -> None:
        super().__init__()
        
        with open(data_split_filepath, 'r') as fp:
            data_split = [x.strip().split(' ') for x in fp]
        
        # Collect all videos for each category
        vid_paths_by_category = {}
        for (video_frames_path, frame_count, category_index) in data_split:
            category_index = int(category_index)
            if category_index not in vid_paths_by_category:
                vid_paths_by_category[category_index] = []
            vid_paths_by_category[category_index].append(video_frames_path)
        
        self.category_indices = np.unique(list(vid_paths_by_category.keys()))
        
        # Create video dataloaders (sampling n_support + n_query videos per request) for each category
        self.category_dataloaders = {
            i: torch.utils.data.DataLoader(vid_paths_by_category[i], batch_size=n_support + n_query, shuffle=True)
            for i in self.category_indices
        }
        
        # Collect all category names from folder structure
        self.category_names = {
            i: category_name_from_video_path(vid_paths_by_category[i][0])
            for i in self.category_indices
        }
        
        self.n_episodes = n_episodes
        self.n_way = n_way
        
    def __iter__(self):
        for i in range(self.n_episodes):
            # Select categories
            sampled_categories = np.random.choice(self.category_indices, self.n_way, replace=False)
            
            # Collect category videos
            sampled_videos_per_sampled_category = [
                next(iter(self.category_dataloaders[category_index]))
                for category_index in sampled_categories
            ]
            sampled_videos_per_sampled_category = np.array(sampled_videos_per_sampled_category)
            
            # Collect category text
            sampled_category_names = [
                self.category_names[category_index]
                for category_index in sampled_categories
            ]
            sampled_category_names = np.array(sampled_category_names)
            
            yield sampled_videos_per_sampled_category, sampled_category_names
            
    def __len__(self):
        return self.n_episodes
        
