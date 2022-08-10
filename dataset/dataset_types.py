import itertools
from multiprocessing.sharedctypes import Value
import torch
import os
import numpy as np


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
        
        
        
'''
Few-Shot task dataset for sampling few-shot tasks.
Task sampling algorithm matches that used in FSL-Video.
Each element is a tuple with two elements:
    1.  An array of video paths,
        with shape (n_way, n_support + n_query)
    2.  An array of names for the categories used in the task
        with shape (n_way,)
'''
class FewShotTaskDataset(torch.utils.data.IterableDataset):
    '''
    Args:
        data_dict ({str -> [str]}): Dictionary from class names to lists of video paths in that class.
        n_episodes (int):           Number of few-shot tasks/datapoints that can be sampled from this dataset. This sets the dataset instance's length.
        n_way (int):                Number of categories given in each few-shot task/datapoint sampled.
        n_support (int):            Number of example videos for each category in each few-shot task/datapoint sampled.
        n_query (int):              Number of videos a model must predict for each category in each few-shot task/datapoint sampled.
    '''
    def __init__(self, data_dict: dict, n_episodes: int, n_way: int, n_support: int, n_query: int) -> None:
        super().__init__()
        
        self.category_names = list(name for name, vids in data_dict.items() if len(vids) >= n_support + n_query)
        self.category_indices = np.arange(len(self.category_names))
        
        if len(self.category_names) < n_way:
            raise ValueError(f"Only {len(self.category_names)} categories have enough videos (>= {n_support + n_query}) to be sampled for few-shot tasks. Needs at least {n_way} categories for {n_way}-way few-shot tasks.")
        
        self.category_dataloaders = [
            torch.utils.data.DataLoader(data_dict[category], batch_size=n_support + n_query, shuffle=True)
            for category in self.category_names
        ]
        
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
        
