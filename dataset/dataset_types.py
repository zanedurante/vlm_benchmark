import itertools
from typing import Optional
import numpy as np
import torch


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
Each element is a tuple with four elements, outlining a few-shot classification task:
    1.  An array of names for the categories used in the task
        with shape (n_way,)
    2.  An array of support video paths for each task category
        with shape (n_way, n_support). Can be None for zero-shot.
    3.  A 1D array of query video paths over all task categories.
        If n_query is not None, this will have length n_way * n_query.
        If n_query is None, this will contain all non-support videos
        in all categories in the sampled task
    4.  A 1D array of query video labels. Same shape as query video paths.
'''
class FewShotTaskDataset(torch.utils.data.IterableDataset):
    '''
    Args:
        data_dict ({str -> [str]}): Dictionary from class names to lists of video paths in that class.
        n_episodes (int):           Number of few-shot tasks/datapoints that can be sampled from this dataset. This sets the dataset instance's length.
        n_way (int):                Number of categories given in each few-shot task/datapoint sampled.
        n_support (int):            Number of example videos for each category in each few-shot task/datapoint sampled.
        n_query (Optional[int]):    Number of videos a model must predict for each category in each few-shot task/datapoint sampled. If None, uses all videos not in support set.
    '''
    def __init__(self, data_dict: dict, n_episodes: int, n_way: int, n_support: int, n_query: Optional[int]) -> None:
        super().__init__()
        
        self.n_episodes = n_episodes
        self.n_way = n_way
        self.n_support = n_support
        self.n_query = n_query
        
        min_category_vids = n_support + (n_query or 1)
        self.category_names = list(name for name, vids in data_dict.items() if len(vids) >= min_category_vids)
        self.category_indices = np.arange(len(self.category_names))
        
        if len(self.category_names) < n_way:
            raise ValueError(f"Only {len(self.category_names)} categories have enough videos (>= {min_category_vids}) to be sampled for few-shot tasks. Needs at least {n_way} categories for {n_way}-way few-shot tasks.")
        
        # Store videos for each valid category
        self.category_videos = [
            np.array(data_dict[category])
            for category in self.category_names
        ]
        
        # Store the number of videos to sample for each category (support + query)
        if n_query is None:
            self.sample_sizes = [
                len(data_dict[category])
                for category in self.category_names
            ]
        else:
            self.sample_sizes = [n_support + n_query] * len(self.category_names)
        
    def __iter__(self):
        for i in range(self.n_episodes):
            # Select categories
            sampled_categories = np.random.choice(self.category_indices, self.n_way, replace=False)
            
            # Collect category videos
            if self.n_support > 0:
                support_videos = [] # Will construct np array of shape (n_way, n_support)
            else:
                support_videos = None
            query_videos = [] # Will construct 1D np array
            query_labels = [] # Will construct 1D np array
            for cat_label, cat_ind in enumerate(sampled_categories):
                sampled_videos = np.random.choice(self.category_videos[cat_ind], size=self.sample_sizes[cat_ind], replace=False)
                
                sampled_queries = sampled_videos[self.n_support:]
                query_videos += sampled_queries.tolist()
                query_labels += [cat_label] * len(sampled_queries)
                
                if support_videos is not None:
                    sampled_supports = sampled_videos[:self.n_support]
                    support_videos.append(sampled_supports)
                
            query_videos = np.array(query_videos)
            query_labels = np.array(query_labels)
            if support_videos is not None:
                support_videos = np.array(support_videos)
            
            # Collect category text
            sampled_category_names = [
                self.category_names[category_index]
                for category_index in sampled_categories
            ]
            sampled_category_names = np.array(sampled_category_names)
            
            yield sampled_category_names, support_videos, query_videos, query_labels
            
    def __len__(self):
        return self.n_episodes
        
