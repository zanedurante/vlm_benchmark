import itertools
from typing import Optional
import numpy as np
import contextlib
import torch

from .dataset_handler import DatasetHandler

        
        
        
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
        query_dataset (DatasetHandler):     Dataset from which to draw query videos.
        support_dataset (DatasetHandler):   Dataset from which to draw support videos. Can be the same as query_dataset, in which case queries are ensured not to overlap with support
        n_episodes (int):                   Number of few-shot tasks/datapoints that can be sampled from this dataset. This sets the dataset instance's length.
        n_way (int):                        Number of categories given in each few-shot task/datapoint sampled.
        n_support (int):                    Number of example videos for each category in each few-shot task/datapoint sampled.
        n_query (Optional[int]):            Number of videos a model must predict for each category in each few-shot task/datapoint sampled. If None, uses all videos not in support set.
    '''
    def __init__(self, query_dataset: DatasetHandler, support_dataset: DatasetHandler, n_episodes: int, n_way: int, n_support: int, n_query: Optional[int]) -> None:
        super().__init__()
        
        self.n_episodes = n_episodes
        self.n_way = n_way
        self.n_support = n_support
        self.n_query = n_query
        
        self.same_dataset = (query_dataset.id() == support_dataset.id())
        
        if self.same_dataset:
            min_category_vids = n_support + (n_query or 1)
            self.category_names = [name for name, vids in query_dataset.data_dict.items() if len(vids) >= min_category_vids]
        else:
            min_query_vids = n_query or 1
            min_support_vids = n_support
            self.category_names = [
                name for name in query_dataset.data_dict
                if name in support_dataset.data_dict and len(query_dataset.data_dict[name]) >= min_query_vids and len(support_dataset.data_dict[name]) >= min_support_vids
            ]
        
        if len(self.category_names) < n_way:
            raise ValueError(f"Only {len(self.category_names)} valid categories for few-shot tasks with the given parameters. Needs at least {n_way} categories for {n_way}-way few-shot tasks.")
        
        self.category_indices = np.arange(len(self.category_names))
        
        # Store videos for each valid category
        self.query_videos = [
            np.array(query_dataset.data_dict[category])
            for category in self.category_names
        ]
        
        if self.same_dataset:
            self.support_videos = None
        else:
            self.support_videos = [
                np.array(support_dataset.data_dict[category])
                for category in self.category_names
            ]
            
        # Set np random seed
        # TODO: Figure out better way
        # TEMPORARY!!
        #np.random.seed(0)
        # Local random number generator with fixed seed across runs
        self.rng = np.random.default_rng(0)
        
    def __iter__(self):
        for i in range(self.n_episodes):
            # Select categories
            sampled_categories = self.rng.choice(self.category_indices, self.n_way, replace=False)
            
            # Collect category videos
            if self.n_support > 0:
                support_videos = [] # Will construct np array of shape (n_way, n_support)
            else:
                support_videos = None
            query_videos = [] # Will construct 1D np array
            query_labels = [] # Will construct 1D np array
            for cat_label, cat_ind in enumerate(sampled_categories):
                if self.same_dataset:
                    sample_size = self.n_support + (self.n_query or 1)
                    sampled_videos = self.rng.choice(self.query_videos[cat_ind], size=sample_size, replace=False)
                
                    sampled_queries = sampled_videos[self.n_support:]
                    query_videos += sampled_queries.tolist()
                    query_labels += [cat_label] * len(sampled_queries)
                
                    if support_videos is not None:
                        sampled_supports = sampled_videos[:self.n_support]
                        support_videos.append(sampled_supports)
                else:
                    sampled_queries = self.rng.choice(self.query_videos[cat_ind], size=(self.n_query or len(self.query_videos[cat_ind])), replace=False)
                    query_videos += sampled_queries.tolist()
                    query_labels += [cat_label] * len(sampled_queries)
                    
                    if support_videos is not None:
                        sampled_supports = self.rng.choice(self.support_videos[cat_ind], size=self.n_support, replace=False)
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