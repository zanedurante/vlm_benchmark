import os
from typing import Optional
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import json

from SimilarityVLM import SimilarityVLM
from classifier.FewShotClassifier import FewShotClassifier
from dataset.dataset import FewShotTaskDataset, SequentialVideoDataset, SequentialCategoryNameDataset



'''
Class for running few-shot tests, saving the results, and facillitating result analysis
'''

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
TEST_RESULTS_PATH = os.path.join(FILE_DIR, "test_results.csv")


class FewShotTestHandler:
    def __init__(self):
        # Load results DataFrame
        if os.path.exists(TEST_RESULTS_PATH):
            self.results = pd.read_csv(TEST_RESULTS_PATH)
        else:
            self.results = pd.DataFrame()
            
    def fill_cache(self, vlm: SimilarityVLM, dataset_split_path: str) -> None:
        """Triggers the given vlm to generate embeddings for every video referenced
        in the given dataset split, saving the resulting cache.

        Args:
            vlm (SimilarityVLM): VLM to fill the cache of
            dataset_split_path (str): Dataset from which to select videos
        """
        
        video_dataset = SequentialVideoDataset(dataset_split_path)
        for i, vid_path in enumerate(tqdm(video_dataset)):
            if vid_path not in vlm.embed_cache:
                vlm.get_video_embeds(vid_path)
                
            # Save cache periodically in case process is interrupted
            if i % 25:
                vlm.save_cache()
                
        vlm.save_cache()
        
    
        
    def run_few_shot_test(self, classifier: FewShotClassifier, dataset_split_path: str,
                          n_way: int, n_support: int, n_query: int = 1, n_episodes: int = 1000,
                          ) -> None:
        """Runs the given few-shot test if it has not already been performed,
        saving the accuracy.

        Args:
            classifier (FewShotClassifier): Few-Shot video classifier built on top of an arbitrary SimilarityVLM
            dataset_split_path (str): Dataset split to build few-shot tasks from
            n_way (int): Number of categories per few-shot task
            n_support (int): Number of example videos per category per few-shot task
            n_query (int, optional): Number of videos predicted per category per few-shot task
            n_episodes (int, optional): Number of few-shot tasks to sample. Defaults to 1000.
        """
        
        # Skip test if it already exists
        if test_already_stored(self.results, classifier, dataset_split_path, n_way, n_support, n_query, n_episodes):
            return
        
        # Load dataset to generate tasks with the desired params
        dataset = FewShotTaskDataset(dataset_split_path, n_episodes, n_way, n_support, n_query)
        
        correct_predictions = 0
        total_queries = 0
        for vid_paths, category_names in tqdm(dataset, leave=False):
            
            query_vid_paths = vid_paths[:, n_support:]
            if n_support > 0:
                support_vid_paths = vid_paths[:, :n_support]
            else:
                support_vid_paths = None
                
            query_predictions = classifier.predict(category_names, support_vid_paths, query_vid_paths)
            
            correct_predictions += np.sum(query_predictions == np.arange(n_way)[:, None])
            total_queries += n_way * n_query
        
        accuracy = correct_predictions / total_queries
        
        # Add to test results and save
        self.results = append_test_result(self.results, classifier, dataset_split_path, n_way, n_support, n_query, n_episodes, accuracy)
        self.results.to_csv(TEST_RESULTS_PATH, index=False)
        
    
    
    
    
       
'''
Test Results DataFrame Utilities
'''

def dataframe_format(classifier: FewShotClassifier, dataset_split_path: str,
                            n_way: int, n_support: int, n_query: int, n_episodes: int,
                            accuracy: Optional[float] = None) -> dict:
    row = {
        "vlm_class": classifier.vlm.__class__.__name__,
        "classifier_class": classifier.__class__.__name__,
        "dataset_split": dataset_split_path,
        "n_way": n_way,
        "n_support": n_support,
        "n_query": n_query,
        "n_episodes": n_episodes
    }
    row.update({
        f"vlm.{key}": val
        for key, val in classifier.vlm.params().items()
    })
    row.update({
        f"classifier.{key}": val
        for key, val in classifier.params().items()
    })
        
    if accuracy is not None:
        row["accuracy"] = accuracy

    return row

def test_already_stored(results: pd.DataFrame,
                        classifier: FewShotClassifier, dataset_split_path: str,
                        n_way: int, n_support: int, n_query: int, n_episodes: int) -> bool:

    valid_indices = np.ones(len(results)).astype(bool)
    for key, val in dataframe_format(classifier, dataset_split_path, n_way, n_support, n_query, n_episodes).items():
        if key not in results.columns:
            return False
        valid_indices = valid_indices & (results[key] == val)
    
    return np.any(valid_indices)
        

def append_test_result(results: pd.DataFrame,
                      classifier: FewShotClassifier, dataset_split_path: str,
                      n_way: int, n_support: int, n_query: int, n_episodes: int,
                      accuracy: float) -> pd.DataFrame:
    
    formatted_row = dataframe_format(classifier, dataset_split_path, n_way, n_support, n_query, n_episodes, accuracy)
    
    # Check if any new columns need to be added (new vlm/classifier-specific params)
    new_columns = set(formatted_row.keys()) - set(results.columns)
    if len(new_columns):
        # Add new columns
        for col in new_columns:
            results[col] = np.nan
        
        # Reorder columns
        sorted_vlm_param_columns = sorted(col for col in results.columns if "vlm." in col)
        sorted_classifier_param_columns = sorted(col for col in results.columns if "classifier." in col)
        sorted_columns = ["vlm_class"] + sorted_vlm_param_columns + ["classifier_class"] + sorted_classifier_param_columns + \
                         ["dataset_split", "n_way", "n_support", "n_query", "n_episodes", "accuracy"]
        results = results.reindex(columns=sorted_columns)
        
    results.loc[len(results)] = formatted_row
    return results

def extract_test_result_sequence(results: pd.DataFrame,
                                 x_column: str, y_column: str = "accuracy",
                                 filter: dict = {}) -> pd.DataFrame:
    
    filtered_indices = np.ones(len(results)).astype(bool)
    for filter_col, filter_val in filter.items():
        filtered_indices = filtered_indices & (results[filter_col] == filter_val)
    
    return results[filtered_indices].sort_values(x_column).groupby([col for col in results if col not in [x_column, y_column]], as_index=False, dropna=False).agg(list)