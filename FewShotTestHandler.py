import os
from typing import Optional
import numpy as np
import pandas as pd
from tqdm.autonotebook import tqdm
import json

from SimilarityVLM import SimilarityVLM
from classifier import FewShotClassifier
from dataset import DatasetHandler



'''
Class for running few-shot tests, saving the results, and facillitating result analysis
'''

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
TEST_RESULTS_PATH = os.path.join(FILE_DIR, "test_results.csv")


class FewShotTestHandler:
    def __init__(self, test_results_path: Optional[str] = TEST_RESULTS_PATH):
        self.test_results_path = test_results_path
        
        # Load results DataFrame
        if test_results_path is not None and os.path.exists(test_results_path):
            self.results = pd.read_csv(test_results_path)
        else:
            self.results = pd.DataFrame()

    def run_few_shot_test(self, classifier: FewShotClassifier, dataset: DatasetHandler,
                          n_way: int, n_support: int, n_query: int = 1, n_episodes: int = 1000,
                          ) -> None:
        """Runs the given few-shot test if it has not already been performed,
        saving the accuracy.

        Args:
            classifier (FewShotClassifier): Few-Shot video classifier built on top of an arbitrary SimilarityVLM
            dataset (DatasetHandler): Dataset Handler to build few-shot tasks from
            n_way (int): Number of categories per few-shot task
            n_support (int): Number of example videos per category per few-shot task
            n_query (int, optional): Number of videos predicted per category per few-shot task
            n_episodes (int, optional): Number of few-shot tasks to sample. Defaults to 1000.
        """
        
        # Skip test if it already exists
        if test_already_stored(self.results, classifier, dataset, n_way, n_support, n_query, n_episodes):
            return
        
        # Load dataset to generate tasks with the desired params
        few_shot_dataset = dataset.few_shot(n_episodes, n_way, n_support, n_query)
        
        task_accuracies = []
        total_queries = 0
        total_correct = 0
        dataset_iter = tqdm(few_shot_dataset, leave=False)
        for vid_paths, category_names in dataset_iter:
            
            query_vid_paths = vid_paths[:, n_support:]
            if n_support > 0:
                support_vid_paths = vid_paths[:, :n_support]
            else:
                support_vid_paths = None
                
            query_predictions = classifier.predict(category_names, support_vid_paths, query_vid_paths)
            
            # Compute accuracy for this sampled task
            correct_predictions = np.sum(query_predictions == np.arange(n_way)[:, None])
            task_accuracy = correct_predictions / (n_way * n_query)
            task_accuracies.append(task_accuracy)
            
            # Aggregate for accuracy over all sampled tasks
            total_queries += n_way * n_query
            total_correct += correct_predictions
            dataset_iter.set_postfix({"accuracy": total_correct / total_queries})
        
        accuracy = total_correct / total_queries
        accuracy_std = np.std(task_accuracies)
        
        # Add to test results and save
        self.results = append_test_result(self.results, classifier, dataset, n_way, n_support, n_query, n_episodes, accuracy, accuracy_std)
        if self.test_results_path is not None:
            self.results.to_csv(self.test_results_path, index=False)
        
    
    
    
    
       
'''
Test Results DataFrame Utilities
'''

def dataframe_format(classifier: FewShotClassifier, dataset: DatasetHandler,
                     n_way: int, n_support: int, n_query: int, n_episodes: int,
                     accuracy: Optional[float] = None, accuracy_std: Optional[float] = None) -> dict:
    row = {
        "vlm_class": classifier.vlm.__class__.__name__,
        "classifier_class": classifier.__class__.__name__,
        "dataset": dataset.id(),
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
        
    if accuracy_std is not None:
        row["accuracy_std"] = accuracy_std

    return row

def test_already_stored(results: pd.DataFrame,
                        classifier: FewShotClassifier, dataset: DatasetHandler,
                        n_way: int, n_support: int, n_query: int, n_episodes: int) -> bool:

    valid_indices = np.ones(len(results)).astype(bool)
    for key, val in dataframe_format(classifier, dataset, n_way, n_support, n_query, n_episodes).items():
        if key not in results.columns:
            return False
        valid_indices = valid_indices & (results[key] == val)
    
    return np.any(valid_indices)
        

def append_test_result(results: pd.DataFrame,
                       classifier: FewShotClassifier, dataset: DatasetHandler,
                       n_way: int, n_support: int, n_query: int, n_episodes: int,
                       accuracy: float, accuracy_std: float) -> pd.DataFrame:
    
    formatted_row = dataframe_format(classifier, dataset, n_way, n_support, n_query, n_episodes, accuracy, accuracy_std)
    
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
                         ["dataset", "n_way", "n_support", "n_query", "n_episodes", "accuracy", "accuracy_std"]
        results = results.reindex(columns=sorted_columns)
        
    results.loc[len(results)] = formatted_row
    return results

def extract_test_result_sequence(results: pd.DataFrame,
                                 x_column: str, y_column: str = "accuracy",
                                 filter: dict = {}) -> pd.DataFrame:
    
    filtered_indices = np.ones(len(results)).astype(bool)
    for filter_col, filter_val_list in filter.items():
        if filter_col not in results.columns:
            continue
        
        valid_col_indices = np.zeros(len(results)).astype(bool)
        for filter_val in filter_val_list:
            valid_col_indices = valid_col_indices | (results[filter_col] == filter_val)
        filtered_indices = filtered_indices & valid_col_indices
    
    return results[filtered_indices].sort_values(x_column).groupby([col for col in results if col not in [x_column, y_column]], as_index=False, dropna=False).agg(list)