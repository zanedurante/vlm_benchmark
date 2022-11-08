import os
from typing import Optional
import numpy as np
import pandas as pd
from tqdm.autonotebook import tqdm
import json

from SimilarityVLM import SimilarityVLM
from classifier import FewShotClassifier
from dataset import DatasetHandler, FewShotTaskDataset



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

    def run_few_shot_test(self, classifier: FewShotClassifier, query_dataset: DatasetHandler, support_dataset: DatasetHandler,
                          n_way: int, n_support: int, n_query: Optional[int] = None, n_episodes: int = 1000,
                          val_tuning_dataset: Optional[DatasetHandler] = None
                          ) -> None:
        """Runs the given few-shot test if it has not already been performed,
        saving the accuracy.

        Args:
            classifier (FewShotClassifier): Few-Shot video classifier built on top of an arbitrary SimilarityVLM
            query_dataset (DatasetHandler): Dataset Handler to draw queries from
            support_dataset (DatasetHandler): Dataset Handler to draw support videos from. Can be the same as query_dataset
            n_way (int): Number of categories per few-shot task
            n_support (int): Number of example videos per category per few-shot task
            n_query (Optional[int], optional): Number of videos predicted per category per few-shot task. If None, uses all videos not in support set. Defaults to None.
            n_episodes (int, optional): Number of few-shot tasks to sample. Defaults to 1000.
            val_tuning_dataset (Optional[DatasetHandler], optional): Optionally provided val dataset which classifiers can use to select the best performing epoch.
        Returns:
            (float) Accuracy for the given test.
        """
        
        # Skip test if it already exists
        if test_already_stored(self.results, classifier, query_dataset, support_dataset, n_way, n_support, n_query, n_episodes, val_tuning_dataset):
            # Find dataframe row with accuracy from this run
            filter_indices = np.ones(len(self.results))
            for key, val in dataframe_format(classifier, query_dataset, support_dataset, n_way, n_support, n_query, n_episodes, val_tuning_dataset).items():
                if pd.isna(val):
                    filter_indices = filter_indices & pd.isna(self.results[key])
                else:
                    filter_indices = filter_indices & (self.results[key] == val)
            matched_row = self.results[filter_indices]
            accuracy = matched_row["accuracy"].values[0]
            return accuracy
        
        # Load dataset to generate tasks with the desired params
        try:
            few_shot_dataset = FewShotTaskDataset(query_dataset, support_dataset, n_episodes, n_way, n_support, n_query, val_tuning_dataset)
        except ValueError:
            # Skip invalid tests (if dataset too small, etc)
            return None
        
        task_accuracies = []
        total_queries = 0
        total_correct = 0
        dataset_iter = tqdm(few_shot_dataset, leave=False)
        for category_names, support_vid_paths, query_vid_paths, query_vid_labels, val_tuning_vid_paths, val_tuning_vid_labels in dataset_iter:
                
            query_predictions = classifier.predict(category_names, support_vid_paths, query_vid_paths, val_tuning_vid_paths, val_tuning_vid_labels)
            
            # Compute accuracy for this sampled task
            correct_predictions = np.sum(query_predictions == query_vid_labels)
            task_accuracy = correct_predictions / len(query_vid_paths)
            task_accuracies.append(task_accuracy)
            
            # Aggregate for accuracy over all sampled tasks
            total_queries += len(query_vid_paths)
            total_correct += correct_predictions
            dataset_iter.set_postfix({"accuracy": total_correct / total_queries})
        
        # TODO: Look into other error/confidence-bound measures we should save
        # - Uncertainty in performance given a particular set of support videos (decreases with number of queries)
        # - Uncertainty in performance over a variety of support videos (decreases with number of sampled tasks (n_episodes))
        accuracy = total_correct / total_queries
        accuracy_std = np.std(task_accuracies)
        
        # Add to test results and save
        self.results = append_test_result(self.results, classifier, query_dataset, support_dataset, n_way, n_support, n_query, n_episodes, val_tuning_dataset, accuracy, accuracy_std)
        if self.test_results_path is not None:
            self.results.to_csv(self.test_results_path, index=False)
            
        return accuracy
        
    
    
    
    
       
'''
Test Results DataFrame Utilities
'''

def dataframe_format(classifier: FewShotClassifier, query_dataset: DatasetHandler, support_dataset: DatasetHandler,
                     n_way: int, n_support: int, n_query: int, n_episodes: int, val_tuning_dataset: Optional[DatasetHandler],
                     accuracy: Optional[float] = None, accuracy_std: Optional[float] = None) -> dict:
    row = {
        "vlm_class": classifier.vlm.__class__.__name__,
        "classifier_class": classifier.__class__.__name__,
        "query_dataset": query_dataset.id(),
        "support_dataset": support_dataset.id(),
        "n_way": n_way,
        "n_support": n_support,
        "n_query": n_query,
        "n_episodes": n_episodes,
        "val_tuning_dataset": None if val_tuning_dataset is None else val_tuning_dataset.id()
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
        
    for key, val in row.items():
        if val is None:
            row[key] = np.nan 

    return row

def filter_test_results(results: pd.DataFrame, column_value_dict: dict) -> pd.DataFrame:
    if len(results) == 0:
        return results
    
    valid_indices = np.ones(len(results)).astype(bool)
    for col, val in column_value_dict.items():
        if col not in results.columns:
            valid_indices &= False
        if pd.isna(val):
            valid_indices &= pd.isna(results[col])
        else:
            valid_indices &= (results[col] == val)
    return results[valid_indices]

def test_already_stored(results: pd.DataFrame,
                        classifier: FewShotClassifier, query_dataset: DatasetHandler, support_dataset: DatasetHandler,
                        n_way: int, n_support: int, n_query: int, n_episodes: int, val_tuning_dataset: Optional[DatasetHandler]) -> bool:

    valid_indices = np.ones(len(results)).astype(bool)
    for key, val in dataframe_format(classifier, query_dataset, support_dataset, n_way, n_support, n_query, n_episodes, val_tuning_dataset).items():
        if key not in results.columns:
            return False
        if pd.isna(val):
            valid_indices = valid_indices & pd.isna(results[key])
        else:
            valid_indices = valid_indices & (results[key] == val)
    
    return np.any(valid_indices)
        

def append_test_result(results: pd.DataFrame,
                       classifier: FewShotClassifier, query_dataset: DatasetHandler, support_dataset: DatasetHandler,
                       n_way: int, n_support: int, n_query: int, n_episodes: int, val_tuning_dataset: Optional[DatasetHandler],
                       accuracy: float, accuracy_std: float) -> pd.DataFrame:
    
    formatted_row = dataframe_format(classifier, query_dataset, support_dataset, n_way, n_support, n_query, n_episodes, val_tuning_dataset, accuracy, accuracy_std)
    
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
                         ["query_dataset", "support_dataset", "n_way", "n_support", "n_query", "n_episodes", "val_tuning_dataset", "accuracy", "accuracy_std"]
        results = results.reindex(columns=sorted_columns)
        
    results.loc[len(results)] = formatted_row
    return results

def find_hyperparameters(results: pd.DataFrame,
                         hyperparam_cols: list,
                         average_over_cols: list = [],
                         target_cols: list = ["accuracy", "accuracy_std"],
                         val_split: str = "val") -> pd.DataFrame:
    """Finds the best hyperparameter values which maximize accuracy in the given val_split.
    Returns a dataframe containing those hyperparameter values for each unique set of test parameters.
    Unlike optimize_hyperparameters, it doesn't then select the corresponding results from a test split,
    it just returns the values of the best hyperparameters.

    Args:
        results (pd.DataFrame): Full test results dataframe.
        hyperparam_cols (list): Columns from which the best performing values per specific test are chosen.
        average_over_cols (list, optional): Columns which will be averaged over before computing performance of difference hyperparameter values. Defaults to ["n_way", "n_support"].
        target_cols (list, optional): Columns which specify the output of a test, rather than an input parameter. Defaults to ["accuracy", "accuracy_std"].
        val_split (str, optional): The dataset split to use to compute the best hyperparameters. Defaults to "val".

    Returns:
        pd.DataFrame: DataFrame containing only the best hyperparameter values for each unique set of parameters in the given val split.
    """
    
    hyperparam_cols = [col for col in hyperparam_cols if col in results.columns]
    average_over_cols = [col for col in average_over_cols if col in results.columns]
    target_cols = [col for col in target_cols if col in results.columns]
    group_by_cols = [col for col in results.columns if col not in average_over_cols + hyperparam_cols + target_cols]
    
    # Filter to val split
    results = results[results["query_dataset"].str.split(".", expand=True)[2] == val_split]
    
    grouped_results = results\
        .groupby(group_by_cols + hyperparam_cols, as_index=False, dropna=False).agg({col: np.mean for col in target_cols})\
        .sort_values("accuracy", ascending=False).drop_duplicates(group_by_cols).drop(columns=target_cols)
        
    return grouped_results

def optimize_hyperparameters(results: pd.DataFrame,
                             hyperparam_cols: list,
                             average_over_cols: list = [],
                             target_cols: list = ["accuracy", "accuracy_std"],
                             val_split: str = "val",
                             test_split: str = "test") -> pd.DataFrame:
    """Given a results dataframe which covers multiple splits of the same dataset, and a choice
    of hyperparameter columns to optimize, returns a results dataframe only containing datapoints
    from the test split which use the hyperparameter selections which performed best in the val split,
    for each specific test. All columns not mentioned in one of the arguments are used as test specifiers,
    meaning optimal hyperparameters are found for each unique value of the test specifiers, and only applied
    if that unique value of test specifiers (and optimal hyperparameters) also exists in the test set.

    Args:
        results (pd.DataFrame): Full test results dataframe.
        hyperparam_cols (list): Columns from which the best performing values per specific test are chosen.
        average_over_cols (list, optional): Columns which will be averaged over before computing performance of difference hyperparameter values. Defaults to ["n_way", "n_support"].
        target_cols (list, optional): Columns which specify the output of a test, rather than an input parameter. Defaults to ["accuracy", "accuracy_std"].
        val_split (str, optional): The dataset split to use to compute the best hyperparameters. Defaults to "val".
        test_split (str, optional): The dataset split to which the optimal hyperparameters are applied. Defaults to "test".

    Returns:
        pd.DataFrame: Results dataframe filtered to contain only test split results which use the hyperparameters which performed best on the val set.
    """
    hyperparam_cols = [col for col in hyperparam_cols if col in results.columns]
    average_over_cols = [col for col in average_over_cols if col in results.columns]
    target_cols = [col for col in target_cols if col in results.columns]
    group_by_cols = [col for col in results.columns if col not in average_over_cols + hyperparam_cols + target_cols]
    
    grouped_results = results\
        .groupby(group_by_cols + hyperparam_cols, as_index=False, dropna=False).agg({col: np.mean for col in target_cols})\
        .sort_values("accuracy", ascending=False).drop_duplicates(group_by_cols)

    output = pd.DataFrame(columns=results.columns)
    for i in grouped_results.index:
        row = grouped_results.loc[i]
        
        if row["query_dataset"].split(".")[2] != val_split:
            continue
        
        # Find all rows corresponding to the test-dataset version of this group, and then further select the correct hyperparams
        filtered_results = results
        for col in group_by_cols + hyperparam_cols:
            if col == "query_dataset":
                dataset_id_fields = row["query_dataset"].split(".")
                dataset_id_fields[2] = test_split
                val = ".".join(dataset_id_fields)
            else:
                val = row[col]

            if pd.isna(val):
                filtered_results = filtered_results[pd.isna(filtered_results[col])]
            else:
                filtered_results = filtered_results[filtered_results[col] == val]

        for j in filtered_results.index:
            output.loc[len(output)] = filtered_results.loc[j]

    # When n_support is 0, text_weight is fixed to 1, even though the datapoint is effectively valid for any text_weight line.
    # If the selected results contain no n_support = 0 points, attempt to find corresponding ones, relabel their text weights, and add them
    if "classifier.text_weight" in hyperparam_cols and "n_support" not in group_by_cols and (output["n_support"] == 0).sum() == 0:
        line_identifiers = output.drop_duplicates(group_by_cols + hyperparam_cols).drop(columns=target_cols + average_over_cols)
        line_identifiers = line_identifiers.drop(columns=["classifier.text_weight"])
        line_identifiers.loc[:, "n_support"] = 0
        
        additional_results = pd.merge(results, line_identifiers)
        for j in additional_results.index:
            output.loc[len(output)] = additional_results.loc[j]
            
    return output