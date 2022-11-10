from typing import Any, Callable, Optional
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# DATAFRAME MANIPULATION FUNCTIONS

'''
Filter the given dataframe based on a dictionary of allowed column values
'''
def filter_results(results: pd.DataFrame, filter: dict) -> pd.DataFrame:
    filtered_indices = np.ones(len(results)).astype(bool)
    for filter_col, filter_val_list in filter.items():
        if filter_col not in results.columns:
            continue
        if type(filter_val_list) is not list:
            filter_val_list = [filter_val_list]
        
        valid_col_indices = np.zeros(len(results)).astype(bool)
        for filter_val in filter_val_list:
            if pd.isna(filter_val): # Special handling required to check for nan results (which imply that the column was not filled for this row)
                valid_col_indices |= pd.isna(results[filter_col])
            else:
                valid_col_indices |= (results[filter_col] == filter_val)
        filtered_indices &= valid_col_indices
        
    return results[filtered_indices]

'''
Concatenate two results dataframes together
'''
def combine_results(*results: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(results, ignore_index=True)

'''
Group dataframe such that the given columns are aggregated into lists, one for every unique value of
the unmentioned columns. The lists for the first sequence_column will be sorted in ascending order
'''
def group_into_sequence(results: pd.DataFrame, sequence_columns: list) -> pd.DataFrame:
    return results.sort_values(sequence_columns[0]).groupby([col for col in results if col not in sequence_columns], as_index=False, dropna=False).agg(list)

'''
Aggregate over values of the source and target columns (grouped by unique values of the other columns),
keeping the target column after applying a particular aggregation function.
'''
def aggregate(results: pd.DataFrame, removed_column: str, agg_func: Callable = max, aggregated_columns: list = ["accuracy", "accuracy_std"]) -> pd.DataFrame:
    if removed_column not in results.columns:
        return results
    
    return results.sort_values(removed_column)\
        .groupby([col for col in results if col not in [removed_column] + aggregated_columns], as_index=False, dropna=False)\
        .agg({c: agg_func for c in aggregated_columns})



# PLOTTING FUNCTIONS

'''
Format plot and line descriptions to more readable form
'''
def column_value_formatter(col: str, val: Any) -> str:
    # Format value
    if col == "vlm_class":
        transform = {
            "ClipVLM": "CLIP",
            "MILES_SimilarityVLM": "MILES",
            "VideoClipVLM": "VideoCLIP"
        }
        return transform.get(val, val)
    
    if col == "classifier_class":
        transform = {
            "HardPromptFewShotClassifier": "VL-Prototype",
            "CoopFewShotClassifier": "CoOp",
            "CoNaFewShotClassifier": "CoNa (Ours)"
        }
        return transform.get(val, val)
    
    if col == "query_dataset":
        val = val.split(".")
        name = val[0]
        split_type = val[1]
        split = val[2]
        if len(val) == 3:
            min_train_vids = None
            class_limit = None
        else:
            min_train_vids = val[3]
            class_limit = val[4]
            
        transform = {
            "kinetics_100": "Kinetics-100",
            "smsm": "Something-Something-v2",
            "moma_act": "MOMA Activity",
            "moma_sact": "MOMA Sub-Activity"
        }
        
        result = transform.get(name, name)
        if split_type == "v":
            pass
        elif split_type == "c":
            result += " (few-shot)"
        else:
            raise NotImplementedError
        
        if split != "all":
            result += f" ({split})"
        if min_train_vids is not None:
            result += f" ({min_train_vids} vid min)"
        if class_limit is not None:
            result += f" ({class_limit} class max)"
        return result
    
    if col == "n_way":
        return f"{val}-way"
    
    if col == "n_support":
        return f"{val}-shot"
    
    
    # Remove 'vlm.' or 'classifier.' prefix for non-special-cases
    if "." in col:
        col = col[col.index(".") + 1:]
    
    return f"{col}: {val}"

'''
Format axis names to more readable form
'''
def column_description(col: str) -> str:
    if col == "n_support":
        return "Support Videos"
    
    if col == "classifier.text_weight":
        return "Text Weight"
    
    return col

'''
General purpose few-shot-result line-plotting function
Args:
    results (pd.DataFrame):         Test results dataframe, in format of FewShotTestHandler
    x_col (str):                    Column for x-axis of line plots
    y_col (str):                    Column for y-axis of line plots
    plot_descriptor_cols ([str]):   List of columns, a separate plot will be created for each unique value
    line_descriptor_cols ([str]):   List of columns, determines what values will be listed in each line's legend label
    agg_dict ({str -> Callable}):   Dictionary of columns which will be aggregated out of the final result, along with
                                    the function for aggregating the values of accuracy which were only differentiated
                                    by the specified aggregated-out column.
    filter_dict ({str -> [Any]}):   Dictionary from column names to allowed values. Filter will be applied after aggregation,
                                    except for any columns which would be aggregated out.
    show_error_bars (bool):         Whether to display error bars when y-axis is 'accuracy'. Error bars will show standard error.
    savedir (Optional[str], optional): Optional directory path in which plots will be saved.
'''
def plot(results: pd.DataFrame, x_col: str, y_col: str, plot_descriptor_cols: list, line_descriptor_cols: list,
         agg_dict: dict = {}, filter_dict: dict = {}, show_error_bars: bool = False, savedir: Optional[str] = None):
    pre_agg_filter_dict = {col: vals for col, vals in filter_dict.items() if col in agg_dict.keys()}
    post_agg_filter_dict = {col: vals for col, vals in filter_dict.items() if col not in agg_dict.keys()}
    
    results = filter_results(results, pre_agg_filter_dict)
    for agg_col, agg_func in agg_dict.items():
        results = aggregate(results, agg_col, agg_func)
    results = filter_results(results, post_agg_filter_dict)
    
    # Create directory for plots if needed
    if savedir is not None:
        os.makedirs(savedir, exist_ok=True)
    
    # Group around all unique line identifiers (plot_descriptors + line_descriptors), then aggregate x_col and y_col into lists
    grouped_results = results.sort_values(x_col).groupby(plot_descriptor_cols + line_descriptor_cols, as_index=False, dropna=False).agg(list)
    print(f"{len(grouped_results)} Overall Lines")
    
    if len(grouped_results) == 0:
        return
    
    plot_descriptors = grouped_results.reset_index().groupby(plot_descriptor_cols, as_index=False, dropna=False).agg({"index": list})
    print(f"{len(plot_descriptors)} Separate Plots")
    
    for plot_ind in range(len(plot_descriptors)):
        plot_name = ", ".join([column_value_formatter(col, plot_descriptors.loc[plot_ind, col]) for col in plot_descriptor_cols])
        
        fig, ax = plt.subplots(figsize=(8,4))
        fig.suptitle(plot_name, fontsize=15)
        
        lines = grouped_results.loc[plot_descriptors.loc[plot_ind, "index"]].reset_index()
        
        for i in range(len(lines)):
            line_name = ", ".join([column_value_formatter(col, lines.loc[i, col]) for col in line_descriptor_cols])
            
            x = lines.loc[i, x_col]
            y = lines.loc[i, y_col]
            if y_col == "accuracy" and show_error_bars:
                y_std = lines.loc[i, "accuracy_std"]
                n_episodes = lines.loc[i, "n_episodes"]
                y_err = np.nan_to_num(y_std) / np.sqrt(n_episodes)
                if np.all(y_err == 0):
                    y_err = None
            else:
                y_err = None
        
            if x_col == "classifier.text_weight" and lines.loc[i, "n_support"] == 0 and len(x) == 1:
                ax.axhline(y[0], label=line_name, linestyle="dashed")
            elif "classifier.text_weight" in line_descriptor_cols and x_col == "n_support" and x[0] == 0:
                ax.scatter([x[0]], [y[0]])
                ax.errorbar(x[1:], y[1:], yerr=y_err, label=line_name)
            else:
                ax.errorbar(x, y, yerr=y_err, label=line_name)
        
        ax.set_xlabel(column_description(x_col))
        ax.set_ylabel(column_description(y_col))
        if len(lines) > 6:
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        else:
            ax.legend()
            
        # Save plot if needed
        if savedir is not None:
            fig.savefig(os.path.join(savedir, f"plot_{plot_ind}.png"), facecolor="white")
            
        fig.show()