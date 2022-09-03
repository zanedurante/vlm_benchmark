from typing import Any, Callable, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# DATAFRAME MANIPULATION FUNCTIONS

'''
Filter the given dataframe based on a dictionary of allowed column values
'''
def filter(results: pd.DataFrame, filter: dict) -> pd.DataFrame:
    filtered_indices = np.ones(len(results)).astype(bool)
    for filter_col, filter_val_list in filter.items():
        if filter_col not in results.columns:
            continue
        if type(filter_val_list) is not list:
            filter_val_list = [filter_val_list]
        
        valid_col_indices = np.zeros(len(results)).astype(bool)
        for filter_val in filter_val_list:
            valid_col_indices = valid_col_indices | (results[filter_col] == filter_val)
        filtered_indices = filtered_indices & valid_col_indices
        
    return results[filtered_indices]

'''
Group dataframe such that the given x,y columns are aggregated into lists, one for every unique value of
the unmentioned columns
'''
def group_into_sequence(results: pd.DataFrame, x_column: str, y_column: str = "accuracy") -> pd.DataFrame:
    return results.sort_values(x_column).groupby([col for col in results if col not in [x_column, y_column]], as_index=False, dropna=False).agg(list)

'''
Aggregate over values of the source and target columns (grouped by unique values of the other columns),
keeping the target column after applying a particular aggregation function.
'''
def aggregate(results: pd.DataFrame, removed_column: str, agg_func: Callable = max, aggregated_column: str = "accuracy") -> pd.DataFrame:
    return results.sort_values(removed_column)\
        .groupby([col for col in results if col not in [removed_column, aggregated_column]], as_index=False, dropna=False)\
        .agg({aggregated_column: agg_func})



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
    
    if col == "dataset":
        name, split = val.split(".")
            
        transform = {
            "kinetics_100": "Kinetics-100",
            "moma_act": "MOMA Activity",
            "moma_sact": "MOMA Sub-Activity"
        }
        
        result = transform.get(name, name)
        if split != "all":
            result += f" ({split})"
        return result
    
    if col == "n_way":
        return f"{val}-way"
    
    if col == "n_support":
        return f"{val}-shot"
    
    if col == "classifier.text_weight":
        return f"text_weight: {val}"
    
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
'''
def plot(results: pd.DataFrame, x_col: str, y_col: str, plot_descriptor_cols: list[str], line_descriptor_cols: list[str], agg_dict: dict = {}, filter_dict: dict = {}):
    pre_agg_filter_dict = {col: vals for col, vals in filter_dict.items() if col in agg_dict.keys()}
    post_agg_filter_dict = {col: vals for col, vals in filter_dict.items() if col not in agg_dict.keys()}
    
    results = filter(results, pre_agg_filter_dict)
    for agg_col, agg_func in agg_dict.items():
        results = aggregate(results, agg_col, agg_func)
    results = filter(results, post_agg_filter_dict)
    
    grouped_results = group_into_sequence(results, x_col, y_col)
    print(f"{len(grouped_results)} Overall Lines")
    
    plot_descriptors = grouped_results.reset_index().groupby(plot_descriptor_cols, as_index=False, dropna=False).agg({"index": list})
    print(f"{len(plot_descriptors)} Separate Plots")
    
    for plot_descriptor in plot_descriptors.itertuples():
        plot_name = ", ".join([column_value_formatter(col, getattr(plot_descriptor, col)) for col in plot_descriptor_cols])
        
        fig, ax = plt.subplots(figsize=(8,4))
        fig.suptitle(plot_name, fontsize=15)
        
        lines = grouped_results.loc[plot_descriptor.index].reset_index()
        
        for i in range(len(lines)):
            line_name = ", ".join([column_value_formatter(col, lines.loc[i, col]) for col in line_descriptor_cols])
                
            x, y = lines.loc[i, [x_col, y_col]]
        
            if x_col == "classifier.text_weight" and lines.loc[i, "n_support"] == 0 and len(x) == 1:
                ax.axhline(y[0], label=line_name, linestyle="dashed")
            elif "classifier.text_weight" in line_descriptor_cols and x_col == "n_support" and x[0] == 0:
                ax.scatter([x[0]], [y[0]])
                ax.plot(x[1:], y[1:], label=line_name)
            else:
                ax.plot(x, y, label=line_name)
        
        ax.set_xlabel(column_description(x_col))
        ax.set_ylabel(column_description(y_col))
        if len(lines) > 6:
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        else:
            ax.legend()
        fig.show()