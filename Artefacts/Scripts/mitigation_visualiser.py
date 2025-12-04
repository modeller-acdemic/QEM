# -*- coding: utf-8 -*-
"""
Created on Sat Sep 13 13:45:45 2025

Reads in the results of each strata from the test results and provides clustered bar charts of varying levels of analysis

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

current_pvals = None
bootstrap_iters = 10000
baseline_pvals = None
baseline_values = None

# load result files
def load_results():
    
    # set allowed values
    fractions = [0, 25, 50, 75, 100]
    models = ["", "_DT", "_GBR", "_RFRetuned", "_rt"]
    dataframes = []
    
    # iterate declared files
    for real_share in fractions:
        
        # iterate models
        for model in models:
            
            # build file path
            file_path = f"../Data/test_results/test_results_{real_share}{model}.csv"
            
            # derive algorithm
            algo_name = "RF" if model == "" else model[1:]
            
            # read file
            data_frame = pd.read_csv(file_path)
            
            # pick columns
            pick_cols = data_frame.loc[:, ["circuit_family", "n", "d", "rmse", "rmse_unmitigated"]].copy()
            pick_cols["real_percentage"] = real_share
            pick_cols["algorithm"] = algo_name
            
            # collect frame
            dataframes.append(pick_cols)
    
    # combine frames
    result_frame = pd.concat(dataframes, ignore_index=True)
    
    return result_frame

# build a pivot with consistent ordering
def make_pivot(frame, aggregate_fn, proportion_order, algorithm_order):
    
    # group values for aggregation
    grouped_series = frame.groupby(["real_percentage", "algorithm"])["rmse"].apply(aggregate_fn)
    
    # reshape to wide format
    pivot = grouped_series.unstack("algorithm")
    
    # order proportions and algorithms
    pivot = pivot.reindex(proportion_order)
    pivot = pivot.reindex(columns=[a for a in algorithm_order if a in pivot.columns])
    
    return pivot

# compute one-sample two-sided bootstrap p value for the median
def onesample_pval(values):
    
    # set null median
    null_median = 0.0
    
    # set bootstrap seed
    bootstrap_seed = 42
    
    # convert to array
    values_array = np.asarray(values)
    
    # create random generator
    random_generator = np.random.default_rng(bootstrap_seed)
    
    # compute observed median
    observed_median = np.median(values_array)
    
    # centre data to the null
    centred_values = values_array - observed_median + null_median
    
    # collect bootstrap medians under the null
    medians_under_null = np.array([np.median(random_generator.choice(centred_values, size=centred_values.size, replace=True)) for _ in range(bootstrap_iters)])
    
    # compute extreme count
    extreme_count = (np.abs(medians_under_null - null_median) >= np.abs(observed_median - null_median)).sum()
    
    # compute p value
    p_val = (extreme_count + 1) / (bootstrap_iters + 1)
    
    return float(p_val)

# build a p value pivot aligned to the aggregated pivot
def make_pvals(frame, proportion_order, algorithm_order):
    
    # compute p per group
    p_series = frame.groupby(["real_percentage", "algorithm"])["rmse"].apply(onesample_pval)
    
    # reshape to wide
    p_pivot = p_series.unstack("algorithm")
    
    # order proportions
    p_pivot = p_pivot.reindex(proportion_order)
    
    # order algorithms
    p_pivot = p_pivot.reindex(columns=[a for a in algorithm_order if a in p_pivot.columns])
    
    return p_pivot

# render one grouped horizontal bar chart
def plot_group(pivot, title_text, algorithm_order, axis_object_in=None, draw_legend=True):
    
    # create axes
    axis_object = axis_object_in or plt.subplots(figsize=(10, 7), constrained_layout=True)[1]
    
    # get row and column labels
    row_labels = list(pivot.index)
    all_cols = list(algorithm_order)
    baseline_cols = [c for c in pivot.columns if c not in all_cols]
    all_cols.extend(baseline_cols)

    # build colour map
    base_colors = ['#88CCEE', '#44AA99', '#117733', '#999933', '#DDCC77', '#A05B65', '#7B2E2E']
    column_colors = {name: base_colors[i % len(base_colors)] for i, name in enumerate(all_cols)}
    
    # define bar geometry
    cluster_span = 0.8
    bar_height = cluster_span / max(len(all_cols), 1)
    
    # set y tick positions
    y_positions = np.arange(len(row_labels))
    
    # set a small fixed text offset for labels
    text_offset = 0.0005

    # iterate rows to plot bars and p-values
    for row_pos, row_name in zip(y_positions, row_labels):
        
        # get columns with data in this row
        cols_in_row = [c for c in all_cols if c in pivot.columns and not pd.isna(pivot.loc[row_name, c])]
        
        # calculate start index to centre the group
        start_index = (len(all_cols) - len(cols_in_row)) / 2.0
        
        # plot each bar and its p-value for the current row
        for i, col_name in enumerate(cols_in_row):
            
            # calculate position for this bar
            slot_index = start_index + i
            y_offset = cluster_span / 2 - slot_index * bar_height - bar_height / 2
            
            # draw bar
            bar_value = pivot.loc[row_name, col_name]
            axis_object.barh(row_pos + y_offset, float(bar_value), height=bar_height, label=col_name, color=column_colors.get(col_name))

            # build value string
            val_str = f"{bar_value:.3f}"
            if val_str.startswith("0."):
                val_str = val_str[1:]
            
            # decide p source
            p_val = None
            if row_name == "Unmit" and col_name not in algorithm_order:
                base_key = "Baseline B" if "Retest" in col_name else "Baseline A"
                p_val = baseline_pvals.get(base_key)
            elif row_name != "Unmit" and col_name in algorithm_order:
                p_val = current_pvals.loc[row_name, col_name]
                        
            # build label text
            label_text = val_str
            p_str = f"{p_val:.3f}"
            if p_str.startswith("0."):
                p_str = p_str[1:]
            label_text = f"{val_str}, p={p_str}"
            
            # decide label placement
            is_baseline_bar = (row_name == "Unmit" and col_name not in algorithm_order)
            
            # set label position and alignment
            label_x = float(bar_value) - text_offset if is_baseline_bar else float(bar_value) + text_offset
            h_align = "right" if is_baseline_bar else "left"
            
            # add the bar labels
            # axis_object.text(label_x, row_pos + y_offset, label_text, va="center", ha=h_align, fontsize=7)

    # set chart properties
    axis_object.set_yticks(y_positions)
    axis_object.set_yticklabels(row_labels)
    axis_object.grid(True, axis="x") # vertical marker lines
    axis_object.set_xlabel("RMSE")
    axis_object.set_ylabel("% hardware")
    axis_object.yaxis.labelpad = -2
    axis_object.set_title(title_text)
    
    # build legend only if requested
    if draw_legend:
        all_handles, all_labels = axis_object.get_legend_handles_labels()
        label_to_handle = dict(zip(all_labels, all_handles))
        ordered_labels = [name for name in all_cols if name in label_to_handle]
        axis_object.legend([label_to_handle[name] for name in ordered_labels], ordered_labels, loc='lower center', bbox_to_anchor=(0.5, -0.45), ncol=min(len(ordered_labels), 4))
        
# choose baseline bars for the overall chart
def choose_baseline_bars(algorithm_order):
    
    # return mapping of baseline labels to values
    if any(name in algorithm_order for name in ["RFRetuned", "rt"]):
        return {"RF Unmit.": baseline_values["Baseline A"], "RF Retest Unmit.": baseline_values["Baseline B"]}
    return {"RF unmit.": baseline_values["Baseline A"]}

# add baseline cluster row to a pivot
def add_baseline_cluster(pivot, bars_dict, proportion_order, algorithm_order):
    
    # select algorithm columns in requested order
    algo_cols = [a for a in algorithm_order if a in pivot.columns]
    
    # build final column order
    final_cols = algo_cols + list(bars_dict.keys())
    
    # reindex columns to the final order
    pivot = pivot.reindex(columns=final_cols)
    
    # build baseline row
    baseline_row = pd.Series(np.nan, index=pivot.columns, name="Unmit")
    
    # fill baseline values
    for name, val in bars_dict.items():
        baseline_row[name] = val
    
    # append baseline row
    pivot = pd.concat([pivot, baseline_row.to_frame().T], axis=0)
    
    # reorder rows
    pivot = pivot.reindex(list(proportion_order) + ["Unmit"])
    
    return pivot

# build and plot grouped charts for a categorical field
def plot_grouped(filtered, field_name, title_prefix, aggregate_fn, proportion_order, algorithm_order, cols_count=2, group=1, full_frame=None):
    
    # list values
    values = sorted(filtered[field_name].unique())
    
    # set grid size
    rows_count = (len(values) + cols_count - 1) // cols_count
    
    # define figure height per row to calculate total height
    height_per_row = 5.0
    figure_height = height_per_row * rows_count
    
    # create grid without constrained_layout
    fig, axes = plt.subplots(rows_count, cols_count, figsize=(6 * cols_count, figure_height), squeeze=False)
    
    # iterate values
    for idx, value in enumerate(values):
        
        # filter by value
        value_frame = filtered[filtered[field_name] == value]
        
        # build pivot
        pivot = make_pivot(value_frame, aggregate_fn, proportion_order, algorithm_order)
        
        # compute p values
        global current_pvals
        current_pvals = make_pvals(value_frame, proportion_order, algorithm_order)
        
        # compute subset baselines from the full dataset for this subplot
        subset_full = full_frame[full_frame[field_name] == value]
        
        non_rt_unmit = subset_full[subset_full["algorithm"] != "rt"]["rmse_unmitigated"]
        rt_unmit = subset_full[subset_full["algorithm"] == "rt"]["rmse_unmitigated"]
        
        # set subset baseline medians and p values
        global baseline_values, baseline_pvals
        baseline_values = {"Baseline A": aggregate_fn(non_rt_unmit)}
        baseline_pvals = {"Baseline A": onesample_pval(non_rt_unmit)}
        if not rt_unmit.empty:
            baseline_values["Baseline B"] = aggregate_fn(rt_unmit)
            baseline_pvals["Baseline B"] = onesample_pval(rt_unmit)
        
        # add baseline bars to this subplot’s pivot
        bars_for_subplot = choose_baseline_bars(algorithm_order)
        pivot = add_baseline_cluster(pivot, bars_for_subplot, proportion_order, algorithm_order)
        
        # locate subplot
        row_index = idx // cols_count
        col_index = idx % cols_count
        axis_object = axes[row_index][col_index]
        
        # plot chart without its own legend
        plot_group(pivot, f"{title_prefix}: {value}", algorithm_order, axis_object_in=axis_object, draw_legend=False)
    
    # hide unused axes
    for extra_index in range(len(values), rows_count * cols_count):
        row_index = extra_index // cols_count
        col_index = extra_index % cols_count
        axes[row_index][col_index].set_visible(False)
        
    # calculate the bottom margin
    legend_space_inches = 0.7
    bottom_margin_ratio = legend_space_inches / figure_height
    
    # calculate the legend's y-position
    legend_y_pos = bottom_margin_ratio / 2.0
    
    # create a shared legend
    handles, labels = axes[0][0].get_legend_handles_labels()
    unique_items = dict(zip(labels, handles))
    fig.legend(unique_items.values(), unique_items.keys(), loc='lower center', ncol=len(unique_items), bbox_to_anchor=(0.5, legend_y_pos))
    
    # use tight_layout
    fig.tight_layout(rect=[0, bottom_margin_ratio, 1, 1])
    # save figure 
    fig.savefig(f"../Data/analysis_results/graphs/{field_name.lower()}_median_rmse_{group}.png", dpi=300)
    
# produce all four scopes of charts
def chart_builder(result_frame, aggregate_fn, proportion_order, algorithm_order, group=1):
    
    # filter algorithms
    filtered = result_frame[result_frame["algorithm"].isin(algorithm_order)]
    
    # build overall pivot
    overall_pivot = make_pivot(filtered, aggregate_fn, proportion_order, algorithm_order)
    
    # compute overall p values
    global current_pvals
    current_pvals = make_pvals(filtered, proportion_order, algorithm_order)
    
    # compute unmitigated baselines
    non_rt_unmit = result_frame[result_frame["algorithm"] != "rt"]["rmse_unmitigated"]
    rt_unmit = result_frame[result_frame["algorithm"] == "rt"]["rmse_unmitigated"]
    
    # set baseline medians
    global baseline_values
    baseline_values = {"Baseline A": aggregate_fn(non_rt_unmit)}
    if not rt_unmit.empty:
        baseline_values["Baseline B"] = aggregate_fn(rt_unmit)
    
    # set baseline p-values
    global baseline_pvals
    baseline_pvals = {"Baseline A": onesample_pval(non_rt_unmit)}
    if not rt_unmit.empty:
        baseline_pvals["Baseline B"] = onesample_pval(rt_unmit)
    
    # add baseline cluster
    baseline_bars = choose_baseline_bars(algorithm_order)
    overall_pivot = add_baseline_cluster(overall_pivot, baseline_bars, proportion_order, algorithm_order)
    
    # create figure for overall plot
    fig, axis_object = plt.subplots(figsize=(10, 7))
    
    # plot overall without an axis-level legend
    plot_group(overall_pivot, "Median RMSE", algorithm_order, axis_object_in=axis_object, draw_legend=False)
    
    # set font sizes for title, axis labels and both tick axes
    axis_object.set_title(axis_object.get_title(), fontsize=14)
    axis_object.set_xlabel(axis_object.get_xlabel(), fontsize=12)
    axis_object.set_ylabel(axis_object.get_ylabel(), fontsize=12)
    axis_object.tick_params(axis='both', labelsize=12)
    
    # collect legend entries from the plotted artists
    all_handles, all_labels = axis_object.get_legend_handles_labels()
    label_to_handle = dict(zip(all_labels, all_handles))
    
    # order legend entries to match algorithms and baseline bars
    all_possible_labels = algorithm_order + list(baseline_bars.keys())
    legend_order = [label for label in all_possible_labels if label in label_to_handle]
    ordered_handles = [label_to_handle[label] for label in legend_order]
    
    # centre the legend on the frame
    # fig.legend(ordered_handles, legend_order, loc='lower center', ncol=len(legend_order), bbox_to_anchor=(0.5, 0.045), fontsize=12, columnspacing=1.0)
    
    # centre the legend on the lower axis
    axis_object.legend(ordered_handles, legend_order, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=len(legend_order), fontsize=12, columnspacing=1.0)
    fig.tight_layout()
    
    # add padding between the x-axis ticks and the x-axis label
    axis_object.xaxis.labelpad = 3

    # save figure
    fig.savefig(f"../Data/analysis_results/graphs/overall_median_rmse_{group}.png", dpi=300)

    # plot families
    plot_grouped(filtered, "circuit_family", "Median RMSE Per Family", aggregate_fn, proportion_order, algorithm_order, cols_count=2, group=group, full_frame=result_frame)
    
    # plot depths
    plot_grouped(filtered, "d", "Median RMSE Per Depth", aggregate_fn, proportion_order, algorithm_order, cols_count=2, group=group, full_frame=result_frame)
    
    # plot lengths
    plot_grouped(filtered, "n", "Median RMSE Per Length", aggregate_fn, proportion_order, algorithm_order, cols_count=2, group=group, full_frame=result_frame)
    
# define bootstrap estimator
def bootstrap(values):
    
    # convert to array
    arr = np.asarray(values)

    # create random generator
    rng = np.random.default_rng(42)
    
    # collect bootstrap medians
    resampled = [np.median(rng.choice(arr, size=arr.size, replace=True)) for _ in range(bootstrap_iters)]
    
    return np.median(resampled)

def main():
   
    # set aggregation function
    aggregate_fn = bootstrap
    
    # set proportion order
    proportion_order = [0, 25, 50, 75, 100]
    
    # load results
    results = load_results()
    
    # # draw charts for first group
    # chart_builder(results, aggregate_fn, proportion_order, ["RF", "GBR", "DT"], group=1)
    
    # # draw charts for second group
    # chart_builder(results, aggregate_fn, proportion_order, ["RF", "RFRetuned", "rt"], group=2)
    
    # draw charts for combined group
    chart_builder(results, aggregate_fn, proportion_order, ["RF", "GBR", "DT", "RFRetuned", "rt"], group=3)
    
    # show all plots
    plt.show()
    
    print("Images saved to file")

if __name__ == "__main__":
    main()