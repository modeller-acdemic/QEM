# -*- coding: utf-8 -*-
"""
Created on Thu Sep 11 20:34:12 2025

combines that messy analysis csv file into a series of neat tables for importing into Excel

"""

import pandas


# define the fixed factor comparison pairs
pair_order = [
    ("0%", "25%"),
    ("0%", "50%"),
    ("0%", "75%"),
    ("0%", "100%"),
    ("25%", "50%"),
    ("25%", "75%"),
    ("25%", "100%"),
    ("50%", "75%"),
    ("50%", "100%"),
    ("75%", "100%")
]

# define the fixed model labels
model_labels = ["RF", "Retuned RF", "DT", "GBR", "RF Retest"]

# map model labels to algorithm values
model_to_algorithm = {
    "RF": "RF",
    "Retuned RF": "RFRetuned",
    "DT": "DT",
    "GBR": "GBR",
    "RF Retest": "RF Retest"
}

# extract wilcoxon p values for a given metric
def extract_wilcoxon_p(data_model, metric_name):
    
    # filter rows for pairwise results of the metric
    data_filtered = data_model[(data_model["result_type"] == "pairwise") & (data_model["metric"] == metric_name)]
    
    map_pair_to_p = {}
    
    # iterate over rows to populate the mapping
    for row_index, row_data in data_filtered.iterrows():
        
        # get first factor
        first_label = row_data["first_factor"]
        
        # get second factor
        second_label = row_data["second_factor"]
        
        # get p value
        p_data = float(row_data["p_value"])
        
        # add entry to mapping
        map_pair_to_p[(first_label, second_label)] = p_data
    
    return map_pair_to_p

# build wilcoxon results table for a metric
def wilcoxon_table(data_full, metric_name):
    rows_out = []
    
    # iterate over all fixed pairs
    for first_label, second_label in pair_order:
        
        # start row with factor labels
        row_dict = {"first factor": first_label, "second factor": second_label}
        
        # iterate over models in order
        for model_name in model_labels:
            
            # get algorithm name for this model
            algo_name = model_to_algorithm[model_name]
            
            # filter rows for this algorithm
            data_model = data_full[data_full["algorithm"] == algo_name]
            
            # get mapping of pair to p values
            map_pair_to_p = extract_wilcoxon_p(data_model, metric_name)
            
            # try to assign p value for the pair as written
            if (first_label, second_label) in map_pair_to_p:
                row_dict[model_name] = map_pair_to_p[(first_label, second_label)]
            
            # if missing, try the reversed pair
            elif (second_label, first_label) in map_pair_to_p:
                row_dict[model_name] = map_pair_to_p[(second_label, first_label)]
        
        # add row dictionary to results
        rows_out.append(row_dict)
    
    # create dataframe with correct column order
    data_out = pandas.DataFrame(rows_out, columns=["first factor", "second factor"] + model_labels)
    
    return data_out

# extract bootstrap statistics for a metric
def bootstrap_stats(data_model, metric_name):
    map_pair_to_stats = {}
    
    # filter rows for bootstrap results of the metric
    data_filtered = data_model[(data_model["result_type"] == "pairwise_median") & (data_model["metric"] == metric_name)]
    
    # iterate over rows to populate the mapping
    for row_index, row_data in data_filtered.iterrows():
        
        # get first factor
        first_label = row_data["first_factor"]
        
        # get second factor
        second_label = row_data["second_factor"]
        
        # get effect size
        effect_data = float(row_data["median"])
        
        # get percentage change
        change_data = float(row_data["percentage_change"])
        
        # get p value
        p_data = float(row_data["p_value"])
        
        # get confidence interval lower bound
        ci_lower = float(row_data["ci_lower"])
        
        # get confidence interval upper bound
        ci_upper = float(row_data["ci_upper"])
        
        # add entry to mapping
        map_pair_to_stats[(first_label, second_label)] = (effect_data, change_data, p_data, ci_lower, ci_upper)
    
    return map_pair_to_stats

# build bootstrap results table for a metric
def bootstrap_table(data_full, metric_name):
    row_keys = []
    
    # create dictionary with empty lists for each model
    data_blocks = {model_name: [] for model_name in model_labels}
    
    # iterate over all fixed pairs
    for first_label, second_label in pair_order:
        
        # add current pair to row keys
        row_keys.append((first_label, second_label))
        
        # iterate over models in order
        for model_name in model_labels:
            
            # get algorithm name for this model
            algo_name = model_to_algorithm[model_name]
            
            # filter rows for this algorithm
            data_model = data_full[data_full["algorithm"] == algo_name]
            
            # get mapping of pair to stats
            map_pair_to_stats = bootstrap_stats(data_model, metric_name)
            
            # set key in written order
            key_pair = (first_label, second_label)
            
            # use reversed order if needed
            if key_pair not in map_pair_to_stats and (second_label, first_label) in map_pair_to_stats:
                key_pair = (second_label, first_label)
            
            # fill with missing markers if absent
            if key_pair not in map_pair_to_stats:
                effect_data = float("nan")
                change_data = float("nan")
                p_data = float("nan")
                ci_text = ""
            else:
                effect_data, change_data, p_data, ci_lower, ci_upper = map_pair_to_stats[key_pair]
                ci_text = f"[{ci_lower:.3f}, {ci_upper:.3f}]"
            
            # append values for this model
            data_blocks[model_name].append([effect_data, change_data, p_data, ci_text])
    
    # create dataframe for left columns
    data_left = pandas.DataFrame(row_keys, columns=[("","First Factor"), ("","Second Factor")])
    
    model_db = []
    
    # iterate over models
    for model_name in model_labels:
        
        # create dataframe for model values
        data_model = pandas.DataFrame(data_blocks[model_name], columns=[(model_name, "Effect Size"), (model_name, "% Change"), (model_name, "p"), (model_name, "CI")])
        
        # append model frame
        model_db.append(data_model)
    
    # concatenate all parts into one dataframe
    data_full_out = pandas.concat([data_left] + model_db, axis=1)
    
    return data_full_out

# main program entry point
def main():
    
    input_path = "../Data/analysis_results/analysis_results_all.csv"
    data_full = pandas.read_csv(input_path)
    
    wilcoxon_rmse = wilcoxon_table(data_full, "rmse")
    wilcoxon_rmse.to_csv("../Data/analysis_results/tables/wilcoxon_grouped_rmse.csv", index=False)
    print("wilcoxon_grouped_rmse.csv written to file")
    
    bootstrap_rmse = bootstrap_table(data_full, "rmse")
    bootstrap_rmse.to_csv("../Data/analysis_results/tables/bootstrapping_rmse.csv", index=False)
    print("bootstrapping_rmse.csv written to file")
    
# run main entry point
if __name__ == "__main__":
    main()
