# -*- coding: utf-8 -*-
"""

This code runs an omnibus permutation test with a studentised max-t statistic 
and runs pairwise Wilcoxon signed-rank tests with Benjamini-Hochberg correction.
It then prints readable summaries and writes p-value matrices to CSV 

"""
import math
import numpy
import pandas
import scikit_posthocs as posthocs

# bootstrap median paired differences
def bootstrap(long_df, factor_labels, metric_name):
    result_rows = []
    
    # reshape to wide format
    wide_table = long_df.pivot(index="instance_key", columns="factor", values="value")
    
    # convert to a numeric matrix for fast resampling
    value_matrix = wide_table.to_numpy()
    
    # build the ordered factor labels used for pair generation
    factor_order = [f"{label}%" for label in factor_labels]
    
    # map each factor label to its column index in the matrix
    label_index = {label: wide_table.columns.get_loc(label) for label in factor_order}
    
    # create a reproducible random generator
    random_generator = numpy.random.default_rng(42)
    
    # count the number of paired instances
    instance_count = value_matrix.shape[0]
    
    # set fixed bootstrap configuration
    bootstrap_count = 10000
    lower_percentile = 2.5
    upper_percentile = 97.5
    
    # set fixed permutation configuration for the p value
    permutation_count = 10000
    
    # iterate over unique ordered factor pairs
    for first_position in range(len(factor_order)):
        for second_position in range(first_position + 1, len(factor_order)):
            
            # read the two factor labels for this pair
            first_label = factor_order[first_position]
            second_label = factor_order[second_position]
            
            # extract the two value columns
            first_values = value_matrix[:, label_index[first_label]]
            second_values = value_matrix[:, label_index[second_label]]
            
            # compute paired differences
            paired_differences = second_values - first_values
            
            # compute the observed median difference
            median_difference = float(numpy.median(paired_differences))
            
            # compute the percentage change relative to the second group
            percentage_change = median_difference / ((float(numpy.median(second_values)) * 100.0) + 1e-10)
                     
            # allocate bootstrap storage
            bootstrap_values = numpy.empty(bootstrap_count, dtype=float)
            
            # resample instances with replacement and compute the median each time
            for bootstrap_index in range(bootstrap_count):
                sample_index = random_generator.integers(0, instance_count, size=instance_count)
                resample_differences = paired_differences[sample_index]
                bootstrap_values[bootstrap_index] = float(numpy.median(resample_differences))
            
            # compute percentile confidence limits
            ci_lower = float(numpy.percentile(bootstrap_values, lower_percentile))
            ci_upper = float(numpy.percentile(bootstrap_values, upper_percentile))
            
            # run sign-flip permutation test for the median difference
            exceed_count = 0
            for permutation_index in range(permutation_count):
                flip_signs = (random_generator.integers(0, 2, size=instance_count) * 2) - 1
                permuted_median = float(numpy.median(paired_differences * flip_signs))
                if abs(permuted_median) >= abs(median_difference):
                    exceed_count += 1
            p_value = (exceed_count + 1.0) / (permutation_count + 1.0)
            
            # append a single record containing print fields and csv fields
            result_rows.append({
                "first_factor": first_label,
                "second_factor": second_label,
                "metric": metric_name,
                "median": median_difference,
                "percentage_change": percentage_change,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "p_value": p_value,
                "result_type": "pairwise_median",
                "value": median_difference
            })
    
    return result_rows

# compute the studentised max-t statistic from a wide instances-by-factors matrix
def studentised_t(wide_matrix):
    
    # compute mean per factor across instances
    means = numpy.nanmean(wide_matrix, axis=0)
    
    # compute grand mean across all entries
    grand_mean = numpy.nanmean(wide_matrix)
    
    # compute sample standard deviation per factor
    sample_sds = numpy.nanstd(wide_matrix, axis=0, ddof=1)
    
    # read number of instances from matrix shape
    number_of_instances = wide_matrix.shape[0]
    
    # compute standard error per factor using sample standard deviation and instance count
    standard_errors = sample_sds / math.sqrt(number_of_instances)
    
    # replace zero standard errors with a small epsilon to avoid division by zero
    standard_errors = numpy.where(standard_errors == 0, 1e-12, standard_errors)
    
    # compute studentised t values per factor relative to grand mean
    t_values = (means - grand_mean) / standard_errors
    
    # compute maximum absolute t value as omnibus statistic
    max_abs_t = float(numpy.nanmax(numpy.abs(t_values)))
    
    return max_abs_t

# run the permutation-based omnibus test using label shuffling within instances
def permutation_test(long_df, number_of_permutations=10000, random_seed=42):
    exceed_count = 0
    
    # pivot the long-format dataframe to a wide matrix of shape instances by factors
    pivot_df = long_df.pivot(index="instance_key", columns="factor", values="value")
    
    # convert the pivot table to a numpy array for fast permutation operations
    original_matrix = pivot_df.to_numpy()
    
    # compute observed studentised max-t statistic on the original matrix
    observed_stat = studentised_t(original_matrix)
    
    # initialise a random number generator with the provided seed
    rng = numpy.random.default_rng(random_seed)
    
    # read instance and factor counts from the matrix shape
    instance_count, factor_count = original_matrix.shape

    # create a working matrix buffer for permuted rows
    working_matrix = original_matrix.copy()
    
    # iterate over the requested number of permutations to approximate the null distribution
    for permutation_index in range(number_of_permutations):
        
        # permute factor labels independently within each instance to respect dependence
        for instance_index in range(instance_count):
            perm_indices = rng.permutation(factor_count)
            working_matrix[instance_index, :] = original_matrix[instance_index, perm_indices]
        
        # compute the studentised max-t statistic for the permuted matrix
        permuted_stat = studentised_t(working_matrix)
        
        # increment exceedance counter when the permuted statistic equals or exceeds the observed statistic
        if permuted_stat >= observed_stat:
            exceed_count += 1
    
    # compute one-sided p-value with plus-one correction for finite sampling
    p_value = (exceed_count + 1.0) / (number_of_permutations + 1.0)
    
    return observed_stat, p_value, pivot_df

# compute pairwise Wilcoxon signed-rank tests
def pairwise_wilcoxon(long_df):
    
    # wide table where rows are instances and columns are factors
    pivot_df = long_df.pivot(index="instance_key", columns="factor", values="value")
    
    # factor labels for the final matrix
    factor_labels = list(pivot_df.columns)
    
    # numpy array with factors along the first axis
    array_for_test = pivot_df.to_numpy().T
    
    # perform pairwise Wilcoxon tests with Benjamini-Hochberg correction
    p_matrix = posthocs.posthoc_wilcoxon(array_for_test, p_adjust="fdr_bh")
    
    # restore factor labels on rows and columns
    p_matrix.index = factor_labels
    p_matrix.columns = factor_labels
    
    # collect unique factor pairs with  p-values
    significant_pairs = [((first_label, second_label), float(p_matrix.loc[first_label, second_label])) for first_label in factor_labels for second_label in factor_labels if first_label < second_label]
    
    return significant_pairs, factor_labels

def main():
    
    # configure analysis
    num_permutations = 10000
    seed = 42
    output_path = "../Data/analysis_results/analysis_results_all.csv"
    
    # set input location
    base_directory_path = "../Data/test_results"
    
    # set factor labels
    factors = ["0", "25", "50", "75", "100"]
    
    # map algorithms to suffixes
    algo_suffixes = {
        "RF": "",
        "RFRetuned": "_RFRetuned",
        "DT": "_DT",
        "GBR": "_GBR",
        "RF Retest": "_rt"
    }
    
    # load all algorithm × factor files
    factor_frames = []
    for algo_name, algo_suffix in algo_suffixes.items():
        for factor_label in factors:
            csv_file_path = f"{base_directory_path}/test_results_{factor_label}{algo_suffix}.csv"
            df = pandas.read_csv(csv_file_path)
            df["factor"] = f"{factor_label}%"
            df["algorithm"] = algo_name
            df["instance_key"] = list(zip(df["circuit_family"], df["n"], df["d"], df["random_instance_identifier"]))
            factor_frames.append(df)
    
    # combine all rows
    combined_frame = pandas.concat(factor_frames, ignore_index=True)
    
    # set factor display order
    order_by_factor = {f"{label}%": index for index, label in enumerate(factors)}
    
    results_rows = []
    
    # run analyses per algorithm
    for algo_name in algo_suffixes.keys():
        
        # select rows for algorithm
        sub = combined_frame[combined_frame["algorithm"] == algo_name]
        
        # build long table for rmse
        long_df = sub[["instance_key", "factor", "rmse"]].rename(columns={"rmse": "value"})
        long_df["metric"] = "rmse"
        long_df = long_df[["instance_key", "factor", "metric", "value"]]
        
        # permutation test for rmse
        observed_stat, p_value, _ = permutation_test(long_df, number_of_permutations=num_permutations, random_seed=seed)
        print(f"\n{algo_name} RMSE Permutation Test")
        print(f"studentised max-t statistic: {observed_stat:.6f}")
        print(f"permutation p-value with {num_permutations} permutations: {p_value:.6f}")
        results_rows.append({"result_type": "permutation_test", "metric": "rmse", "value": observed_stat, "p_value": p_value, "algorithm": algo_name})
        
        # pairwise Wilcoxon for rmse
        significant_pairs, _ = pairwise_wilcoxon(long_df)
        print(f"\n{algo_name} RMSE Wilcoxon")
        for (first_label, second_label), p_adj in sorted(significant_pairs, key=lambda t: (order_by_factor[t[0][0]], order_by_factor[t[0][1]])):
            print(f"{first_label} vs {second_label} p: {p_adj:.6f}")
            results_rows.append({"result_type": "pairwise", "metric": "rmse", "value": numpy.nan, "p_value": p_adj, "first_factor": first_label, "second_factor": second_label, "algorithm": algo_name})
        
        # bootstrap for rmse
        bootstrap_results = bootstrap(sub[["instance_key", "factor", "rmse"]].rename(columns={"rmse": "value"}).assign(metric="rmse")[["instance_key", "factor", "metric", "value"]], factors, "rmse")
        print(f"\n{algo_name} RMSE Bootstrap (95% CI)")
        for row in bootstrap_results:
            print(f"{row['first_factor']} vs {row['second_factor']} median: {row['median']:.6f}, % change: {row['percentage_change']:.2f}%, 95% CI: [{row['ci_lower']:.6f}, {row['ci_upper']:.6f}], p: {row['p_value']:.6f}")
            row["algorithm"] = algo_name
        results_rows.extend(bootstrap_results)
    
    # write all results
    results_df = pandas.DataFrame(results_rows)
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    
    main()
