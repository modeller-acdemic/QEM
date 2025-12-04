# -*- coding: utf-8 -*-
"""
Created on Fri Sep  5 23:52:42 2025

Initially a processing script, as I forgot to mark which of the circuits were simulated and which were drawn from the real QC
This makes a new dataframe with each recoded by examining the JSON backup file.

Then, it measures the extent of difference between the two domains using the maximum mean discrepancy (MMD) 
to determine whether special processing is needed, using the 50% split to avoid bias in calculations.
It outputs the actual MMD test statistic, the permutation test results and the asymptoptic results for triangulation.

"""

import json
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import math

# define the fraction to select the partition and file names
fraction = 0.50

# assign labels to circuit indices
def index_label(part):
    
    # get sets of hardware and simulated indices
    hardware = set(part['hardware_indices'])
    simulated = set(part['simulator_indices'])
    
    # assign source label to each circuit index
    idx_to_source = {i: 'real' for i in hardware} | {i: 'simulated' for i in simulated}
    
    # assign backend name to each circuit index
    idx_to_backend = {i: 'ibm_torino' for i in hardware} | {i: 'FakeTorino' for i in simulated}
    
    return idx_to_source, idx_to_backend

# append source columns
def append_source(df, idx_to_source, idx_to_backend):
    
    # append source columns to the dataframe
    coded_df = df.copy()
    coded_df['source'] = coded_df['circuit_index'].map(idx_to_source)
    coded_df['backend_name'] = coded_df['circuit_index'].map(idx_to_backend)
   
    return coded_df

# selects feature columns
def feature_cols(dataframe_for_test, excluded_columns_metadata):

    # identify columns present in the dataframe
    numeric_feature_columns = [column for column in dataframe_for_test.columns if np.issubdtype(dataframe_for_test[column].dtype, np.number)]
    
    # select feature columns
    selected_feature_columns = [column for column in numeric_feature_columns if column not in excluded_columns_metadata]
    
    return selected_feature_columns

# builds group matrices and pooled matrix
def feature_matrices(dataframe_for_test, selected_feature_columns):
    
    # build the feature matrix for the real group
    real_feature_matrix = dataframe_for_test.loc[dataframe_for_test['source'] == 'real', selected_feature_columns].to_numpy(copy=True)
    
    # build the feature matrix for the simulated group
    simulated_feature_matrix = dataframe_for_test.loc[dataframe_for_test['source'] == 'simulated', selected_feature_columns].to_numpy(copy=True)
    
    # create a pooled matrix
    combined_feature_matrix = np.vstack([real_feature_matrix, simulated_feature_matrix]).astype(np.float64, copy=False)
    
    return real_feature_matrix, simulated_feature_matrix, combined_feature_matrix

# runs pooled z score scaling
def pooled_scale(real_feature_matrix, simulated_feature_matrix):
    
    # create a pooled matrix
    combined_feature_matrix = np.vstack([real_feature_matrix, simulated_feature_matrix]).astype(np.float64, copy=False)
    
    # compute the column means for z score scaling
    feature_means = combined_feature_matrix.mean(axis=0)
    
    # compute the column standard deviations for z score scaling
    feature_stds = combined_feature_matrix.std(axis=0, ddof=0)
    
    # avoid division by zero
    feature_stds[feature_stds == 0.0] = 1.0
    
    # apply z score scaling to the real group
    real_scaled = (real_feature_matrix - feature_means) / feature_stds
    
    # apply z score scaling to the simulated group
    simulated_scaled = (simulated_feature_matrix - feature_means) / feature_stds
    return real_scaled, simulated_scaled


# computes pairwise squared distances
def pairwise_sq_dists(combined_feature_matrix):
    
    # compute squared norms for all pooled observations
    squared_norms_column = np.sum(combined_feature_matrix * combined_feature_matrix, axis=1, keepdims=True)
    
    # construct a full pairwise squared euclidean distance matrix
    pairwise_squared_distances = squared_norms_column + squared_norms_column.T - 2.0 * (combined_feature_matrix @ combined_feature_matrix.T)
    
    return pairwise_squared_distances

# select a gaussian bandwidth by grid search around a median distance
def gaussian_bandwidth(pairwise_squared_distances, real_sample_size, simulated_sample_size):
    
    # obtain strict upper-triangle indices to skip the diagonal
    upper_triangle_row_indices, upper_triangle_col_indices = np.triu_indices(pairwise_squared_distances.shape[0], k=1)
    
    # extract upper-triangle squared distances for the heuristic
    upper_triangle_squared_distances = pairwise_squared_distances[upper_triangle_row_indices, upper_triangle_col_indices]
    
    # compute the median squared distance as the centre of the grid
    median_squared_distance = float(np.median(upper_triangle_squared_distances)) if upper_triangle_squared_distances.size > 0 else 1.0
    
    # create a grid of sigma values around the median distance
    sigma_grid_values = np.sqrt(median_squared_distance) * np.power(2.0, np.arange(-3, 4, 1, dtype=float))
    
    # prepare pooled index ranges for the two groups
    indices_real_group = np.arange(real_sample_size)
    indices_simulated_group = np.arange(real_sample_size, real_sample_size + simulated_sample_size)
    
    # initialise trackers for the best bandwidth and statistic
    best_gaussian_kernel_bandwidth = None
    best_gaussian_kernel_gamma = None
    best_full_kernel_matrix_all = None
    best_unbiased_mmd2_value = -np.inf
    
    # evaluate the unbiased MMD^2 across the grid
    for sigma_candidate in sigma_grid_values:
        
        # compute gamma from the candidate bandwidth
        gaussian_kernel_gamma_candidate = 1.0 / (2.0 * sigma_candidate * sigma_candidate) if sigma_candidate > 0.0 else 1.0
        
        # build the gaussian gram matrix from pairwise squared distances
        full_kernel_matrix_candidate = np.exp(-gaussian_kernel_gamma_candidate * pairwise_squared_distances, dtype=np.float64)
        
        # zero the diagonal to remove self-similarity from within-group terms
        np.fill_diagonal(full_kernel_matrix_candidate, 0.0)
        
        # slice blocks for real-real, simulated-simulated, and real-simulated
        kernel_block_real_real_candidate = full_kernel_matrix_candidate[np.ix_(indices_real_group, indices_real_group)]
        kernel_block_simulated_simulated_candidate = full_kernel_matrix_candidate[np.ix_(indices_simulated_group, indices_simulated_group)]
        kernel_block_real_simulated_candidate = full_kernel_matrix_candidate[np.ix_(indices_real_group, indices_simulated_group)]
        
        # compute unbiased within-group terms
        term_real_real_candidate = kernel_block_real_real_candidate.sum() / (real_sample_size * (real_sample_size - 1)) if real_sample_size > 1 else 0.0
        term_simulated_simulated_candidate = kernel_block_simulated_simulated_candidate.sum() / (simulated_sample_size * (simulated_sample_size - 1)) if simulated_sample_size > 1 else 0.0
        
        # compute the cross-group term
        term_real_simulated_candidate = 2.0 * kernel_block_real_simulated_candidate.sum() / (real_sample_size * simulated_sample_size) if real_sample_size > 0 and simulated_sample_size > 0 else 0.0
        
        # form the unbiased quadratic-time MMD^2 statistic
        unbiased_mmd2_candidate = term_real_real_candidate + term_simulated_simulated_candidate - term_real_simulated_candidate
        
        # update the best bandwidth if the statistic improves
        if unbiased_mmd2_candidate > best_unbiased_mmd2_value:
            
            # store the best statistic value
            best_unbiased_mmd2_value = float(unbiased_mmd2_candidate)
            
            # store the corresponding bandwidth
            best_gaussian_kernel_bandwidth = float(sigma_candidate)
            
            # store the corresponding gamma
            best_gaussian_kernel_gamma = float(gaussian_kernel_gamma_candidate)
            
            # store the corresponding gram matrix
            best_full_kernel_matrix_all = full_kernel_matrix_candidate
    
    # assign the selected bandwidth and gamma
    gaussian_kernel_bandwidth = best_gaussian_kernel_bandwidth
    gaussian_kernel_gamma = best_gaussian_kernel_gamma
    
    # assign the gram matrix at the selected bandwidth
    full_kernel_matrix_all = best_full_kernel_matrix_all
    
    return gaussian_kernel_bandwidth, gaussian_kernel_gamma, full_kernel_matrix_all

# extract kernel blocks and compute unbiased terms
def kernel_blocks(full_kernel_matrix_all, real_sample_size, simulated_sample_size):
    
    # zero the diagonal to remove self-similarity from pooled matrix
    np.fill_diagonal(full_kernel_matrix_all, 0.0)
    
    # build pooled index ranges for both groups
    indices_real_group = np.arange(real_sample_size)
    indices_simulated_group = np.arange(real_sample_size, real_sample_size + simulated_sample_size)
    
    # slice blocks for real-real, simulated-simulated, and real-simulated
    kernel_block_real_real = full_kernel_matrix_all[np.ix_(indices_real_group, indices_real_group)]
    kernel_block_simulated_simulated = full_kernel_matrix_all[np.ix_(indices_simulated_group, indices_simulated_group)]
    kernel_block_real_simulated = full_kernel_matrix_all[np.ix_(indices_real_group, indices_simulated_group)]
    
    # zero the diagonals of within-group blocks
    np.fill_diagonal(kernel_block_real_real, 0.0)
    np.fill_diagonal(kernel_block_simulated_simulated, 0.0)
    
    # compute the unbiased within-real term
    term_real_real = kernel_block_real_real.sum() / (real_sample_size * (real_sample_size - 1)) if real_sample_size > 1 else 0.0
    
    # compute the unbiased within-simulated term
    term_simulated_simulated = kernel_block_simulated_simulated.sum() / (simulated_sample_size * (simulated_sample_size - 1)) if simulated_sample_size > 1 else 0.0
    
    # compute the cross-group term
    term_real_simulated = 2.0 * kernel_block_real_simulated.sum() / (real_sample_size * simulated_sample_size) if real_sample_size > 0 and simulated_sample_size > 0 else 0.0
    
    return (kernel_block_real_real, kernel_block_simulated_simulated, kernel_block_real_simulated, term_real_real, term_simulated_simulated, term_real_simulated)

# compute the asymptotic p-value using equal-sized blocks
def asymptotic_p_value(full_kernel_matrix_all, real_sample_size, simulated_sample_size):
    
    # determine the smaller group size for equal pairing
    common_sample_size = int(min(real_sample_size, simulated_sample_size))
    
    # build index ranges for equal-sized subsets
    indices_real_group = np.arange(real_sample_size)
    indices_simulated_group = np.arange(real_sample_size, real_sample_size + simulated_sample_size)
    indices_real_equal = indices_real_group[:common_sample_size]
    indices_simulated_equal = indices_simulated_group[:common_sample_size]
    
    # extract equal-sized within and cross blocks
    kernel_block_real_equal = full_kernel_matrix_all[np.ix_(indices_real_equal, indices_real_equal)].copy()
    kernel_block_simulated_equal = full_kernel_matrix_all[np.ix_(indices_simulated_equal, indices_simulated_equal)].copy()
    kernel_block_cross_equal = full_kernel_matrix_all[np.ix_(indices_real_equal, indices_simulated_equal)].copy()
    
    # zero the diagonals of within-group equal-sized blocks
    np.fill_diagonal(kernel_block_real_equal, 0.0)
    np.fill_diagonal(kernel_block_simulated_equal, 0.0)
    
    # form the combined block used for the U-statistic
    combined_kernel_block = kernel_block_real_equal + kernel_block_simulated_equal - kernel_block_cross_equal - kernel_block_cross_equal.T
    
    # compute the unbiased U-statistic from the combined block
    u_statistic_value = combined_kernel_block.sum() / (common_sample_size * (common_sample_size - 1)) if common_sample_size > 1 else 0.0
    
    # compute per-sample contrast values from row sums
    contrast_values = combined_kernel_block.sum(axis=1) / (common_sample_size - 1) if common_sample_size > 1 else np.zeros(common_sample_size, dtype=float)
    
    # estimate the variance of the contrast values
    variance_estimate = float(np.var(contrast_values, ddof=1)) if common_sample_size > 1 else 0.0
    
    # compute the asymptotic z-statistic under the null
    asymptotic_test_statistic = (np.sqrt(common_sample_size) * u_statistic_value) / (2.0 * np.sqrt(variance_estimate)) if variance_estimate > 0.0 else 0.0
    
    # compute the one-sided p-value from the complementary error function
    p_value_asymptotic = 0.5 * math.erfc(asymptotic_test_statistic / np.sqrt(2.0)) if variance_estimate > 0.0 else 1.0
    
    return p_value_asymptotic

# run a permutation test for the quadratic-time statistic
def permutation_p_value(full_kernel_matrix_all, real_sample_size, simulated_sample_size, observed_mmd_squared_quadratic, number_of_permutations):
    
    # create a random number generator with a fixed seed
    random_generator = np.random.default_rng(12345)
    
    # allocate an array to store permuted MMD^2 statistics
    permuted_mmd_statistics = np.empty(number_of_permutations, dtype=float)
    
    # iterate over permutations using pooled-index randomisation
    for permutation_index in tqdm(range(number_of_permutations), total=number_of_permutations, desc='permutation test'):
        
        # permute pooled indices for a new group assignment
        permuted_pooled_indices = random_generator.permutation(real_sample_size + simulated_sample_size)
        
        # split permuted indices into real and simulated groups
        permuted_indices_real_group = permuted_pooled_indices[:real_sample_size]
        permuted_indices_simulated_group = permuted_pooled_indices[real_sample_size:]
        
        # extract permuted within and cross blocks
        permuted_kernel_block_real_real = full_kernel_matrix_all[np.ix_(permuted_indices_real_group, permuted_indices_real_group)]
        permuted_kernel_block_simulated_simulated = full_kernel_matrix_all[np.ix_(permuted_indices_simulated_group, permuted_indices_simulated_group)]
        permuted_kernel_block_real_simulated = full_kernel_matrix_all[np.ix_(permuted_indices_real_group, permuted_indices_simulated_group)]
        
        # compute permuted within-real term
        permuted_term_real_real = permuted_kernel_block_real_real.sum() / (real_sample_size * (real_sample_size - 1)) if real_sample_size > 1 else 0.0
        
        # compute permuted within-simulated term
        permuted_term_simulated_simulated = permuted_kernel_block_simulated_simulated.sum() / (simulated_sample_size * (simulated_sample_size - 1)) if simulated_sample_size > 1 else 0.0
        
        # compute permuted cross-group term
        permuted_term_real_simulated = 2.0 * permuted_kernel_block_real_simulated.sum() / (real_sample_size * simulated_sample_size) if real_sample_size > 0 and simulated_sample_size > 0 else 0.0
        
        # store the permuted MMD^2 statistic
        permuted_mmd_statistics[permutation_index] = permuted_term_real_real + permuted_term_simulated_simulated - permuted_term_real_simulated
    
    # compute a one-sided p-value with a finite-sample correction
    p_value_permutation = (np.sum(permuted_mmd_statistics >= observed_mmd_squared_quadratic) + 1.0) / (number_of_permutations + 1.0)
    
    return p_value_permutation

def main():
    
    # build file paths
    temp_path = Path(f'../Data/training_data/cached_data/temp_{int(fraction*100)}')
    results_csv_path = Path(f'../Data/training_data/feature_engineered/engineered_results_{int(fraction*100)}.csv')
    raw_csv_path = Path(f'../Data/training_data/raw_data/raw_outputs_{int(fraction*100)}.csv')
    partition_path = temp_path / 'partition.json'
    
    # read partition json
    with open(partition_path, 'r') as f:
        part = json.load(f)
        
    # look up label from partition
    idx_to_source, idx_to_backend = index_label(part)
    
    # load results and attach labels
    results_df = pd.read_csv(results_csv_path)
    coded_results_df = append_source(results_df, idx_to_source, idx_to_backend)
    
    # load raw outputs and attach labels
    raw_df = pd.read_csv(raw_csv_path)
    coded_raw_df = append_source(raw_df, idx_to_source, idx_to_backend)
    
    # verification check
    print(coded_results_df['source'].value_counts())
    print()
    print(coded_raw_df['source'].value_counts())
    
    # set dataframe for test
    dataframe_for_test = coded_results_df
    
    # define excluded columns
    excluded_columns_metadata = {'circuit_index', 'num_qubits', 'observable', 'exp_val_noisy', 'exp_val_ideal', 'source', 'backend_name', 'shots'}
    
    # select numeric feature columns
    selected_feature_columns = feature_cols(dataframe_for_test, excluded_columns_metadata)
    
    # build feature matrices by group
    real_feature_matrix, simulated_feature_matrix, _ = feature_matrices(dataframe_for_test, selected_feature_columns)
    
    # record group sizes
    real_sample_size = real_feature_matrix.shape[0]
    simulated_sample_size = simulated_feature_matrix.shape[0]
    
    # apply pooled z-score scaling
    real_feature_matrix, simulated_feature_matrix = pooled_scale(real_feature_matrix, simulated_feature_matrix)
    
    # pool scaled matrices
    combined_feature_matrix = np.vstack([real_feature_matrix, simulated_feature_matrix]).astype(np.float64, copy=False)
    
    # compute pairwise squared distances
    pairwise_squared_distances = pairwise_sq_dists(combined_feature_matrix)
    
    # select gaussian bandwidth and gram matrix
    gaussian_kernel_bandwidth, gaussian_kernel_gamma, full_kernel_matrix_all = gaussian_bandwidth(pairwise_squared_distances, real_sample_size, simulated_sample_size)
    
    # extract kernel blocks and unbiased terms
    kernel_block_real_real, kernel_block_simulated_simulated, kernel_block_real_simulated, term_real_real, term_simulated_simulated, term_real_simulated = kernel_blocks(full_kernel_matrix_all, real_sample_size, simulated_sample_size)
    
    # compute observed MMD^2
    observed_mmd_squared_quadratic = term_real_real + term_simulated_simulated - term_real_simulated
    
    # compute asymptotic p value
    p_value_asymptotic = asymptotic_p_value(full_kernel_matrix_all, real_sample_size, simulated_sample_size)
    
    # compute permutation p value
    number_of_permutations = 2000
    p_value_permutation = permutation_p_value(full_kernel_matrix_all, real_sample_size, simulated_sample_size, observed_mmd_squared_quadratic, number_of_permutations)
    
    print(f'\nsamples real {real_sample_size} simulated {simulated_sample_size}')
    print(f'mmd2 quadratic_unbiased observed {observed_mmd_squared_quadratic:.8f}')
    print(f'permutations {number_of_permutations}')
    print(f'p_value permutation {p_value_permutation:.6f}')
    print(f'\nfeatures used {len(selected_feature_columns)}')
    print(f'kernel gaussian_rbf bandwidth {gaussian_kernel_bandwidth:.6f} gamma {gaussian_kernel_gamma:.6f}')
    print(f'p_value asymptotic_normal {p_value_asymptotic:.6f}\n')


if __name__ == '__main__':
    main()
