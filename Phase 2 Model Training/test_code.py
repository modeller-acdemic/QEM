# -*- coding: utf-8 -*-
"""
This code loads in the mitigated data and then simply checks the error.
It prints the aggregate results but saves the per-row results.

"""

import json
import pandas as pd
from math import sqrt

# loads both datasets from csv into dataframes
def load_data(unmitigated_file_path, mitigated_file_path):
    
    # read the unmitigated csv into a dataframe
    unmitigated_vals = pd.read_csv(unmitigated_file_path)

    # read the mitigated csv into a dataframe
    mitigated_dataframe = pd.read_csv(mitigated_file_path)
    
    return unmitigated_vals, mitigated_dataframe

# computes per qubit noisy z expectations from raw histograms for the unmitigated dataframe
def unmitigated(unmitigated_vals):
    unmitigated_expectations_json_list = []

    # iterate over unmitigated rows to compute true noisy expectations from raw histograms
    for row in unmitigated_vals.itertuples(index=False):
        
        # read number of qubits
        n = int(row.n)

        # parse histogram json to dictionary
        counts_dictionary = json.loads(row.raw_counts_histogram_json)

        # compute total counts
        total_counts = sum(counts_dictionary.values())

        # handle empty histogram by assigning zeros
        if total_counts == 0:
            expectations_list = [0.0] * n
        else:
            expectations_list = []

            # iterate logical qubits
            for qubit_index in range(n):
                accumulator = 0.0

                # sum weighted by empirical probabilities with endianness correction
                for bitstring_text, count in counts_dictionary.items():
                    accumulator += (1.0 if bitstring_text[n - 1 - qubit_index] == "0" else -1.0) * (count / total_counts)

                # append expectation for this qubit
                expectations_list.append(accumulator)

        # serialise expectations list to json text
        expectations_json_text = json.dumps(expectations_list)

        # append json text to results list
        unmitigated_expectations_json_list.append(expectations_json_text)

    return unmitigated_expectations_json_list

# attach expectations json and dataset markers then combine frames
def prep_combine(unmitigated_vals, mitigated_dataframe, unmitigated_expectations_json_list):
    
    # assign the computed expectations column to the unmitigated dataframe
    unmitigated_vals["corrected_expectations_json"] = unmitigated_expectations_json_list

    # ensure mitigated expectations column is string typed
    mitigated_dataframe["corrected_expectations_json"] = mitigated_dataframe["corrected_expectations_json"].astype(str)

    # add a marker column so we know which dataset a row belongs to
    unmitigated_vals["dataset_marker"] = "unmitigated"
    mitigated_dataframe["dataset_marker"] = "mitigated"

    # combine both into one dataframe
    combined_dataframe = pd.concat([unmitigated_vals, mitigated_dataframe], ignore_index=True)

    return combined_dataframe

# build per instance summaries from combined dataframe
def summarise_instances(combined_dataframe):
    all_instance_summaries = []

    # iterate once over the combined data
    for combined_row in combined_dataframe.itertuples(index=False):
       
        # store the circuit family from the row
        circuit_family = combined_row.circuit_family

        # store the number of qubits from the row
        number_of_qubits = int(combined_row.n)

        # store the depth value from the row
        circuit_depth = int(combined_row.d)

        # store the instance identifier from the row
        random_instance_identifier = int(combined_row.random_instance_identifier)

        # store the total number of shots from the row
        total_shots = int(combined_row.total_shots)

        # parse the histogram json string into a dictionary of counts
        counts_dictionary = json.loads(combined_row.raw_counts_histogram_json)

        # compute the total of the histogram values
        histogram_total = sum(counts_dictionary.values())

        # read the raw pauli label string
        pauli_label_raw = combined_row.pauli_label

        # reverse the pauli label so index 0 corresponds to qubit 0
        pauli_label_logical = pauli_label_raw[::-1] if isinstance(pauli_label_raw, str) else None

        # set the expected logical bitstring for rb circuits
        if circuit_family == "RB":
            expected_bitstring_logical = "0" * number_of_qubits
        # build expected logical bitstring for mirror using the corrected order
        elif circuit_family == "mirror":
            expected_bitstring_logical = "".join(["1" if symbol in ("X", "Y") else "0" for symbol in pauli_label_logical])
        # default to none for any other family
        else:
            expected_bitstring_logical = None

        # convert the logical bitstring to the counts key order by reversing
        expected_bitstring_counts = expected_bitstring_logical[::-1] if expected_bitstring_logical is not None else None

        # compute the probability of the expected bitstring using the counts key order
        probability_of_expected_bitstring = (counts_dictionary.get(expected_bitstring_counts, 0) / histogram_total) if histogram_total > 0 and expected_bitstring_counts is not None else 0.0


        # append the per instance summary record with the dataset marker
        all_instance_summaries.append({
            "dataset_marker": combined_row.dataset_marker,
            "circuit_family": circuit_family,
            "number_of_qubits": number_of_qubits,
            "circuit_depth": circuit_depth,
            "random_instance_identifier": random_instance_identifier,
            "total_shots": total_shots,
            "probability_of_expected_bitstring": probability_of_expected_bitstring,
            "corrected_expectations_json": combined_row.corrected_expectations_json,
            "simulated_ideal_expectations_json": combined_row.simulated_ideal_expectations_json
        })

    # return the complete list of per instance summaries
    return all_instance_summaries

# split summaries by dataset marker
def split_dataset(all_instance_summaries):
    
    # split into two lists by dataset
    unmitigated_instance_summaries = [summary for summary in all_instance_summaries if summary["dataset_marker"] == "unmitigated"]

    # create the mitigated subset
    mitigated_instance_summaries = [summary for summary in all_instance_summaries if summary["dataset_marker"] == "mitigated"]

    return unmitigated_instance_summaries, mitigated_instance_summaries

# index summaries by a composite key
def index_key(unmitigated_instance_summaries, mitigated_instance_summaries):
    unmitigated_by_key = {}
    mitigated_by_key = {}

    # populate unmitigated dictionary
    for record in unmitigated_instance_summaries:
        
        # build the composite key for the record
        key = (record["circuit_family"], record["number_of_qubits"], record["circuit_depth"], record["random_instance_identifier"])

        # assign the record into the dictionary
        unmitigated_by_key[key] = record

    # populate mitigated dictionary
    for record in mitigated_instance_summaries:
        
        # build the composite key for the record
        key = (record["circuit_family"], record["number_of_qubits"], record["circuit_depth"], record["random_instance_identifier"])

        # assign the record into the dictionary
        mitigated_by_key[key] = record

    return unmitigated_by_key, mitigated_by_key

# compute per instance outputs and global aggregates
def instance_aggregates(unmitigated_by_key, mitigated_by_key):
    
    # sorted list of all per instance keys present in both datasets
    common_keys_sorted = sorted(set(unmitigated_by_key.keys()) & set(mitigated_by_key.keys()))
    
    # containers and counters
    unmitigated_errors = []
    mitigated_errors = []
    unmitigated_rmse_errors = []
    mitigated_rmse_errors = []
    per_instance_rows = []
    total_unmitigated_shots = 0
    total_mitigated_shots = 0

    # iterate over every common instance across unmitigated and mitigated summaries
    for key in common_keys_sorted:
        
        # unpack the composite key fields
        circuit_family, number_of_qubits, circuit_depth, instance_identifier = key

        # extract the unmitigated and mitigated records for this instance
        unmitigated_record = unmitigated_by_key[key]
        mitigated_record = mitigated_by_key[key]

        # accumulate the total shots for cost normalisation
        total_unmitigated_shots += int(unmitigated_record["total_shots"])
        total_mitigated_shots += int(mitigated_record["total_shots"])

        # set the ideal expectation value for this benchmark
        ideal_expectation = 1.0

        # extract the probabilities of the expected bitstring
        probability_expected_unmitigated = float(unmitigated_record["probability_of_expected_bitstring"])
        probability_expected_mitigated = float(mitigated_record["probability_of_expected_bitstring"])

        # parse the simulator-derived ideal expectation vector from the mitigated record
        ideal_vector = json.loads(mitigated_record["simulated_ideal_expectations_json"])

        # parse the noisy unmitigated expectation vector from the unmitigated record
        unmitigated_vector = json.loads(unmitigated_record["corrected_expectations_json"])

        # parse the corrected mitigated expectation vector from the mitigated record
        mitigated_vector = json.loads(mitigated_record["corrected_expectations_json"])
        
        # compute per qubit root mean square errors for unmitigated and mitigated expectation vectors
        rmse_unmitigated = sqrt(sum((u - i) ** 2 for u, i in zip(unmitigated_vector, ideal_vector)) / len(ideal_vector))
        rmse_mitigated = sqrt(sum((m - i) ** 2 for m, i in zip(mitigated_vector, ideal_vector)) / len(ideal_vector))        
        # accumulate global errors for the improvement factor
        unmitigated_errors.append(probability_expected_unmitigated - ideal_expectation)
        mitigated_errors.append(probability_expected_mitigated - ideal_expectation)

        # accumulate global rmse values separately
        unmitigated_rmse_errors.append(rmse_unmitigated)
        mitigated_rmse_errors.append(rmse_mitigated)

        # construct the dataset
        per_instance_rows.append({
            "circuit_family": circuit_family,
            "n": number_of_qubits,
            "d": circuit_depth,
            "random_instance_identifier": instance_identifier,
            "rmse": rmse_mitigated,
            "rmse_unmitigated": rmse_unmitigated
        })

    # compute the aggregate rmse error across all instances
    mean_rmse_unmitigated = sum(unmitigated_rmse_errors) / len(unmitigated_rmse_errors)
    mean_rmse_mitigated = sum(mitigated_rmse_errors) / len(mitigated_rmse_errors)

    return mean_rmse_unmitigated, mean_rmse_mitigated, per_instance_rows

# write outputs and print aggregates
def write_outputs(per_instance_rows, mean_rmse_unmitigated, mean_rmse_mitigated, fraction, model):

    print("Aggregate across all instances")
    print(f"Average RMSE (unmitigated): {mean_rmse_unmitigated:.6f}")
    print(f"Average RMSE : {mean_rmse_mitigated:.6f}")

    # write the per instance results csv
    per_instance_results_dataframe = pd.DataFrame(per_instance_rows, columns=[
        "circuit_family", "n", "d", "random_instance_identifier", "rmse", "rmse_unmitigated"])

    # export the dataframe to csv
    output_path = fr"test_results\test_results_{int(fraction*100)}{model}.csv"
    per_instance_results_dataframe.to_csv(output_path, index=False)
    print(f"saved results_{int(fraction*100)}{model}.csv to file\n")

def main(fraction, model):
    print(f"Running with {int(fraction*100)}% empirical data...")
   
    # import data and save as dataframe
    unmitigated_file_path = "test_data_raw_rt.csv"
    mitigated_file_path = f"mitigated_results/test_data_mitigated_{int(fraction*100)}{model}.csv"
    unmitigated_vals, mitigated_dataframe = load_data(unmitigated_file_path, mitigated_file_path)

    # compute unmitigated expectations as json strings
    unmitigated_expectations_json_list = unmitigated(unmitigated_vals)

    # prepare and combine dataframes
    combined_dataframe = prep_combine(unmitigated_vals, mitigated_dataframe, unmitigated_expectations_json_list)

    # build per instance summaries
    all_instance_summaries = summarise_instances(combined_dataframe)

    # split by dataset marker
    unmitigated_instance_summaries, mitigated_instance_summaries = split_dataset(all_instance_summaries)

    # index both sets by composite key
    unmitigated_by_key, mitigated_by_key = index_key(unmitigated_instance_summaries, mitigated_instance_summaries)

    # compute aggregates and rows
    mean_rmse_unmitigated, mean_rmse_mitigated, per_instance_rows = instance_aggregates(unmitigated_by_key, mitigated_by_key)
    mean_rmse_unmitigated, mean_rmse_mitigated, per_instance_rows = instance_aggregates(unmitigated_by_key, mitigated_by_key)
    
    # write outputs and file
    write_outputs(per_instance_rows, mean_rmse_unmitigated, mean_rmse_mitigated, fraction, model)

if __name__ == "__main__":
    
    # call main for all models
    # models = ["", "_DT", "_GBR", "_RFRetuned"]
    # for model in models:
    #     for fraction_value in [0.0, 0.25, 0.50, 0.75, 1.0]:
    #         main(fraction=fraction_value, model=model)

    for fraction_value in [0.0, 0.25, 0.50, 0.75, 1.0]:
        main(fraction=fraction_value, model="_rt")