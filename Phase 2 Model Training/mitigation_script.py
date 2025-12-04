# -*- coding: utf-8 -*-
"""
This code applies a trained model to raw test data to predict mitigated expectation values.
It rebuilds the training feature structure to match the model parameters and simulates
noiseless ideal vectors for validation before writing the mitigated results.

"""

import json
import math
import itertools
import numpy as np
import pandas as pd
from joblib import load
from qiskit import QuantumCircuit
from operator import itemgetter
from qiskit_aer import AerSimulator
from qiskit.quantum_info import SparsePauliOp
from qiskit import transpile
from tqdm import tqdm

# builds the feature columns as used at training time
def feature_cols(training_results):
    
    # define the non feature columns exactly as in the training script
    non_feature_columns = ["circuit_index", "observable", "exp_val_ideal", "n", "d"]
    
    # build the base feature column list in the same order as the training csv
    training_base_feature_columns = [column_name for column_name in training_results.columns if column_name not in non_feature_columns]
    
    # build the observable dummy column list in the same order as the training csv
    training_observable_dummy_columns = pd.get_dummies(training_results["observable"], prefix="obs").columns.tolist()
    
    # create the full ordered list of inference feature columns used by the model
    inference_feature_columns = training_base_feature_columns + training_observable_dummy_columns
    
    return training_base_feature_columns, training_observable_dummy_columns, inference_feature_columns

# loads the trained model and required dataframes
def load_inputs(model_path, unmitigated_path, training_results_path):
    
    # load the trained model from disk
    model_object = load(model_path)
    
    # read the unmitigated csv into a dataframe
    unmitigated = pd.read_csv(unmitigated_path)
    
    # read the training-time feature-engineered csv
    training_results = pd.read_csv(training_results_path)
    
    return model_object, unmitigated, training_results

# compute the noisy z expectation for a single qubit from counts
def noisy_expects(counts, total_shots, number_of_qubits, observable_index):
    
    # set the running sum for the expectation calculation
    running_sum = 0
    
    # iterate over all bitstrings and accumulate with endianness correction
    for bitstring_text, count in counts.items():
        if bitstring_text[number_of_qubits - 1 - observable_index] == "1":
            running_sum -= int(count)
        else:
            running_sum += int(count)
    
    # compute the expectation value
    expectation = running_sum / total_shots

    return expectation

# determine the bin index for a gate parameter angle
def bin_angle(angle, angle_bins):
    
    # reduce the angle into the interval [0, 2pi)
    wrapped_angle = float(angle) % (2 * np.pi)
    
    # compute and return the bin index using right closed bins
    return int(np.digitize(wrapped_angle, angle_bins, right=True) - 1)

# reconstruct gate-count and parameter-bin features from a circuit
def gate_features(circuit_object, training_base_feature_columns, angle_bins):
    
    # initialise a dictionary of feature counts at zero
    feature_count = {column_name: 0 for column_name in training_base_feature_columns}
    
    # iterate over the circuit instructions
    for instruction_object in circuit_object.data:
        
        # read the operation object from the instruction
        operation_object = instruction_object.operation
        
        # read the gate name from the operation
        gate_name = operation_object.name
        
        # read the number of parameters on the operation
        number_of_parameters = len(operation_object.params)
        
        # increment count for non-parameterised gate if present
        if number_of_parameters == 0 and gate_name in feature_count:
            feature_count[gate_name] += 1
        
        # process each parameter for parameterised gates
        elif number_of_parameters > 0:
            for parameter_index, parameter_object in enumerate(operation_object.params):
                
                # compute the angle bin index
                bin_index = bin_angle(parameter_object, angle_bins)
                
                # build the binned feature name
                binned_feature_name = f"{gate_name}_param{parameter_index}_bin_{bin_index}"
                
                # increment the counter if the feature exists
                if binned_feature_name in feature_count:
                    feature_count[binned_feature_name] += 1
    
    return feature_count

# build a single inference row in exact training feature order
def build_rows(feature_count, inference_feature_columns, observable_label_text):
    
    # construct a single inference feature row dictionary initialised to zero
    single_row_feature = {column_name: 0 for column_name in inference_feature_columns}
    
    # copy base feature counts into the single row dictionary where present
    for column_name, column in feature_count.items():
        if column_name in single_row_feature:
            single_row_feature[column_name] = column
    
    # set the correct observable dummy if present
    observable_dummy_name = f"obs_{observable_label_text}"
    if observable_dummy_name in single_row_feature:
        single_row_feature[observable_dummy_name] = 1
    
    # build row dictionary in exact column order
    row = {column_name: single_row_feature[column_name] for column_name in inference_feature_columns}
     
    # build one-row dataframe using the row dictionary
    dataframe = pd.DataFrame([row], columns=inference_feature_columns)
     
    return dataframe

# convert corrected expectations to a mitigated histogram json
def expectations_hist(corrected_expectations, total_shots, number_of_qubits):
    probability = {}
    
    # compute per-qubit probability of measuring one in logical order
    per_qubit_one_probability = [0.5 * (1.0 - expectation) for expectation in corrected_expectations]
    
    # iterate over all bitstrings expressed in logical qubit order
    for bits_tuple in itertools.product([0, 1], repeat=number_of_qubits):
        
        # initialise probability for the current bitstring as 1 before multiplication
        prob = 1.0
        
        # iterate over each qubit index and its corresponding bit
        for index, bit in enumerate(bits_tuple):
            
            # if the bit is one, multiply by the probability of measuring one for this qubit
            if bit == 1:
                prob *= per_qubit_one_probability[index]
            
            # if the bit is zero, multiply by the probability of measuring zero for this qubit
            else:
                prob *= (1.0 - per_qubit_one_probability[index])
        
        # convert logical-order tuple to counts key text by reversing to little-endian
        bitstring_key_text = "".join(str(bit) for bit in bits_tuple[::-1])
        
        # assign probability in counts/key order
        probability[bitstring_key_text] = prob
    
    # convert probabilities to floating counts
    floating_counts = {key_text: probability[key_text] * total_shots for key_text in probability.keys()}
    
    # round each floating count down to the nearest integer
    integer_counts = {key_text: int(math.floor(floating_counts[key_text])) for key_text in floating_counts.keys()}
    
    # compute how many counts are missing after rounding down, to reach the total number of shots
    remainder = total_shots - sum(integer_counts.values())
    
    # compute the fractional parts of each floating count and sort them in descending order
    residuals = sorted([(key_text, floating_counts[key_text] - integer_counts[key_text]) for key_text in integer_counts.keys()], key=itemgetter(1), reverse=True)
    
    # initialise index used to select residuals cyclically
    residual_index = 0
    
    # distribute the missing counts one by one until the remainder is zero
    while remainder > 0:
        
        # select the key with the next largest fractional part, cycling if needed
        selected_key_text = residuals[residual_index % len(residuals)][0]
        
        # increase the integer count for this key by one
        integer_counts[selected_key_text] += 1
        
        # decrease the remainder by one
        remainder -= 1
        
        # move to the next entry in the residuals list
        residual_index += 1
    
    # prune zeros and serialise with lexicographic key order
    integer_counts = {key_text: count for key_text, count in integer_counts.items() if count > 0}
    mitigated_histogram_json_text = json.dumps(dict(sorted(integer_counts.items(), key=itemgetter(0))))
    
    return mitigated_histogram_json_text

# simulate the noise free state and return per qubit ideal <Z> expectations for the first num_qubits qubits
def simulate_ideal(qasm_string, num_qubits):
    labels = []
    
    # construct the circuit from a qasm string
    circuit = QuantumCircuit.from_qasm_str(qasm_string)
    
    # remove final measurements so expectation values are taken on the pre measurement state
    circuit.remove_final_measurements(inplace=True)
    
    # read the ordered list of circuit qubits
    ordered_qubits = list(circuit.qubits)
    
    # compute the number of qubits actually present in the circuit
    circuit_num_qubits = len(ordered_qubits)
    
    # determine how many qubits to evaluate, matching the dataset n
    target_qubit_count = min(int(num_qubits), circuit_num_qubits)
    
    # iterate over each qubit index up to the dataset n
    for qubit_index in range(target_qubit_count):
        
        # build a single qubit Z operator
        observable = SparsePauliOp('Z')
        
        # build a unique label for this save instruction
        label = f"expval_z_{qubit_index}"
        
        # remember the label for later retrieval
        labels.append(label)
        
        # append the save expectation instruction targeted to this qubit
        circuit.save_expectation_value(observable, [ordered_qubits[qubit_index]], label=label)
    
    # create a noise free stabilizer simulator to avoid memory blowups
    simulator = AerSimulator(method='stabilizer')
    
    # transpile the circuit for the simulator
    transpiled_circuit = transpile(circuit, simulator, optimization_level=0)
    
    # execute the circuit on the simulator
    result = simulator.run(transpiled_circuit, shots=1).result()
    
    # extract the saved expectation values in the same order as inserted
    data_dict = result.data()
    
    # build the list of ideal expectation values
    ideal_values = [float(data_dict[label]) for label in labels]
    
    return ideal_values

# mitigate a single input row and return the output record
def mitigate_row(row_tuple, model_object, training_base_feature_columns, inference_feature_columns, angle_bins, model):
    corrected_expectations = []
    
    # read circuit metadata and parameters from the row tuple
    circuit_family = row_tuple.circuit_family
    number_of_qubits = int(row_tuple.n)
    circuit_depth = int(row_tuple.d)
    random_instance_identifier = int(row_tuple.random_instance_identifier)
    total_shots = int(row_tuple.total_shots)
    pauli_label = row_tuple.pauli_label
    circuit_description_text = row_tuple.circuit_description
    
    # parse the histogram json into a dictionary of counts
    counts = json.loads(row_tuple.raw_counts_histogram_json)
    
    # create the quantum circuit object from the qasm description
    circuit_object = QuantumCircuit.from_qasm_str(circuit_description_text)
    
    # reconstruct gate-count features exactly as in training
    feature_count = gate_features(circuit_object, training_base_feature_columns, angle_bins)
    
    # set the number of qubits metadata feature if present
    if "num_qubits" in feature_count:
        feature_count["num_qubits"] = number_of_qubits
    
    # iterate over each single-qubit z observable index
    for observable_index in range(number_of_qubits):
        
        # compute the noisy z expectation for this qubit from the counts
        noisy_expectation = noisy_expects(counts, total_shots, number_of_qubits, observable_index)
        
        # set the noisy expectation feature
        feature_count["exp_val_noisy"] = noisy_expectation

        # build the observable label string
        observable_label_text = f"Z{observable_index}"
        
        # build a one-row dataframe in exact feature order
        inference_row = build_rows(feature_count, inference_feature_columns, observable_label_text)
        
        # obtain the predicted error from the model
        predicted_ideal = float(model_object.predict(inference_row)[0])
        
        # compute the corrected z expectation
        corrected_expectation = predicted_ideal
        
        # append the corrected expectation to the per-qubit list
        corrected_expectations.append(corrected_expectation)
    
    # convert corrected expectations to a mitigated histogram json
    mitigated_histogram_json_text = expectations_hist(corrected_expectations, total_shots, number_of_qubits)
    
    # serialise expectations for analysis parity
    corrected_expectations_json_text = json.dumps(corrected_expectations)
    
    # get the ideal expectation values from the noise-free simulator
    simulated_ideal = simulate_ideal(circuit_description_text, number_of_qubits)
    simulated_ideal_json_text = json.dumps(simulated_ideal)
    
    # build and return the mitigated row
    return {
        "circuit_family": circuit_family,
        "n": number_of_qubits,
        "d": circuit_depth,
        "random_instance_identifier": random_instance_identifier,
        "circuit_description": circuit_description_text,
        "raw_counts_histogram_json": mitigated_histogram_json_text,
        "total_shots": total_shots,
        "pauli_label": pauli_label,
        "corrected_expectations_json": corrected_expectations_json_text,
        "simulated_ideal_expectations_json": simulated_ideal_json_text
    }

# computes the RMSE between a vector of expectations and the corresponding ideal expectations
def rmse_error(expectation, ideal_expectation):
    
    # convert inputs to numpy arrays of type float
    expectations_array = np.array(expectation, dtype=float)
    ideal_array = np.array(ideal_expectation, dtype=float)
    
    # compute the rmse norm of the difference
    difference_array = expectations_array - ideal_array
    rmse = np.linalg.norm(difference_array, ord=2)
    
    # compute the per qubit rmse
    per_qubit_rmse = rmse / np.sqrt(len(difference_array))
    
    return per_qubit_rmse

# computes mean rmse across all circuits for both unmitigated and mitigated data
def compute_rmse(unmitigated, mitigated):
    unmitigated_rmse_sum = 0.0
    mitigated_rmse_sum = 0.0
    number_of_rows = 0
    
    # iterate over paired rows assuming identical ordering
    for unmitigated_row, mitigated_row in zip(unmitigated.itertuples(index=False), mitigated.itertuples(index=False)):
        
        # read the number of qubits and total shots
        number_of_qubits = int(unmitigated_row.n)
        total_shots = int(unmitigated_row.total_shots)
        
        # parse the unmitigated counts histogram
        counts = json.loads(unmitigated_row.raw_counts_histogram_json)
        
        # compute the per qubit noisy expectations from counts
        noisy_expectations = [noisy_expects(counts, total_shots, number_of_qubits, observable_index) for observable_index in range(number_of_qubits)]
        
        # read the ideal and mitigated expectations
        ideal_expectations = json.loads(mitigated_row.simulated_ideal_expectations_json)
        mitigated_expectations = json.loads(mitigated_row.corrected_expectations_json)
        
        # compute the rmse error for the mitigated and unmitigated vector against ideal
        unmitigated_rmse = rmse_error(noisy_expectations, ideal_expectations)
        mitigated_rmse = rmse_error(mitigated_expectations, ideal_expectations)
        
        # accumulate the rmse errors
        unmitigated_rmse_sum += unmitigated_rmse
        mitigated_rmse_sum += mitigated_rmse
        
        number_of_rows += 1
    
    # compute mean rmse across all circuits
    mean_unmitigated_rmse = unmitigated_rmse_sum / max(1, number_of_rows)
    mean_mitigated_rmse = mitigated_rmse_sum / max(1, number_of_rows)
    
    return mean_unmitigated_rmse, mean_mitigated_rmse

def main(fraction, model):
    print(f"Running model mitigation with {int(fraction*100)}% empirical data...")
    mitigated_rows = []
    
    # input paths
    model_path = f"models/model_{int(fraction*100)}{model}.joblib"
    unmitigated_path = "test_data_raw_rt.csv"
    training_results_path = rf"..\Phase 1 Circuit Definitions\Data\engineered_results_{int(fraction*100)}.csv"
    print(f"loaded model: {model_path}")
    
    # define the angular bin edges used for parameterised gates
    angle_bins = [0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi]
    
    # load model and inputs
    model_object, unmitigated, training_results = load_inputs(model_path, unmitigated_path, training_results_path)
      
    # recover feature columns
    training_base_feature_columns, training_observable_dummy_columns, inference_feature_columns = feature_cols(training_results)
    
    # iterate over rows to apply mitigation with a progress bar
    for row_tuple in tqdm(unmitigated.itertuples(index=False), total=len(unmitigated), desc="Simulating circuits"):
        
        # mitigate the single row
        mitigated_row = mitigate_row(row_tuple, model_object, training_base_feature_columns, inference_feature_columns, angle_bins, model=model)
        
        # append to the results list
        mitigated_rows.append(mitigated_row)
    
    # create a dataframe from mitigated rows
    mitigated = pd.DataFrame(mitigated_rows, columns=[
        "circuit_family",
        "n",
        "d",
        "random_instance_identifier",
        "circuit_description",
        "raw_counts_histogram_json",
        "total_shots",
        "pauli_label",
        "corrected_expectations_json",
        "simulated_ideal_expectations_json"
    ])
    
    # build the mitigated output path
    mitigated_output_path = f"mitigated_results/test_data_mitigated_{int(fraction*100)}{model}_rt.csv"
    mitigated.to_csv(mitigated_output_path, index=False)
    print(f"\nwrote {len(mitigated)} rows to {mitigated_output_path}")
    
    # compute errors
    mean_unmit_rmse, mean_mit_rmse = compute_rmse(unmitigated, mitigated)
    print(f"mean RMSE unmitigated: {mean_unmit_rmse:.6f}")
    print(f"mean RMSE mitigated: {mean_mit_rmse:.6f}")
    print("overall_RMSE_all_circuits", np.sqrt(np.mean((np.concatenate([np.array(json.loads(r.corrected_expectations_json), dtype=float) for r in mitigated.itertuples(index=False)]) - np.concatenate([np.array(json.loads(r.simulated_ideal_expectations_json), dtype=float) for r in mitigated.itertuples(index=False)]))**2)))
    print()
    
    return mean_mit_rmse
    
if __name__ == "__main__":
    
    # call main for all models
    # models = ["", "_DT", "_GBR", "_RFRetuned"]
    # for model in models:
    #     for fraction_value in [0.0, 0.25, 0.50, 0.75, 1.0]:
    #         main(fraction=fraction_value, model=model)

    for fraction_value in [0.0, 0.25, 0.50, 0.75, 1.0]:
        main(fraction=fraction_value, model="")