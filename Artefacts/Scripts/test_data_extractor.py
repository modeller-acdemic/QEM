# -*- coding: utf-8 -*-
"""

This collects the raw test data from the quantum hardware and saves it as a csv file.

"""

import json
import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit.quantum_info import Pauli, Clifford, random_clifford, random_pauli_list
from qiskit.qasm2 import dumps as qasm2_dumps
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler


# sets up the service backend and sampler
def sampler(account_name, backend_name, shots_per_circuit):
    backend_hardware = None
    sampler_hardware = None
    
    # create the runtime service using the saved account
    service = QiskitRuntimeService(name=account_name)
    
    # select the hardware backend by name
    backend_hardware = service.backend(backend_name)
    
    # create a sampler bound to the backend
    sampler_hardware = Sampler(mode=backend_hardware)
    
    # set the default number of shots
    sampler_hardware.options.default_shots = shots_per_circuit
    
    # disable measure twirling
    sampler_hardware.options.twirling.enable_measure = False
    
    # disable dynamical decoupling
    sampler_hardware.options.dynamical_decoupling.enable = False
    
    return backend_hardware, sampler_hardware

# builds a layer of pairwise cliffords plus a trailing single qubit
def pairwise_layers(qubits, random_generator):
    layer_operations = []
    
    # iterate over disjoint neighbour pairs on a line topology
    for left_qubit_index in range(0, qubits - 1, 2):
        
        # draw a random seed and build a two qubit clifford gate
        random_seed = int(random_generator.integers(0, 2**31 - 1))
        two_qubit_clifford = random_clifford(2, seed=random_seed)
        
        # append the two qubit gate and its qubit arguments
        layer_operations.append((two_qubit_clifford.to_circuit().to_gate(), [left_qubit_index, left_qubit_index + 1]))
    
    # draw a single qubit clifford on the last qubit if the count is odd
    if qubits % 2 == 1:
        
        # draw a random seed and build a single qubit clifford gate
        random_seed = int(random_generator.integers(0, 2**31 - 1))
        one_qubit_clifford = random_clifford(1, seed=random_seed)
        
        # append the single qubit gate and its target
        layer_operations.append((one_qubit_clifford.to_circuit().to_gate(), [qubits - 1]))
    
    return layer_operations

# builds d layers of pairwise cliffords for a given configuration
def pairwise_layerss(qubits, depth, random_generator):
    pairwise_layers = []
    
    # repeat the layer construction depth times
    for _ in range(depth):
        
        # build one layer using the helper
        layer_operations = pairwise_layers(qubits, random_generator)
        
        # append the completed layer
        pairwise_layers.append(layer_operations)
    
    return pairwise_layers

# assembles an rb circuit and appends the global inverse
def rb_circuits(qubits, pairwise_layers):
    
    # create the execution circuit with classical registers for measurement
    quantum_circuit = QuantumCircuit(qubits, qubits)
    
    # create the forward tracker circuit to accumulate the clifford layers
    forward_tracker = QuantumCircuit(qubits)
        
    # append all forward layers to both circuits
    for layer in pairwise_layers:
        for gate_object, qubit_args in layer:
            quantum_circuit.append(gate_object, qubit_args)
            forward_tracker.append(gate_object, qubit_args)
    
    # form the composed forward clifford and append its inverse
    forward_clifford = Clifford(forward_tracker)
    quantum_circuit.append(forward_clifford.adjoint().to_circuit().to_gate(), range(qubits))
    
    # measure all qubits
    quantum_circuit.measure(range(qubits), range(qubits))
    
    # no pauli label is used in rb circuits
    pauli_label_for_record = None
    
    # the target bitstring is always all zeros in rb circuits
    target_bitstring_for_record = "0" * qubits
    
    return quantum_circuit, pauli_label_for_record, target_bitstring_for_record

# the mirror circuit method with interleaved pauli layers and a global inverse
def mirror_circuits(qubits, pairwise_layers, random_generator):
    
    # create the execution circuit with classical registers for measurement
    quantum_circuit = QuantumCircuit(qubits, qubits)
    
    # create the forward tracker circuit used to accumulate the clifford
    forward_tracker = QuantumCircuit(qubits)
    
    # initialise the cumulative x and z mask for the logical pauli
    cumulative_x_mask = np.zeros(qubits, dtype=bool)
    cumulative_z_mask = np.zeros(qubits, dtype=bool)
            
    # iterate each stored layer and interleave logical pauli in the forward frame
    for layer in pairwise_layers:
        
        # append the forward layer to both circuits
        for gate_object, qubit_args in layer:
            quantum_circuit.append(gate_object, qubit_args)
            forward_tracker.append(gate_object, qubit_args)
        
        # compute the forward clifford realised so far
        forward_clifford_so_far = Clifford(forward_tracker)
        
        # sample one n qubit logical pauli label
        random_seed = int(random_generator.integers(0, 2**31 - 1))
        sampled_pauli_list = random_pauli_list(qubits, seed=random_seed)
        
        # form the logical pauli object and update the cumulative masks
        logical_pauli_layer = Pauli(sampled_pauli_list.to_labels()[0])
        cumulative_x_mask ^= logical_pauli_layer.x
        cumulative_z_mask ^= logical_pauli_layer.z
        
        # conjugate the logical pauli into the physical layer in the schrodinger frame
        conjugated_pauli = logical_pauli_layer.evolve(forward_clifford_so_far, frame='s')
        
        # apply x where required by the conjugated pauli
        for qubit_index in range(qubits):
            if conjugated_pauli.x[qubit_index]:
                quantum_circuit.x(qubit_index)
        
        # apply z where required by the conjugated pauli
        for qubit_index in range(qubits):
            if conjugated_pauli.z[qubit_index]:
                quantum_circuit.z(qubit_index)
    
    # append the inverse of the total forward clifford and measure
    forward_clifford = Clifford(forward_tracker)
    quantum_circuit.append(forward_clifford.adjoint().to_circuit().to_gate(), range(qubits))
    quantum_circuit.measure(range(qubits), range(qubits))
    
    # construct the cumulative logical pauli and derive metadata
    cumulative_logical_pauli = Pauli((cumulative_z_mask, cumulative_x_mask))
    pauli_label_for_record = cumulative_logical_pauli.to_label()
    target_bitstring_for_record = "".join("1" if bit else "0" for bit in cumulative_x_mask)
    
    return quantum_circuit, pauli_label_for_record, target_bitstring_for_record

# create all records for submission and storage
def build_records(circuit_families, qubit_counts, depths, instance_identifiers, backend_hardware, random_seed):
    built_records = []
    random_generator = np.random.default_rng(random_seed)
    
    # iterate over families, qubit counts, depths, and instances
    for circuit_family in circuit_families:
        for qubits in qubit_counts:
            for depth in depths:
                for instance_identifier in instance_identifiers:
                    
                    # build all forward layers for this configuration
                    pairwise_layers = pairwise_layerss(qubits, depth, random_generator)
                    
                    # assemble the circuit for the selected family
                    if circuit_family == "RB":
                        quantum_circuit, pauli_label_for_record, target_bitstring_for_record = rb_circuits(qubits, pairwise_layers)
                    else:
                        quantum_circuit, pauli_label_for_record, target_bitstring_for_record = mirror_circuits(qubits, pairwise_layers, random_generator)
                    
                    # transpile the circuit to the target backend
                    quantum_circuit = transpile(quantum_circuit, backend=backend_hardware, optimization_level=1)
                    
                    # append the record containing circuit metadata, circuit object, and expected outputs
                    built_records.append({
                        "family": circuit_family,
                        "num_qubits": qubits,
                        "depth": depth,
                        "instance_id": instance_identifier,
                        "circuit": quantum_circuit,
                        "pauli_label": pauli_label_for_record,
                        "target_bitstring": target_bitstring_for_record
                    })
    
    return built_records

# sort helper
def first_key(item_pair):
    return item_pair[0]

# execute circuits, collect results, and serialise rows
def run_collect(sampler_hardware, built_records, shots_per_circuit):
    csv_rows = []
    all_circuits_for_submission = [record["circuit"] for record in built_records]
    
    # submit all circuits to the runtime sampler
    job_handle = sampler_hardware.run(all_circuits_for_submission, shots=shots_per_circuit)
    
    # obtain the completed results object
    job_result_object = job_handle.result()
    
    # keep a backup of the raw results
    job_results_list = list(job_result_object)
    np.save("job_results_backup.npy", np.array(job_results_list, dtype=object))
    
    # align outputs with inputs and build rows
    for record, publication in zip(built_records, job_result_object):
        
        # obtain counts and convert to a plain dictionary
        publication_counts = publication.join_data().get_counts()
        merged_counts = dict(publication_counts)
        
        # produce a stable key order and serialise to json
        sorted_items_for_json = sorted(merged_counts.items(), key=first_key)
        histogram_json_text = json.dumps(dict(sorted_items_for_json))
        
        # serialise the circuit to qasm text
        circuit_qasm_text = qasm2_dumps(record["circuit"])
        
        # append the row to the results table with metadata and serialised outputs
        csv_rows.append({
            "circuit_family": record["family"],
            "n": record["num_qubits"],
            "d": record["depth"],
            "random_instance_identifier": record["instance_id"],
            "circuit_description": circuit_qasm_text,
            "raw_counts_histogram_json": histogram_json_text,
            "total_shots": shots_per_circuit,
            "pauli_label": record["pauli_label"]
        })
    
    return csv_rows

def main():
    
    # set the number of shots for each circuit execution
    shots_per_circuit = 10000
    
    # define the circuit families to generate
    circuit_families = ["RB", "mirror"]
    
    # define the qubit counts to test
    qubit_counts = [3, 5]
    
    # define the depth settings to test
    depths = [1, 3, 5, 7, 9]
    
    # define the random instance identifiers
    instance_identifiers = [1, 2, 3, 4]
    
    # requires credentials before the script will work
    ibm_platform_token = 'removed'
    ibm_platform_instance = 'removed'
    account_name = 'ibm_quantum_platform'
    
    # set the backend name to run on
    backend_name = "ibm_torino"
    random_seed = 42 # original random seed
    # random_deed = 54321 # retest random seed
    
    # set the output path for the results file
    output_path = "../Data/test_data/test_data_raw_rt.csv"
    
    # save the credentials for later use
    QiskitRuntimeService.save_account(channel='ibm_quantum_platform', token=ibm_platform_token, instance=ibm_platform_instance, overwrite=True, name=account_name)
    
    # construct the backend and sampler
    backend_hardware, sampler_hardware = sampler(account_name, backend_name, shots_per_circuit)
    
    # create all circuit records
    built_records = build_records(circuit_families, qubit_counts, depths, instance_identifiers, backend_hardware, random_seed)
    
    # run the circuits and collect serialised rows
    csv_rows = run_collect(sampler_hardware, built_records, shots_per_circuit)
    
    # write the results file
    results_dataframe = pd.DataFrame(csv_rows, columns=[
        "circuit_family",
        "n",
        "d",
        "random_instance_identifier",
        "circuit_description",
        "raw_counts_histogram_json",
        "total_shots",
        "pauli_label"
    ])
    results_dataframe.to_csv(output_path, index=False)
    
    print(f"saved results for {len(csv_rows)} circuits to {output_path}")

if __name__ == "__main__":
    main()
