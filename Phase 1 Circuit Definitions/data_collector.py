# -*- coding: utf-8 -*-
"""
Created on Fri Aug 29 16:38:29 2025

This is the general data collection script.
It extracts the simulated/real quantum data and the noise-free states of the same circuits using AerSimulator.
The defined fraction of real data is retrieved from the IBM quantum computer (ibm_torino), 
The rest is retreived from the simulated version of the same computer (set to FakeTorino).
However, as QC time is limited, it will save a JSON file with the sampling and batches the QC circuits, saving them as temp files.
That way, if you run out of time. You can stop the job and start again when you have more time, and it will pick up where it left off.

It outputs a couple of verification graphs and some aggregate data, to observe the transformation process;
It also outputs the transformed data for both simulated and QC, and the raw data, also used for verifying the transformation.

- Select the number of shots, the fraction of real data and the batch size.

Finally, it converts the data using the feature engineering method outlined in Liao et al. (2024) to allow for machine learning.

"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from qiskit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeTorino
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
import math
from pathlib import Path
import glob

shots = 1000
fraction = 0.0
batch_size = 500

# define four angular bins from 0 to 2*pi for categorising gate parameters
angle_bins = [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]

# identify gates and build features
def scan_gates(circuits_df, angle_bins):
    all_non_param_gates = set()
    all_param_gates = set()

    # iterate over each circuit definition to empirically identify all unique gates
    for circuit_str in circuits_df['qasm_definition']:
        
        # create a circuit object from its string representation
        circuit = QuantumCircuit.from_qasm_str(circuit_str)
        
        # iterate through each instruction in the circuit's data
        for instruction in circuit.data:
            
            # get the gate operation from the instruction
            operation = instruction.operation
            
            # get the name of the gate
            name = operation.name
            
            # filter out non-computational instructions
            if name in ['measure', 'barrier']:
                continue
            
            # get the number of parameters for the gate
            num_params = len(operation.params)
            
            # categorise the gate as parameterised or non-parameterised
            if num_params > 0:
                all_param_gates.add((name, num_params))
            else:
                all_non_param_gates.add(name)

    feature_list = sorted(list(all_non_param_gates))

    # iterate through the sorted parameterised gates to create binned feature names
    for name, num_params in sorted(list(all_param_gates)):
        
        # create a feature for each parameter of the gate
        for i in range(num_params):
            
            # create a feature for each angular bin
            for j in range(len(angle_bins) - 1):
                
                # append the formatted feature name to the list
                feature_list.append(f'{name}_param{i}_bin_{j}')
    
    return all_non_param_gates, all_param_gates, feature_list

# build circuits for a batch
def build_circuits(circuits_df, batch_idx_slice):
    
    # construct circuit objects from qasm strings
    batch_circuits = [QuantumCircuit.from_qasm_str(circuits_df.loc[idx, 'qasm_definition']) for idx in batch_idx_slice]
    
    return batch_circuits

# run a hardware batch
def hardware_run(ctx, batch_circuits):
    
    # transpile circuits for the hardware backend
    transpiled_qc_noisy_list = transpile(batch_circuits, ctx['noisy_backend'])
    
    # run the transpiled circuits on the hardware backend using sampler v2
    noisy_job = ctx['sampler_noisy'].run(transpiled_qc_noisy_list, shots=ctx['shots'])
    
    # retrieve sampler results and convert to counts per circuit
    pub_results = noisy_job.result()
    noisy_counts_list = [r.join_data().get_counts() for r in pub_results]
    
    # transpile the same circuits for the ideal backend
    transpiled_qc_ideal_list = transpile(batch_circuits, ctx['ideal_backend'])
    
    # run the transpiled circuits on the ideal simulator
    ideal_job = ctx['ideal_backend'].run(transpiled_qc_ideal_list, shots=ctx['shots'])
    
    # retrieve the measurement counts from the ideal simulator
    ideal_counts_list = ideal_job.result().get_counts()
    
    # normalise the counts object to a list
    if isinstance(ideal_counts_list, dict):
        ideal_counts_list = [ideal_counts_list]
    
    return noisy_counts_list, ideal_counts_list

# orchestrate hardware batches
def hw_loop(ctx, hardware_indices, batch_size):
    
    # compute the number of hardware batches
    num_hw_batches = math.ceil(len(hardware_indices) / batch_size)
    
    # iterate through hardware batches
    for b in tqdm(range(num_hw_batches), total=num_hw_batches, desc="running hardware batches"):
        
        # compute the slice for this batch of indices
        batch_idx_slice = hardware_indices[b * batch_size:(b + 1) * batch_size]
        
        # skip this hardware batch if its csvs already exist
        hw_results_path = ctx['temp_path'] / f'hw_batch_{b:03d}_results.csv'
        hw_raw_path = ctx['temp_path'] / f'hw_batch_{b:03d}_raw.csv'
        if hw_results_path.exists() and hw_raw_path.exists():
            continue
        
        # build the list of circuits for this batch
        batch_circuits = build_circuits(ctx['circuits_df'], batch_idx_slice)
        
        # execute on hardware and ideal backends
        noisy_counts_list, ideal_counts_list = hardware_run(ctx, batch_circuits)
        
        # engineer features and aggregate results
        batch_results, batch_raw = process_batch(ctx, batch_idx_slice, noisy_counts_list, ideal_counts_list)
        
        # write this hardware batch to csv files under temp
        save_batch(ctx['temp_path'], 'hw', b, batch_results, batch_raw)
    
    return

# transpile simulator batch
def sim_transpile(ctx, batch_circuits):
    
    # transpile the circuits for the fake torino backend
    transpiled_qc_noisy_list = transpile(batch_circuits, ctx['noisy_backend_sim'])
    
    # transpile the circuits for the ideal backend
    transpiled_qc_ideal_list = transpile(batch_circuits, ctx['ideal_backend'])
    
    return transpiled_qc_noisy_list, transpiled_qc_ideal_list

# execute simulator batch
def sim_execute(ctx, transpiled_qc_noisy_list, transpiled_qc_ideal_list):
    
    # run the transpiled circuits on the fake torino backend
    noisy_job = ctx['noisy_backend_sim'].run(transpiled_qc_noisy_list, shots=ctx['shots'])
    
    # retrieve the measurement counts from the fake torino simulator
    noisy_counts_list = noisy_job.result().get_counts()
    
    # normalise the counts object to a list
    if isinstance(noisy_counts_list, dict):
        noisy_counts_list = [noisy_counts_list]
    
    # run the transpiled circuits on the ideal simulator
    ideal_job = ctx['ideal_backend'].run(transpiled_qc_ideal_list, shots=ctx['shots'])
    
    # retrieve the measurement counts from the ideal simulator
    ideal_counts_list = ideal_job.result().get_counts()
    
    # normalise the counts object to a list
    if isinstance(ideal_counts_list, dict):
        ideal_counts_list = [ideal_counts_list]
    
    return noisy_counts_list, ideal_counts_list

# orchestrate simulator batches
def sim_loop(ctx, simulator_indices, batch_size):
    
    # compute the number of simulator batches
    num_sim_batches = math.ceil(len(simulator_indices) / batch_size)
    
    # iterate through simulator batches
    for b in tqdm(range(num_sim_batches), total=num_sim_batches, desc="simulating circuits"):
        
        # compute the slice for this batch of indices
        batch_idx_slice = simulator_indices[b * batch_size:(b + 1) * batch_size]
        
        # skip this simulator batch if its csvs already exist
        sim_results_path = ctx['temp_path'] / f'sim_batch_{b:03d}_results.csv'
        sim_raw_path = ctx['temp_path'] / f'sim_batch_{b:03d}_raw.csv'
        if sim_results_path.exists() and sim_raw_path.exists():
            continue
        
        # build the list of circuits for this batch
        batch_circuits = build_circuits(ctx['circuits_df'], batch_idx_slice)
        
        # transpile simulator and ideal versions
        transpiled_qc_noisy_list, transpiled_qc_ideal_list = sim_transpile(ctx, batch_circuits)
        
        # execute simulator and ideal jobs
        noisy_counts_list, ideal_counts_list = sim_execute(ctx, transpiled_qc_noisy_list, transpiled_qc_ideal_list)
        
        # engineer features and aggregate results
        batch_results, batch_raw = process_batch(ctx, batch_idx_slice, noisy_counts_list, ideal_counts_list)
        
        # write this simulator batch to csv files under temp
        save_batch(ctx['temp_path'], 'sim', b, batch_results, batch_raw)
    
    return

# count features and build outputs for a batch
def process_batch(ctx, batch_idx_slice, noisy_counts_list, ideal_counts_list):
    batch_results = []
    batch_raw = []

    # iterate over the circuits in this batch to perform feature engineering and build outputs
    for local_i, idx in enumerate(batch_idx_slice):
        
        # get the circuit string and create a circuit object
        circuit_str = ctx['circuits_df'].loc[idx, 'qasm_definition']
        circuit = QuantumCircuit.from_qasm_str(circuit_str)
        
        # get the counts dictionaries for this circuit
        noisy_counts = noisy_counts_list[local_i] if local_i < len(noisy_counts_list) else {}
        ideal_counts = ideal_counts_list[local_i] if local_i < len(ideal_counts_list) else {}
        
        # store the raw measurement distributions
        batch_raw.append({
            'circuit_index': idx,
            'noisy_counts': json.dumps(noisy_counts),
            'ideal_counts': json.dumps(ideal_counts)
        })
        
        # initialise a dictionary to store the gate counts for the current circuit
        gate_counts = {feature: 0 for feature in ctx['feature_list']}
    
        # count the occurrences of each gate type to build the feature vector
        for instruction in circuit.data:
            
            # get the gate operation from the instruction
            operation = instruction.operation
            
            # get the name of the instruction
            name = operation.name
            
            # increment the count if the gate is non-parameterised
            if name in ctx['all_non_param_gates']:
                gate_counts[name] += 1
                ctx['processed_gate_count'] += 1
            # process the gate if it is parameterised
            elif (name, len(operation.params)) in ctx['all_param_gates']:
                ctx['processed_gate_count'] += 1
                
                # iterate over each parameter of the gate
                for i, param in enumerate(operation.params):
                    
                    # normalise the angle to be within the 0 to 2*pi range
                    angle = float(param) % (2 * np.pi)
                    
                    # determine which bin the angle falls into
                    bin_index = np.digitize(angle, ctx['angle_bins']) - 1
                    
                    # increment the count for the corresponding angle bin
                    ctx['angle_bin_distribution'][bin_index] += 1
                    
                    # construct the feature name for the specific parameter and bin
                    feature_name = f'{name}_param{i}_bin_{bin_index}'
                    
                    # increment the count for that binned feature
                    gate_counts[feature_name] += 1
                    
        # store the completed gate count dictionary for the current circuit
        ctx['per_circuit_gate_counts'][f'circuit_{idx}'] = gate_counts
        
        # aggregate the gate counts for the summary visualisation
        for gate, count in gate_counts.items():
            ctx['total_gate_counts'][gate] += count
    
        # get the number of qubits from the circuit
        num_qubits = circuit.num_qubits
        
        # create a distinct feature vector for each pauli-z observable
        for i in range(num_qubits):
            
            # copy the gate counts to create a base for the feature vector
            observable_features = gate_counts.copy()
            
            # add metadata for the current observable to the feature dictionary
            observable_features['circuit_index'] = idx
            observable_features['num_qubits'] = num_qubits
            observable_features['observable'] = f'Z{i}'
    
            # calculate the pauli-z expectation value from the noisy measurement counts
            noisy_exp_val = 0
            if noisy_counts:
                for bitstring, count in noisy_counts.items():
                    
                    # decrement the running sum for a '1' and increment for a '0'
                    if bitstring[num_qubits - 1 - i] == '1':
                        noisy_exp_val -= count
                    else:
                        noisy_exp_val += count
            
            # normalise the sum by the total number of shots
            observable_features['exp_val_noisy'] = noisy_exp_val / ctx['shots']
    
            # calculate the pauli-z expectation value from the ideal measurement counts
            ideal_exp_val = 0
            if ideal_counts:
                for bitstring, count in ideal_counts.items():
                    
                    # decrement the running sum for a '1' and increment for a '0'
                    if bitstring[num_qubits - 1 - i] == '1':
                        ideal_exp_val -= count
                    else:
                        ideal_exp_val += count
            
            # normalise the sum by the total number of shots
            observable_features['exp_val_ideal'] = ideal_exp_val / ctx['shots']
            
            # append the complete feature vector to the batch list
            batch_results.append(observable_features)

    return batch_results, batch_raw

# write one batch to disk
def save_batch(temp_path, prefix, batch_index, batch_results, batch_raw):
    
    # write engineered features to results file
    pd.DataFrame(batch_results).to_csv((temp_path / f'{prefix}_batch_{batch_index:03d}_results.csv').as_posix(), index=False)
    
    # write raw shot distributions to raw file
    pd.DataFrame(batch_raw).to_csv((temp_path / f'{prefix}_batch_{batch_index:03d}_raw.csv').as_posix(), index=False)
    
    return

# make or load the backup partition file (if it exists)
def load_partition(temp_path, circuits_df, fraction):
    
    # construct path to partition file
    partition_path = temp_path / 'partition.json'
    
    # load an existing partition if present
    if partition_path.exists():
        with open(partition_path, 'r') as f:
            part = json.load(f)
        hardware_indices = part['hardware_indices']
        simulator_indices = part['simulator_indices']
    else:
        seed = 12345
        
        # partition the circuits based on the pre-defined proportion of hardware
        hardware_indices = circuits_df.sample(frac=fraction, random_state=seed).index.tolist()
        simulator_indices = [i for i in circuits_df.index.tolist() if i not in hardware_indices]
        with open(partition_path, 'w') as f:
            json.dump({'fraction': fraction, 'seed': seed, 'hardware_indices': hardware_indices, 'simulator_indices': simulator_indices}, f)
    
    return hardware_indices, simulator_indices, partition_path

def main():
    
    # add the account details (removed for privacy reasonas) for all the proportions greater than 0
    if fraction > 0.0:
        ibm_platform_token = 'removed'
        ibm_platform_instance = 'removed'
        QiskitRuntimeService.save_account(channel='ibm_quantum_platform', token=ibm_platform_token, instance=ibm_platform_instance, overwrite=True, name='ibm_quantum_platform')

    # load the circuit definitions from the CSV file
    circuits_df = pd.read_csv('circuits.csv')
    all_non_param_gates, all_param_gates, feature_list = scan_gates(circuits_df, angle_bins)
    
    # instantiate backends
    ideal_backend = AerSimulator()
    noisy_backend_sim = FakeTorino()
    service = None
    noisy_backend = None
    if fraction > 0.0:
        service = QiskitRuntimeService(channel='ibm_quantum_platform', instance=ibm_platform_instance, token=ibm_platform_token)
        noisy_backend = service.backend('ibm_torino')
    
    # create a sampler bound to the hardware backend and set default shots
    sampler_noisy = None
    if fraction > 0.0:
        sampler_noisy = Sampler(mode=noisy_backend)
        sampler_noisy.options.default_shots = shots
    
        # ensure any mitigation is off
        sampler_noisy.options.twirling.enable_measure = False
        sampler_noisy.options.dynamical_decoupling.enable = False
    
    # create a directory for per-batch outputs
    temp_path = Path(f'temp_{int(fraction*100)}')
    temp_path.mkdir(parents=True, exist_ok=True)
    
    # set up containers and counters
    total_gate_counts = {feature: 0 for feature in feature_list}
    angle_bin_distribution = [0] * (len(angle_bins) - 1)
    per_circuit_gate_counts = {}
    processed_gate_count = 0
    
    # assign circuits to hardware or simulator (depending on whether partition file exists)
    hardware_indices, simulator_indices, partition_path = load_partition(temp_path, circuits_df, fraction)
    
    # store all runtime data and backends for batch processing
    circuits = {
        'circuits_df': circuits_df,
        'feature_list': feature_list,
        'all_non_param_gates': all_non_param_gates,
        'all_param_gates': all_param_gates,
        'noisy_backend': noisy_backend,
        'ideal_backend': ideal_backend,
        'noisy_backend_sim': noisy_backend_sim,
        'sampler_noisy': sampler_noisy,
        'shots': shots,
        'angle_bins': angle_bins,
        'temp_path': temp_path,
        'per_circuit_gate_counts': per_circuit_gate_counts,
        'processed_gate_count': processed_gate_count,
        'total_gate_counts': total_gate_counts,
        'angle_bin_distribution': angle_bin_distribution,
        'service': service,
        'partition_path': partition_path
    }
    
    # process the hardware partition in batches and save each batch as csv under temp
    if fraction > 0.0:
        hw_loop(circuits, hardware_indices, batch_size)
    
    # process the simulator partition in batches and save each batch as csv under temp
    sim_loop(circuits, simulator_indices, batch_size)
    
    # save feature-engineered results
    results_files = sorted(glob.glob((temp_path / '*_results.csv').as_posix()))
    raw_files = sorted(glob.glob((temp_path / '*_raw.csv').as_posix()))
    results_df = pd.concat((pd.read_csv(f) for f in results_files), ignore_index=True) if results_files else pd.DataFrame()
    results_df.to_csv(f'Data/engineered_results_{int(fraction*100)}.csv', index=False)
    print(f"Saved simulation results to Data/engineered_results_{int(fraction*100)}.csv")
    
    # rebuild aggregated gate counts from saved results in case batches were skipped
    for feature in feature_list:
        total_gate_counts[feature] = int(results_df[feature].sum()) if feature in results_df.columns else 0
      
    # save raw shot distributions without feature engineering
    raw_df = pd.concat((pd.read_csv(f) for f in raw_files), ignore_index=True) if raw_files else pd.DataFrame()
    raw_df.to_csv(f'Raw Data/raw_outputs_{int(fraction*100)}.csv', index=False)
    print(f"Saved raw outputs to Raw Data/raw_outputs_{int(fraction*100)}.csv")
    
    # separate the gate counts into parameterised and non-parameterised
    non_param_counts = {k: v for k, v in circuits['total_gate_counts'].items() if '_param' not in k}
    param_counts = {k: v for k, v in circuits['total_gate_counts'].items() if '_param' in k}
    
    # print the distributions
    print("\nDistribution of non-parameterised gate features:")
    print(json.dumps(non_param_counts, indent=4))
    print("\nDistribution of parameterised (binned) gate features:")
    print(json.dumps(param_counts, indent=4))
    
    # create a bar chart for non-parameterised gates
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    ax1.bar(non_param_counts.keys(), non_param_counts.values())
    ax1.set_ylabel('total count')
    ax1.set_title('distribution of non-parameterised gate features')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f'Figs/non_param_gate_distribution_{int(fraction*100)}.png', dpi=300)
    print(f"\nSaved non-parameterised gate distribution plot to Figs/non_param_gate_distribution_{int(fraction*100)}.png")
    
    # create a bar chart for parameterised gates
    fig2, ax2 = plt.subplots(figsize=(20, 8))
    ax2.bar(param_counts.keys(), param_counts.values())
    ax2.set_ylabel('total count')
    ax2.set_title('distribution of parameterised (binned) gate features')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f'Figs/param_gate_distribution_{int(fraction*100)}.png', dpi=300)
    print(f"\nSaved parameterised gate distribution plot to Figs/param_gate_distribution_{int(fraction*100)}.png")

if __name__ == "__main__":
    main()
