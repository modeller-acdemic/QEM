# -*- coding: utf-8 -*-
"""
Generates the random circuits with depths 2-9 and with a random length between 2-5.
Verifies distribution with a graph.

"""

from qiskit.circuit.random import random_circuit
from qiskit import qasm2
from random import randint, seed
import csv
from collections import Counter
import matplotlib.pyplot as plt

seed(42) # random seed

circuits_per_depth = 500
circuit_data = [['depth', 'qubits', 'qasm_definition']]

# iterate through the specified depths from 2 to 9
for depth in range(2, 10):
    
    # generate a set number of circuits for the current depth
    for i in range(circuits_per_depth):
    
        # select a random number of qubits (length) for this specific circuit
        num_qubits = randint(2, 5)
        
        # create the random circuit with the selected number of qubits and current depth
        circuit = random_circuit(num_qubits, depth, measure=True)
        
        # get a readable qasm string with natural line breaks for printing
        qasm_pretty = qasm2.dumps(circuit)
        
        # get a flattened qasm string for csv only
        qasm_csv = qasm_pretty.replace('\n', ' ')
        
        # print the circuit diagram
        # print(f"\ncircuit {i+1} ({num_qubits} qubits):")
        # print(circuit)
        
        # print the readable qasm with line breaks
        # print(f"\ndepth: {depth}, qubits: {num_qubits}\n{qasm_pretty}")
        
        # store the flattened qasm in the csv
        circuit_data.append([depth, num_qubits, qasm_csv])
        
# write the collected data to a csv file
with open('circuits.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(circuit_data)

print("\ncircuit definitions have been saved to circuits.csv")

# compute aggregates
qubits = [int(row[1]) for row in circuit_data[1:]]
depths = [int(row[0]) for row in circuit_data[1:]]

# print distribution of circuit lengths
qubit_counts = Counter(qubits)
print("\ndistribution of circuit lengths")
for l in sorted(qubit_counts):
    print(f"circuit length {l}: {qubit_counts[l]}")
    
# print distribution of depths - sanity check
depth_counts = Counter(depths)
print("\ndistribution of depths")
for d in sorted(depth_counts):
    print(f"depth {d}: {depth_counts[d]}")

# plot distribution of circuit lengths
plt.hist(qubits, bins=range(min(qubits), max(qubits) + 2), align='left', rwidth=0.9)
plt.xlabel("circuit length")
plt.ylabel("count")
plt.title("distribution of circuit lengths")
plt.xticks(range(min(qubits), max(qubits) + 1))
plt.savefig("Figs/length_distribution.png", dpi=300)
plt.show()

