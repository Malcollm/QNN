from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit import ParameterVector
from qiskit_aer import AerSimulator
import matplotlib.pyplot as plt
from qiskit.quantum_info import Statevector
import numpy as np
import random
import copy

# returns the most frequent state in a dictionary of states
def find_key(states):
    keys_list = list(states.keys())
    largest = 0
    max_key = states[keys_list[0]]
    for key in keys_list:
        if states[key] > largest:
            largest = states[key]
            max_key = key
    return max_key, str(states[max_key])

# QNN class
class Network:
    def __init__(self, layers, q_bits):
        self.layers = layers
        self.q_bits = q_bits

        self.statevector = ""
        self.counts = {}
        self.probs = [0]*self.q_bits

        # sets all parameters to a random value [0,2pi]
        self.params = []
        self.param_vector = [[], [], []]
        for layer in range(layers):
            self.params.append([])
            self.param_vector[0].append(ParameterVector(f'thetaX-{layer}', self.q_bits))
            self.param_vector[1].append(ParameterVector(f'thetaY-{layer}', self.q_bits))
            self.param_vector[2].append(ParameterVector(f'thetaZ-{layer}', self.q_bits))
            for i in range(3):
                self.params[layer].append([])
                for n in range(q_bits):
                    self.params[layer][i].append(random.random()*np.pi*2)

        self.input_vector = ParameterVector('inputs', self.q_bits)

        # sets up registers and circuits
        self.qr = QuantumRegister(self.q_bits, 'q')
        self.cr = ClassicalRegister(self.q_bits, 'c')
        self.qc = QuantumCircuit(self.qr, self.cr)
        self.qcD = None

        # sets the initial state of the qubits to the input vector
        for i in range(self.q_bits):
            self.qc.initialize([1, 0], self.qr[i])
            self.qc.rx(self.input_vector[i], self.qr[i])

        self.qc.barrier()

        # adds 3 parameterized circuits (rx, ry, rz) to each qubit and entangles them with a CNOT gate
        # repeats for every layer
        for layer in range(self.layers):
            for n in range(self.q_bits):
                self.qc.rx(self.param_vector[0][layer][n], n)
                self.qc.ry(self.param_vector[1][layer][n], n)
                self.qc.rz(self.param_vector[2][layer][n], n)
            if self.q_bits > 1:
                self.qc.mcx(list(range(self.q_bits - 1)), self.q_bits - 1)
            self.qc.barrier()

        self.backend = AerSimulator()

    def load_params(self, params):
        self.params = params

    def pass_state(self, state):
        self.qcD = copy.deepcopy(self.qc)

        # sets input parameters
        self.qcD = self.qcD.assign_parameters(dict(zip(self.input_vector, state)))

        # sets rotational parameters
        for layer in range(self.layers):
            for i in range(3):
                self.qcD = self.qcD.assign_parameters(dict(zip(self.param_vector[i][layer], self.params[layer][i])))

        # creates string representation of the state vector
        state = Statevector(self.qcD).data.tolist()
        for i in range(len(state)):
            self.statevector += str(bin(i)) + " - " + str(state[i]) + "\n"

        # performs measurement
        self.qcD.measure(list(range(self.q_bits)), list(range(self.q_bits)))

        # runs simulation
        self.qcD = transpile(self.qcD, self.backend)
        result = self.backend.run(self.qcD, shots=1024).result()
        self.counts = result.get_counts(self.qcD)

        # calculates the probability of each qubit being in the 1 state
        for bitstring, freq in self.counts.items():
            bits = bitstring[::-1]
            for q in range(self.q_bits):
                if bits[q] == '1':
                    self.probs[q] += freq

        for i in range(self.q_bits):
            self.probs[i] /= 1025

        return self.counts

    def get_error(self, state):
        error = 0
        for i in range(self.q_bits):
            error += (state[i] - self.probs[i]) ** 2
        error /= self.q_bits
        return error

    def get_param(self, layer, n, r):
        return self.params[layer][r][n]

    def get_prob(self):
        return self.probs

    def get_shape(self):
        return self.layers, self.q_bits

    def set_param(self, layer, n, r, val):
        self.params[layer][r][n] = val

    def show(self):
        self.qcD.draw(output='mpl')
        plt.show()