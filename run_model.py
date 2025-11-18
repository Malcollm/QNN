import network
from network import Network
from qiskit.visualization import plot_histogram
from math import pi
import json

# loads the optimized parameter values
with open("params.json", "r") as f:
    data = json.load(f)

# creates QNN objects and loads parameters into it
net = Network(2, 2)
net.load_params(data)

# these data points are in the data set so this is just checking the error of the model
counts = net.pass_state([0, 0])
plot_histogram(counts, title="Test input-[0, 0] Expected: [1,1]")
print(f"Result: {network.find_key(counts)} Expected: [1,1]")
counts = net.pass_state([pi, pi])
plot_histogram(counts, title="Test input-[1, 1] Expected: [0, 0]")
print(f"Result: {network.find_key(counts)} Expected: [0, 0]")
counts = net.pass_state([0, pi])
plot_histogram(counts, title="Test input-[0, 1] Expected: [1, 0]")
print(f"Result: {network.find_key(counts)} Expected: [0, 1]") # bits are backwards

# this is data point the model has not seen the output put should be [0,1] if it follows the pattern
counts = net.pass_state([pi, 0])
plot_histogram(counts, title="Test input-[1, 0] Expected: [0, 1]")
print(f"Result: {network.find_key(counts)} Expected: [1, 0]") # bits are backwards

# shows the circuit
net.show()
